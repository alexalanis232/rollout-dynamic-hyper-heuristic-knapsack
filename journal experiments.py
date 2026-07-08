# -*- coding: utf-8 -*-
"""Experimentos para la versión de revista.

Pipeline automatizado: para repetirlo con otro conjunto de datos, solo cambia
las rutas --train y --test y vuelve a correr. Todo lo demás (subgrupos,
etiquetas, clasificadores, tablas) se deriva automáticamente de los datos.

Experimentos:
  E1. Heurísticas base sobre el test set, desglosadas por subgrupo.
  E2. Selección ESTÁTICA: 3 clasificadores (RF, kNN, MLP) aprenden
      estado inicial -> mejor heurística, y resuelven toda la instancia
      con la heurística predicha.
  E3. Rollout por subgrupo: para cada heurística h, las instancias de
      entrenamiento de su subgrupo se resuelven con rollout usando h como
      política de completación (base), generando 4 HHs de rollout y 4
      datasets etiquetados.
  E4. Selección DINÁMICA por subgrupo: 3 clasificadores por cada uno de los
      4 datasets de E3 (12 en total), evaluados por subgrupo y global.

Uso:
    python journal_experiments.py --train Instances/Train-250-256 \
                                  --test Instances/Test-250-256 \
                                  --outdir journal_results --seed 42
"""

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rollout_hh_knapsack import (
    CONFIDENCE_THRESHOLD,
    FEATURE_NAMES,
    HEURISTICS,
    KnapsackState,
    SEED,
    instance_category,
    list_instances,
    step_features,
)

H_TO_IDX = {h: i for i, h in enumerate(HEURISTICS)}


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------
def load_state(folder: str, fname: str) -> KnapsackState:
    return KnapsackState.from_file(os.path.join(folder, fname))


def solve_with(state: KnapsackState, h: str) -> int:
    """Resuelve la instancia completa con una sola heurística.

    Este es el comportamiento "clásico": una heurística fija de principio a
    fin. Se usa en E1 (evaluar cada heurística sola) y en E2 (resolver con
    la heurística que predijo el clasificador estático).
    """
    while state.remaining.any(): # Mientras queden ítems sin empacar.
        item = state.next_item(h)
        if item == -1:
            break
        state.pack(item)
    return state.profit


def completion_profit(state: KnapsackState, h: str) -> int:
    """Beneficio total si el estado actual se completa greedy con la heurística h.

    Simula la completación y restaura el estado (no lo modifica).
    """
    # Guarda una "foto" del estado para poder restaurarlo después de simular.
    saved_rem = state.remaining.copy()
    saved_cap, saved_prof = state.capacity, state.profit
    # Empaca greedy con la heurística h hasta que ya no quepa nada.
    while True:
        item = state.next_item(h)
        if item == -1: # -1 significa que ningún ítem restante cabe.
            break
        state.pack(item)
    total = state.profit # Beneficio total de esta simulación.
    # Restaura la foto: el estado queda exactamente como estaba antes.
    state.remaining = saved_rem
    state.capacity = saved_cap
    state.profit = saved_prof
    return total


def rollout_choice_g(state: KnapsackState, base: str) -> str:
    """Rollout genérico: empaca un ítem con cada heurística y completa con `base`.

    A diferencia del rollout del módulo (que siempre completa con MAXPW),
    aquí la política de completación es la heurística base del subgrupo.
    Los empates se etiquetan con la base.
    """
    best_h, best_profit, base_profit = base, -1, -1
    for h in HEURISTICS:
        # 1. ¿Qué ítem elegiría la heurística h en este estado?
        item = state.next_item(h)
        if item == -1: # h no puede empacar nada, se salta.
            continue
        # 2. Empaca ese ítem (simulación) y completa el resto con la base.
        state.pack(item)
        total = completion_profit(state, base)
        # 3. Deshace el empaque simulado (restaura el ítem).
        state.remaining[item] = True
        state.capacity += int(state.weights[item])
        state.profit -= int(state.profits[item])
        # 4. Registra el beneficio de la base y actualiza al mejor.
        if h == base:
            base_profit = total
        if total > best_profit:
            best_profit, best_h = total, h
    # Desempate: si la base logró el mismo beneficio que la mejor,
    # se etiqueta con la base (evita etiquetas espurias por empates).
    if base_profit == best_profit:
        return base
    return best_h


def solve_rollout_g(state: KnapsackState, base: str) -> int:
    """Resuelve una instancia con el rollout genérico (completación = base)."""
    while state.remaining.any():
        h = rollout_choice_g(state, base)
        item = state.next_item(h)
        if item == -1:
            break
        state.pack(item)
    return state.profit


def subgroup_of(fname: str) -> str:
    """Heurística generadora del subgrupo (DEF, MAXP, MAXPW o MINW)."""
    return instance_category(fname).split("_")[0]


def make_classifiers(seed: int) -> dict:
    """Los tres clasificadores usados en E2 y E4."""
    return {
        "RF": RandomForestClassifier(n_estimators=300, min_samples_leaf=10,
                                     random_state=seed, n_jobs=1),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                          random_state=seed)),
    }


def solve_dynamic_batch(states: list[KnapsackState], model,
                        base: str = "MAXPW",
                        threshold: float = CONFIDENCE_THRESHOLD) -> list[int]:
    """Resuelve instancias con un clasificador dinámico genérico (por lotes).

    En cada paso predice la heurística con el clasificador; si la confianza
    es menor al umbral, usa la heurística `base` como respaldo. Funciona con
    cualquier clasificador de scikit-learn con predict_proba.
    """
    base_idx = H_TO_IDX[base]
    classes = list(model.classes_)
    active = [i for i, s in enumerate(states) if s.remaining.any()]
    while active:
        feats = np.array([step_features(states[i]) for i in active])
        probas = model.predict_proba(feats)
        still = []
        for row, i in enumerate(active):
            s = states[i]
            k = int(np.argmax(probas[row]))
            h_idx = classes[k]
            if h_idx != base_idx and probas[row][k] < threshold:
                h_idx = base_idx
            item = s.next_item(HEURISTICS[h_idx])
            if item == -1:
                continue
            s.pack(item)
            if s.remaining.any():
                still.append(i)
        active = still
    return [s.profit for s in states]


def summarize(df: pd.DataFrame, method_cols: list[str]) -> pd.DataFrame:
    """Media por subgrupo y global para las columnas de métodos."""
    per_group = df.groupby("Subgroup")[method_cols].mean()
    per_group.loc["GLOBAL"] = df[method_cols].mean()
    return per_group.round(2)


# ----------------------------------------------------------------------
# Experimentos
# ----------------------------------------------------------------------
def exp1_heuristics(test_dir: str, test_files: list[str]) -> pd.DataFrame:
    """E1: heurísticas base por subgrupo del test set."""
    rows = []
    for f in test_files:
        row = {"Filename": f, "Subgroup": subgroup_of(f)}
        for h in HEURISTICS:
            row[h] = solve_with(load_state(test_dir, f), h)
        rows.append(row)
    return pd.DataFrame(rows)


def exp2_static(train_dir, train_files, test_dir, test_files, df_e1, seed):
    """E2: clasificadores estáticos (estado inicial -> heurística fija)."""
    # Para cada instancia de entrenamiento:
    #   - X: las 8 features calculadas en el ESTADO INICIAL (antes de empacar nada)
    #   - y: la mejor heurística REAL, que se obtiene resolviendo la instancia
    #        completa con cada una de las 4 y quedándose con la de mayor beneficio.
    X_tr, y_tr = [], []
    for f in train_files:
        X_tr.append(step_features(load_state(train_dir, f)))
        profits = [solve_with(load_state(train_dir, f), h) for h in HEURISTICS]
        y_tr.append(int(np.argmax(profits))) # Índice de la heurística ganadora.
    X_tr, y_tr = np.array(X_tr), np.array(y_tr)

    X_te = np.array([step_features(load_state(test_dir, f))
                     for f in test_files])
    y_te = df_e1[HEURISTICS].values.argmax(axis=1)  # mejor heurística real

    results = df_e1[["Filename", "Subgroup"]].copy()
    accuracies = {}
    for name, clf in make_classifiers(seed).items():
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        accuracies[name] = float((preds == y_te).mean())
        col = f"Static-{name}"
        results[col] = [solve_with(load_state(test_dir, f), HEURISTICS[k])
                        for f, k in zip(test_files, preds)]
    return results, accuracies


def exp3_rollout_per_subgroup(train_dir, train_files, test_dir, test_files):
    """E3: rollout con base = heurística del subgrupo; genera 4 datasets."""
    datasets = {}
    for h in HEURISTICS:
        rows = []
        # Solo las instancias de entrenamiento del subgrupo de h
        # (ej. para h=DEF, las que empiezan con DEF_ en el nombre).
        subset = [f for f in train_files if subgroup_of(f) == h]
        for f in subset:
            state = load_state(train_dir, f)
            while state.remaining.any():
                # En cada paso: guarda las features + la heurística que
                # eligió el rollout (completando con h, no con MAXPW).
                feats = step_features(state)
                choice = rollout_choice_g(state, base=h)
                rows.append(dict(zip(FEATURE_NAMES, feats), BEST_H=choice))
                # Empaca de verdad el ítem elegido y avanza al siguiente estado.
                item = state.next_item(choice)
                if item == -1:
                    break
                state.pack(item)
        datasets[h] = pd.DataFrame(rows) # Un dataset etiquetado por subgrupo.

    # Evalúa cada rollout (base=h) en TODO el test set.
    results = pd.DataFrame({"Filename": test_files,
                            "Subgroup": [subgroup_of(f) for f in test_files]})
    for h in HEURISTICS:
        results[f"Rollout-{h}"] = [
            solve_rollout_g(load_state(test_dir, f), base=h)
            for f in test_files]
    return datasets, results


def exp4_dynamic_per_subgroup(datasets, test_dir, test_files, seed):
    """E4: 3 clasificadores por dataset de E3 (12 en total)."""
    results = pd.DataFrame({"Filename": test_files,
                            "Subgroup": [subgroup_of(f) for f in test_files]})
    # Por cada uno de los 4 datasets de E3 se entrenan 3 clasificadores
    # (RF, kNN y MLP), dando 12 hyper-heurísticas dinámicas en total.
    for h, df_h in datasets.items():
        X = df_h[FEATURE_NAMES].values                # Las 8 features por paso.
        y = df_h["BEST_H"].map(H_TO_IDX).values       # La heurística etiquetada.
        for name, clf in make_classifiers(seed).items():
            clf.fit(X, y) # Entrenamiento offline (una sola vez).
            # Evalúa en TODO el test set: en cada paso el clasificador
            # predice la heurística; si su confianza es baja, usa h (la base
            # de su subgrupo) como respaldo, no MAXPW.
            states = [load_state(test_dir, f) for f in test_files]
            results[f"{name}-HH-{h}"] = solve_dynamic_batch(states, clf,
                                                            base=h)
    return results


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--outdir", default="journal_results")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    train_files = list_instances(args.train)
    test_files = list_instances(args.test)
    print(f"Train: {len(train_files)} instancias | Test: {len(test_files)}")

    print("\n[E1] Heurísticas base por subgrupo...")
    df_e1 = exp1_heuristics(args.test, test_files)
    t1 = summarize(df_e1, HEURISTICS)
    print(t1.to_string())
    df_e1.to_csv(f"{args.outdir}/e1_heuristics_per_instance.csv", index=False)
    t1.to_csv(f"{args.outdir}/e1_heuristics_summary.csv")

    print("\n[E2] Clasificadores estáticos (estado inicial -> heurística)...")
    df_e2, accs = exp2_static(args.train, train_files, args.test, test_files,
                              df_e1, args.seed)
    cols2 = [c for c in df_e2.columns if c.startswith("Static-")]
    t2 = summarize(df_e2, cols2)
    print(t2.to_string())
    print("Accuracy vs mejor heurística real:",
          {k: f"{v:.1%}" for k, v in accs.items()})
    df_e2.to_csv(f"{args.outdir}/e2_static_per_instance.csv", index=False)
    t2.to_csv(f"{args.outdir}/e2_static_summary.csv")

    print("\n[E3] Rollout por subgrupo (base = heurística del subgrupo)...")
    datasets, df_e3 = exp3_rollout_per_subgroup(args.train, train_files,
                                                args.test, test_files)
    for h, d in datasets.items():
        d.to_csv(f"{args.outdir}/e3_dataset_{h}.csv", index=False)
        print(f"  Dataset {h}: {len(d)} transiciones")
    cols3 = [f"Rollout-{h}" for h in HEURISTICS]
    t3 = summarize(df_e3, cols3)
    print(t3.to_string())
    df_e3.to_csv(f"{args.outdir}/e3_rollout_per_instance.csv", index=False)
    t3.to_csv(f"{args.outdir}/e3_rollout_summary.csv")

    print("\n[E4] Clasificadores dinámicos por subgrupo (12 en total)...")
    df_e4 = exp4_dynamic_per_subgroup(datasets, args.test, test_files,
                                      args.seed)
    cols4 = [c for c in df_e4.columns if "-HH-" in c]
    t4 = summarize(df_e4, cols4)
    print(t4.to_string())
    df_e4.to_csv(f"{args.outdir}/e4_dynamic_per_instance.csv", index=False)
    t4.to_csv(f"{args.outdir}/e4_dynamic_summary.csv")

    print(f"\nTodo guardado en {args.outdir}/")


if __name__ == "__main__":
    main()
