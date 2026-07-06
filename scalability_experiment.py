# -*- coding: utf-8 -*-
"""Experimento de escalabilidad computacional: rollout vs RF-HH.

Genera instancias sintéticas del knapsack 0/1 con n en {250, 500, 1000, 2000}
(pesos y beneficios uniformes en [1, 256], capacidad escalada C = round(256*n/250))
y mide el tiempo medio por instancia de la política rollout y del RF-HH.
Estas instancias se usan EXCLUSIVAMENTE para medir tiempo de ejecución,
no calidad de solución (el benchmark solo contiene n = 250).

Uso:
    python scalability_experiment.py [--sizes 250 500 1000 2000]
                                     [--eval-instances 30]
                                     [--train-instances 200]
                                     [--seed 42] [--outdir figures]

Reproduce la figura de escalabilidad (figures/scalability.png) del artículo.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rollout_hh_knapsack import (
    CONFIDENCE_THRESHOLD,
    FEATURE_NAMES,
    KnapsackState,
    SEED,
    rollout_choice,
    solve_rf_hh,
    solve_rollout,
    step_features,
    train_rf,
)

FIG_KW = dict(dpi=300, bbox_inches="tight")


# ----------------------------------------------------------------------
# Generación de instancias sintéticas
# ----------------------------------------------------------------------
def generate_instance(n_items: int, seed: int) -> KnapsackState:
    """Genera una instancia sintética del knapsack 0/1.

    Pesos y beneficios enteros uniformes en [1, 256]; capacidad escalada
    proporcionalmente al benchmark de n = 250: C = round(256 * n / 250).
    """
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 257, size=n_items)
    profits = rng.integers(1, 257, size=n_items)
    capacity = round(256 * n_items / 250)
    return KnapsackState(weights, profits, capacity)


def build_rollout_dataset_for_n(n_items: int, samples: int,
                                seed: int = SEED) -> pd.DataFrame:
    """Construye el dataset etiquetado por el rollout para instancias de tamaño n.

    Genera `samples` instancias sintéticas y, en cada paso de su solución,
    registra las 8 features de estado junto con la heurística elegida por el
    oráculo rollout.
    """
    rows = []
    for i in range(samples):
        state = generate_instance(n_items, seed + i)
        while state.remaining.any():
            feats = step_features(state)
            h = rollout_choice(state)
            rows.append(dict(zip(FEATURE_NAMES, feats), BEST_H=h))
            item = state.next_item(h)
            if item == -1:
                break
            state.pack(item)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Experimento
# ----------------------------------------------------------------------
def run(sizes, eval_instances, train_instances, seed, outdir):
    os.makedirs(outdir, exist_ok=True)
    results = []

    for n in sizes:
        print(f"\n--- n = {n} ---")

        print(f"  1/3 Dataset rollout ({train_instances} instancias, semilla {seed})...")
        t0 = time.perf_counter()
        df_train = build_rollout_dataset_for_n(n, train_instances, seed)
        print(f"      {len(df_train)} transiciones en {time.perf_counter() - t0:.1f}s")

        print("  2/3 Entrenando Random Forest...")
        model = train_rf(df_train)

        print(f"  3/3 Midiendo tiempos en {eval_instances} instancias de prueba...")
        rollout_times, rfhh_times = [], []
        for i in range(eval_instances):
            # Semilla desplazada para no reutilizar instancias de entrenamiento.
            inst_seed = seed + train_instances + i

            state = generate_instance(n, inst_seed)
            t0 = time.perf_counter()
            solve_rollout(state)
            rollout_times.append(time.perf_counter() - t0)

            state = generate_instance(n, inst_seed)
            t0 = time.perf_counter()
            solve_rf_hh(state, model, CONFIDENCE_THRESHOLD)
            rfhh_times.append(time.perf_counter() - t0)

        row = dict(
            n=n,
            rollout_mean=np.mean(rollout_times),
            rollout_std=np.std(rollout_times),
            rfhh_mean=np.mean(rfhh_times),
            rfhh_std=np.std(rfhh_times),
        )
        results.append(row)
        print(f"      Rollout: {row['rollout_mean']:.4f}s ± {row['rollout_std']:.4f}s")
        print(f"      RF-HH:   {row['rfhh_mean']:.4f}s ± {row['rfhh_std']:.4f}s")

    df = pd.DataFrame(results)
    print("\n--- Resultados ---")
    print(df.to_string(index=False))
    csv_path = os.path.join(outdir, "scalability.csv")
    df.to_csv(csv_path, index=False)
    print(f"Tabla guardada en {csv_path}")

    # ---- Figura log-log --------------------------------------------------
    plt.figure(figsize=(7, 5))
    plt.errorbar(df["n"], df["rollout_mean"], yerr=df["rollout_std"],
                 label="Rollout", marker="o", capsize=4)
    plt.errorbar(df["n"], df["rfhh_mean"], yerr=df["rfhh_std"],
                 label="RF-HH", marker="s", capsize=4)
    plt.xlabel("Number of items (n)")
    plt.ylabel("Average time per instance (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(df["n"], labels=[str(v) for v in df["n"]])
    plt.legend()
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    fig_path = os.path.join(outdir, "scalability.png")
    plt.savefig(fig_path, **FIG_KW)
    print(f"Figura guardada en {fig_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[250, 500, 1000, 2000])
    ap.add_argument("--eval-instances", type=int, default=30)
    ap.add_argument("--train-instances", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="figures")
    args = ap.parse_args()

    np.random.seed(args.seed)
    run(args.sizes, args.eval_instances, args.train_instances,
        args.seed, args.outdir)


if __name__ == "__main__":
    main()
