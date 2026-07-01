"""
Rollout-Based Dynamic Hyper-heuristics for the Knapsack Problem
================================================================
Clean, reproducible pipeline matching the paper:

  1. Constructive heuristics: DEF, MAXP, MAXPW, MINW
  2. Rollout policy (one-step heuristic selection, MAXPW completion)
  3. Eight dynamic state features
  4. Random Forest hyper-heuristic (behavioral cloning of the rollout
     oracle) with a confidence-gated fallback to MAXPW
  5. Evaluation on the test set: mean profits, Wilcoxon signed-rank
     test, win rate, per-instance timing, and all paper figures

Usage:
    python rollout_hh_knapsack.py \
        --train Instances/Train-250-256 \
        --test  Instances/Test-250-256 \
        --samples 200 --seed 42

Instance file format (.kp):
    line 1:  nbItems, capacity
    lines:   weight, profit
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
HEURISTICS = ["DEF", "MAXP", "MAXPW", "MINW"]
H_TO_IDX = {h: i for i, h in enumerate(HEURISTICS)}
MAXPW_IDX = H_TO_IDX["MAXPW"]

FEATURE_NAMES = [
    "mean_ratio_norm", "max_ratio_pack", "std_ratio_pack",
    "frac_packable", "tightness", "fill", "slack_per_item",
    "frac_items_left",
]

SEED = 42
CONFIDENCE_THRESHOLD = 0.6
N_ESTIMATORS = 300
MIN_SAMPLES_LEAF = 10
FIG_KW = dict(dpi=300, bbox_inches="tight")


# ----------------------------------------------------------------------
# Problem state (array-based, no deepcopy needed)
# ----------------------------------------------------------------------
class KnapsackState:
    """Array-based 0/1 knapsack state.

    Items are stored once as numpy arrays; the solving process only
    mutates `remaining` (boolean mask) and `capacity`, so simulating a
    rollout is a cheap save/restore of those two values instead of a
    deepcopy of the whole object.
    """

    __slots__ = ("weights", "profits", "ratios", "ratio_order",
                 "remaining", "capacity", "initial_capacity",
                 "n_items", "profit")

    def __init__(self, weights, profits, capacity):
        self.weights = np.asarray(weights, dtype=np.int64)
        self.profits = np.asarray(profits, dtype=np.int64)
        self.ratios = self.profits / self.weights
        # Stable sort so ties break by original index, matching the
        # first-strictly-greater rule of the original implementation.
        self.ratio_order = np.argsort(-self.ratios, kind="stable")
        self.n_items = len(self.weights)
        self.remaining = np.ones(self.n_items, dtype=bool)
        self.capacity = int(capacity)
        self.initial_capacity = int(capacity)
        self.profit = 0

    # -- I/O -----------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "KnapsackState":
        with open(path) as f:
            lines = f.readlines()
        n_items, capacity = (int(x.strip()) for x in lines[0].split(","))
        weights, profits = [], []
        for i in range(n_items):
            w, p = lines[i + 1].split(",")
            weights.append(int(w.strip()))
            profits.append(int(float(p.strip())))
        return cls(weights, profits, capacity)

    # -- Heuristic item selection ---------------------------------------
    def next_item(self, heuristic: str) -> int:
        """Index of the item the heuristic would pack next, or -1."""
        fits = self.remaining & (self.weights <= self.capacity)
        if not fits.any():
            return -1
        idx = np.flatnonzero(fits)
        if heuristic == "DEF":
            return int(idx[0])
        if heuristic == "MAXP":
            return int(idx[np.argmax(self.profits[idx])])
        if heuristic == "MAXPW":
            return int(idx[np.argmax(self.ratios[idx])])
        if heuristic == "MINW":
            return int(idx[np.argmin(self.weights[idx])])
        raise ValueError(f"Unknown heuristic: {heuristic}")

    def pack(self, item: int) -> None:
        self.remaining[item] = False
        self.capacity -= int(self.weights[item])
        self.profit += int(self.profits[item])

    # -- Fast MAXPW completion (used by the rollout) ---------------------
    def maxpw_completion_profit(self) -> int:
        """Profit after greedily completing with MAXPW from this state.

        Iterating the pre-sorted ratio order and packing whatever fits
        is equivalent to repeatedly selecting the best-ratio packable
        item; the state itself is not modified.
        """
        cap = self.capacity
        extra = 0
        w, p, rem = self.weights, self.profits, self.remaining
        for j in self.ratio_order:
            if rem[j] and w[j] <= cap:
                cap -= w[j]
                extra += p[j]
        return self.profit + extra

    # -- Solvers ---------------------------------------------------------
    def solve_static(self, heuristic: str) -> int:
        item = self.next_item(heuristic)
        while item != -1:
            self.pack(item)
            item = self.next_item(heuristic)
        return self.profit


# ----------------------------------------------------------------------
# Rollout policy
# ----------------------------------------------------------------------
def rollout_choice(state: KnapsackState, base: str = "MAXPW") -> str:
    """Best heuristic for the current step according to the rollout.

    Each heuristic packs one item; the remaining items are completed
    with the base policy (MAXPW). Since MAXPW is among the candidates,
    the rollout profit is never below the pure-MAXPW profit
    (sequential improvement property, Bertsekas et al. 1997).
    """
    best_h, best_profit = base, -1
    for h in HEURISTICS:
        item = state.next_item(h)
        if item == -1:
            continue
        # simulate: pack item, evaluate completion, restore
        state.pack(item)
        total = state.maxpw_completion_profit()
        state.remaining[item] = True
        state.capacity += int(state.weights[item])
        state.profit -= int(state.profits[item])
        if total > best_profit:
            best_profit, best_h = total, h
    return best_h


def solve_rollout(state: KnapsackState, base: str = "MAXPW") -> int:
    while state.remaining.any():
        h = rollout_choice(state, base)
        item = state.next_item(h)
        if item == -1:
            break
        state.pack(item)
    return state.profit


# ----------------------------------------------------------------------
# Dynamic state features (Section 3.2 of the paper)
# ----------------------------------------------------------------------
def step_features(state: KnapsackState) -> list[float]:
    rem = state.remaining
    if not rem.any():
        return [0.0] * len(FEATURE_NAMES)
    w = state.weights[rem].astype(float)
    p = state.profits[rem].astype(float)
    r = p / w
    rmax = r.max() if r.max() > 0 else 1.0
    n = len(w)
    cap = state.capacity

    fits = w <= cap
    if fits.any():
        pr = r[fits]
        max_ratio_pack = pr.max() / rmax
        std_ratio_pack = pr.std() / rmax
        frac_packable = fits.sum() / n
        mean_w_pack = w[fits].mean()
    else:
        max_ratio_pack = std_ratio_pack = frac_packable = 0.0
        mean_w_pack = 1.0

    tightness = cap / w.sum() if w.sum() > 0 else 0.0
    fill = 1.0 - cap / state.initial_capacity
    slack_per_item = (cap / mean_w_pack) / n if mean_w_pack > 0 else 0.0
    return [r.mean() / rmax, max_ratio_pack, std_ratio_pack,
            frac_packable, tightness, fill, slack_per_item,
            n / state.n_items]


# ----------------------------------------------------------------------
# Dataset building (rollout as labeling oracle)
# ----------------------------------------------------------------------
def list_instances(folder: str) -> list[str]:
    return sorted(f for f in os.listdir(folder) if f.endswith(".kp"))


def build_rollout_dataset(folder: str, samples: int | None = None,
                          seed: int = SEED,
                          base: str = "MAXPW") -> pd.DataFrame:
    """Label each visited state with the rollout-selected heuristic.

    Instances are shuffled with a fixed seed before sub-sampling so the
    training subset is not biased by filename ordering.
    """
    files = list_instances(folder)
    if samples is not None and samples < len(files):
        random.Random(seed).shuffle(files)
        files = files[:samples]

    rows = []
    for fname in files:
        state = KnapsackState.from_file(os.path.join(folder, fname))
        while state.remaining.any():
            feats = step_features(state)
            h = rollout_choice(state, base)
            rows.append(dict(zip(FEATURE_NAMES, feats), BEST_H=h))
            item = state.next_item(h)
            if item == -1:
                break
            state.pack(item)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Random Forest hyper-heuristic
# ----------------------------------------------------------------------
def train_rf(df: pd.DataFrame) -> RandomForestClassifier:
    X = df[FEATURE_NAMES].values
    y = df["BEST_H"].map(H_TO_IDX).values
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def solve_rf_hh(state: KnapsackState, model: RandomForestClassifier,
                threshold: float = CONFIDENCE_THRESHOLD) -> int:
    """RF-HH solver with confidence-gated fallback to MAXPW.

    When the classifier's top-class probability is below `threshold`
    the safe default MAXPW is used instead. This guards against
    low-confidence predictions in states poorly covered by training.
    """
    classes = list(model.classes_)
    while state.remaining.any():
        feats = np.array([step_features(state)])
        proba = model.predict_proba(feats)[0]
        k = int(np.argmax(proba))
        h_idx = classes[k]
        if h_idx != MAXPW_IDX and proba[k] < threshold:
            h_idx = MAXPW_IDX
        item = state.next_item(HEURISTICS[h_idx])
        if item == -1:
            break
        state.pack(item)
    return state.profit


def solve_rf_hh_batch(states: list[KnapsackState],
                      model: RandomForestClassifier,
                      threshold: float = CONFIDENCE_THRESHOLD) -> list[int]:
    """Solve many instances in lockstep with one predict_proba per step.

    Produces exactly the same solutions as `solve_rf_hh`; batching only
    amortizes the per-call overhead of the scikit-learn predictor
    (a single-row call costs almost the same as a several-hundred-row
    call), so throughput improves by orders of magnitude.
    """
    classes = list(model.classes_)
    active = [i for i, s in enumerate(states) if s.remaining.any()]
    while active:
        feats = np.array([step_features(states[i]) for i in active])
        probas = model.predict_proba(feats)
        still_active = []
        for row, i in enumerate(active):
            s = states[i]
            proba = probas[row]
            k = int(np.argmax(proba))
            h_idx = classes[k]
            if h_idx != MAXPW_IDX and proba[k] < threshold:
                h_idx = MAXPW_IDX
            item = s.next_item(HEURISTICS[h_idx])
            if item == -1:
                continue
            s.pack(item)
            if s.remaining.any():
                still_active.append(i)
        active = still_active
    return [s.profit for s in states]


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def instance_category(filename: str) -> str:
    for h in HEURISTICS:
        for d in ("EASY", "HARD"):
            if f"{h}_{d}" in filename:
                return f"{h}_{d}"
    return "OTHER"


def evaluate(test_folder: str, model: RandomForestClassifier,
             outdir: str) -> pd.DataFrame:
    files = list_instances(test_folder)
    rows, rollout_times = [], []
    oracle_first, rf_first = [], []  # first-step labels for the confusion matrix

    for fname in files:
        path = os.path.join(test_folder, fname)
        row = {"Filename": fname,
               "Category": "HARD" if "HARD" in fname else "EASY",
               "Type": instance_category(fname)}

        for h in HEURISTICS:
            row[h] = KnapsackState.from_file(path).solve_static(h)

        state = KnapsackState.from_file(path)
        t0 = time.perf_counter()
        row["Rollout"] = solve_rollout(state)
        rollout_times.append(time.perf_counter() - t0)

        state = KnapsackState.from_file(path)
        oracle_first.append(H_TO_IDX[rollout_choice(state)])
        rf_first.append(int(model.predict(
            np.array([step_features(state)]))[0]))

        rows.append(row)

    # RF-HH: batched evaluation (identical solutions, amortized overhead)
    states = [KnapsackState.from_file(os.path.join(test_folder, f))
              for f in files]
    t0 = time.perf_counter()
    rf_profits = solve_rf_hh_batch(states, model)
    rf_total = time.perf_counter() - t0
    for row, prof in zip(rows, rf_profits):
        row["RF-HH"] = prof

    df = pd.DataFrame(rows)

    # ---- summary -------------------------------------------------------
    methods = HEURISTICS + ["Rollout", "RF-HH"]
    print("\n=== Mean profit (N={}) ===".format(len(df)))
    print(df[methods].mean().round(2).to_string())

    stat, p = wilcoxon(df["Rollout"], df["MAXPW"])
    wins = int((df["Rollout"] > df["MAXPW"]).sum())
    print(f"\nWilcoxon Rollout vs MAXPW: p = {p:.3e}, "
          f"wins {wins}/{len(df)} ({100 * wins / len(df):.1f}%)")

    stat2, p2 = wilcoxon(df["RF-HH"], df["MAXPW"])
    wins2 = int((df["RF-HH"] > df["MAXPW"]).sum())
    print(f"Wilcoxon RF-HH  vs MAXPW: p = {p2:.3e}, "
          f"wins {wins2}/{len(df)} ({100 * wins2 / len(df):.1f}%)")

    print(f"\nMean time/instance: rollout {np.mean(rollout_times):.4f}s, "
          f"RF-HH (batched) {rf_total / len(df):.4f}s")
    ratio = df['RF-HH'].mean() / df['Rollout'].mean()
    print(f"RF-HH captures {100 * ratio:.1f}% of the rollout profit")

    make_figures(df, model, oracle_first, rf_first, outdir)
    return df


# ----------------------------------------------------------------------
# Figures (all saved at 300 dpi into `outdir`)
# ----------------------------------------------------------------------
def make_figures(df, model, oracle_first, rf_first, outdir):
    os.makedirs(outdir, exist_ok=True)
    methods = HEURISTICS + ["Rollout", "RF-HH"]
    sns.set_style("whitegrid")

    # Fig 2: boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[methods], palette="viridis")
    plt.xlabel("Method")
    plt.ylabel("Profit")
    plt.savefig(os.path.join(outdir, "boxplot.png"), **FIG_KW)
    plt.close()

    # Fig 3: scatter MAXPW vs Rollout
    plt.figure(figsize=(8, 8))
    lims = [min(df["MAXPW"].min(), df["Rollout"].min()),
            max(df["MAXPW"].max(), df["Rollout"].max())]
    plt.scatter(df["MAXPW"], df["Rollout"], alpha=0.5, color="#2a788e")
    plt.plot(lims, lims, "r--", alpha=0.75, label="x = y")
    plt.xlabel("MAXPW profit")
    plt.ylabel("Rollout profit")
    plt.legend()
    plt.savefig(os.path.join(outdir, "scatter.png"), **FIG_KW)
    plt.close()

    # Fig 4: profit distributions
    plt.figure(figsize=(10, 6))
    sns.kdeplot(df["MAXPW"], fill=True, label="MAXPW", color="blue")
    sns.kdeplot(df["RF-HH"], fill=True, label="RF-HH", color="green")
    plt.xlabel("Profit")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig(os.path.join(outdir, "distribution.png"), **FIG_KW)
    plt.close()

    # Fig 5: heatmap of first-step oracle choices per instance type
    hm = pd.DataFrame({
        "Type": df["Type"],
        "Heuristic": [HEURISTICS[i] for i in oracle_first],
    })
    counts = hm.groupby(["Type", "Heuristic"]).size().unstack(fill_value=0)
    plt.figure(figsize=(10, 6))
    sns.heatmap(counts, annot=True, fmt="d", cmap="YlGnBu")
    plt.xlabel("Heuristic selected by the rollout")
    plt.ylabel("Instance type")
    plt.savefig(os.path.join(outdir, "heatmap.png"), **FIG_KW)
    plt.close()

    # Fig 6: accumulated-profit trajectory on one representative instance
    # (re-solve the first instance step by step)
    # left to the notebook if needed; omitted here for brevity of the CLI.

    # Fig 7: feature importances
    imp = model.feature_importances_
    order = np.argsort(imp)
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(order)), imp[order], color="#2a788e")
    plt.yticks(range(len(order)), [FEATURE_NAMES[i] for i in order])
    plt.xlabel("Relative importance")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "feature_importance.png"), **FIG_KW)
    plt.close()

    # Fig 8: confusion matrix (first-step decisions on the test set)
    labels = np.unique(np.concatenate([oracle_first, rf_first]))
    cm = confusion_matrix(oracle_first, rf_first, labels=labels)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=[HEURISTICS[i] for i in labels])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.savefig(os.path.join(outdir, "confusion_matrix.png"), **FIG_KW)
    plt.close()

    # Fig 9: performance by difficulty
    diff = (df.groupby("Category")[["MAXPW", "Rollout", "RF-HH"]]
              .mean().reset_index()
              .melt(id_vars="Category", var_name="Method",
                    value_name="Mean Profit"))
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Category", y="Mean Profit", hue="Method",
                data=diff, palette="viridis")
    plt.savefig(os.path.join(outdir, "difficulty_comparison.png"), **FIG_KW)
    plt.close()

    print(f"Figures written to {outdir}/")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", required=True, help="training instances folder")
    ap.add_argument("--test", required=True, help="test instances folder")
    ap.add_argument("--samples", type=int, default=200,
                    help="training instances to label with the rollout")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--results-csv", default="results.csv")
    args = ap.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    print(f"1/3 Building rollout-labeled dataset "
          f"({args.samples} instances, seed {args.seed})...")
    t0 = time.perf_counter()
    df_train = build_rollout_dataset(args.train, samples=args.samples,
                                     seed=args.seed)
    print(f"    {len(df_train)} transitions in "
          f"{time.perf_counter() - t0:.1f}s")
    print("    Label distribution:")
    print(df_train["BEST_H"].value_counts().to_string())

    print("2/3 Training Random Forest...")
    model = train_rf(df_train)

    print("3/3 Evaluating on the test set...")
    df = evaluate(args.test, model, args.outdir)
    df.to_csv(args.results_csv, index=False)
    print(f"Per-instance results saved to {args.results_csv}")


if __name__ == "__main__":
    main()
