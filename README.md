# Rollout-Based Dynamic Hyper-heuristics for the Knapsack Problem

Reproducible pipeline for the paper. Single entry point:

```bash
pip install -r requirements.txt
python rollout_hh_knapsack.py \
    --train Instances/Train-250-256 \
    --test  Instances/Test-250-256 \
    --samples 200 --seed 42
```

Outputs:
- Console: mean profits per method, Wilcoxon tests (Rollout vs MAXPW and
  RF-HH vs MAXPW), win rates, per-instance timing.
- `results.csv`: per-instance profits for every method.
- `figures/`: all paper figures at 300 dpi (boxplot, scatter,
  distributions, heatmap, feature importances, confusion matrix,
  difficulty comparison).

Method summary:
1. Rollout policy: at each step, each heuristic (DEF, MAXP, MAXPW, MINW)
   packs one item and the rest is completed with MAXPW; the best is kept.
2. The rollout acts as a labeling oracle over 8 dynamic state features.
3. A Random Forest (300 trees, min_samples_leaf=10, seed 42) imitates the
   oracle, with a confidence-gated fallback to MAXPW (threshold 0.6).

Implementation notes:
- Array-based state with save/restore instead of deepcopy; the MAXPW
  completion iterates a pre-sorted ratio order (exactly equivalent to the
  reference implementation, verified item by item).
- RF-HH test evaluation runs all instances in lockstep with one batched
  predict_proba call per step (identical solutions, amortized overhead).
