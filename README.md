# Rollout-Based Dynamic Hyper-heuristics for the Knapsack Problem

We propose a dynamic hyper-heuristic for the 0/1 Knapsack Problem that selects,
at **every step** of the solving process, the most suitable constructive heuristic
(DEF, MAXP, MAXPW, or MINW) using a one-step rollout with MAXPW completion.
Since MAXPW is always among the candidates, the rollout profit is never below
that of pure MAXPW (sequential improvement property). The rollout decisions are
then distilled into a Random Forest classifier (RF-HH) over eight dynamic state
features, which captures 96.4% of the oracle's profit without any simulations
and whose per-step cost is independent of the number of heuristics and the
lookahead depth.

## Repository structure

```
├── rollout_hh_knapsack.py     # Main pipeline: heuristics, rollout policy,
│                              # RF-HH, evaluation, and figure generation
├── scalability_experiment.py  # Runtime scalability experiment (rollout vs RF-HH)
├── requirements.txt
├── Instances/
│   ├── Train-250-256/         # 1000 training instances (n=250, C=256)
│   └── Test-250-256/          # 300 test instances (n=250, C=256)
└── figures/                   # Output figures (generated)
```

Benchmark instances come from Zárate-Aranda & Ortiz-Bayliss (2025), generated
with the evolutionary technique of Plata-González et al. (2019). Each instance
has 250 items with integer weights and profits in [1, 256] and knapsack
capacity 256.

## Requirements

Python 3.10+:

```bash
pip install -r requirements.txt
```

## Reproducing the paper results

### 1. Main experiment (Table 1 and all result figures)

```bash
python rollout_hh_knapsack.py \
    --train Instances/Train-250-256 \
    --test  Instances/Test-250-256 \
    --samples 200 --seed 42 \
    --outdir figures --results-csv results.csv
```

This: (1) labels 200 randomly sampled training instances with the rollout
oracle, (2) trains the Random Forest (300 trees, min leaf size 10, confidence
threshold 0.6 with MAXPW fallback), and (3) evaluates all six methods on the
300 test instances. It prints mean profits, Wilcoxon signed-rank tests
(Rollout vs MAXPW and RF-HH vs MAXPW), win rates, and average solving times;
writes per-instance results to `results.csv`; and saves all figures (300 dpi)
to `figures/` (boxplot, scatter, distributions, heuristic-selection heatmap,
feature importances, confusion matrix, difficulty comparison).

Expected output (seed 42):

| Method  | Mean profit |
|---------|-------------|
| DEF     | 965.41      |
| MAXP    | 1201.00     |
| MINW    | 1838.35     |
| MAXPW   | 2790.88     |
| Rollout | **2888.40** |
| RF-HH   | 2783.85     |

Rollout vs MAXPW: Wilcoxon p = 1.88e-41, wins on 242/300 instances (80.7%).
RF-HH captures 96.4% of the rollout profit with no simulations.

### 2. Scalability experiment (runtime figure, Section 5.2)

```bash
python scalability_experiment.py \
    --sizes 250 500 1000 2000 \
    --eval-instances 30 --train-instances 200 --seed 42 \
    --outdir figures
```

Generates synthetic instances (weights and profits uniform in [1, 256],
capacity scaled as `round(256*n/250)`) used **only to measure runtime**, and
produces `figures/scalability.png` (log-log) and `figures/scalability.csv`.
The rollout exhibits empirical quadratic growth with n, while RF-HH grows
linearly; the RF-HH per-step cost is independent of the number of heuristics
and the lookahead depth.

## Reproducibility notes

- All experiments use `--seed 42` by default (NumPy and Python `random`).
- The Random Forest uses `n_jobs=1`: single-sample inference inside the
  solving loop is faster without joblib parallelism overhead.
- Instance file format: first line `n_items, capacity`, then one
  `weight, profit` pair per line.

## Citation

```bibtex
@inproceedings{alanis2026rollout,
  title     = {Rollout-Based Dynamic Hyper-heuristics for the Knapsack Problem},
  author    = {Alan{\'i}s-Guti{\'e}rrez, V{\'i}ctor Alejandro and
               Romero-Contreras, Emmanuel and
               Espinoza-Miranda, Carlo Guillermo and
               Zambrano-Guti{\'e}rrez, Daniel Fernando and
               Amaya, Ivan and
               Ortiz-Bayliss, Jos{\'e} Carlos},
  booktitle = {Mexican International Conference on Artificial Intelligence (MICAI)},
  year      = {2026}
}
```
