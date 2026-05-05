# scripts/

## Experiment planning (`plan.py` + `run_plan.db`)

All planned training runs are tracked in `run_plan.db` (SQLite).

```bash
python scripts/plan.py list                    # show all runs and their status
python scripts/plan.py init                    # (re-)populate DB from the hardcoded plan
python scripts/plan.py mark-done <run_name>    # manually mark a run as done
python scripts/plan.py reset <run_name>        # re-queue a run
python scripts/plan.py add <run_name> --lambda-adj X --lambda-same Y --beta Z [--seed N] [--group G] [--notes TEXT]
```

## Running experiments

```bash
caffeinate -i bash scripts/run_overnight.sh    # run all pending; Ctrl-C stops after current run
```

On resume, already-`done` runs are skipped automatically. After all runs finish,
`compare_runs.py` runs automatically.

## Final test evaluation

After the ANOVA is complete and the best run has been identified, run the
test evaluation on its checkpoint:

```bash
bash scripts/run_test_eval.sh <run_name>
# e.g. bash scripts/run_test_eval.sh multitask_10_seed_2
```

Results are saved to `runs/<run_name>/test_metrics.json` (co-located with
the checkpoint). This runs 1000 test batches and reports ARI, NMI, purity,
and adjacency precision/recall/F1.

## R analysis

Once all 20 ANOVA runs are done, run the R analysis:

```bash
Rscript scripts/anova_r.R    # from project root
```

This writes three LaTeX tables to `doc/` (`tab_anova_summary.tex`,
`tab_anova_omnibus.tex`, `tab_anova_contrasts.tex`), a plot
(`doc/fig_anova_ari.pdf`), and prints the pooled $\hat{\sigma}$ and
$\Delta_{\min}$ values needed to fill the TODO in `doc/approach.tex`.

## Other scripts

| Script | Purpose |
|---|---|
| `compare_runs.py [--plot]` | Table of all results from `results.jsonl`; `--plot` saves `runs_comparison.png` |
| `anova_r.R` | Unified 5-condition ANOVA (Welch one-way + planned contrasts); run **after all overnight runs complete** |
| `roc_curves.py` | ROC/PR curves for selected runs |
| `failure_modes.py` | Visualise worst-case clustering failures |
| `weight_distribution.py` | Plot encoder weight distributions |
| `test_single_head.sh` | Full-length validation run for the single-head refactor (seed 0, lambda_same=1) |
