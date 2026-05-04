# Runs after single-head refactor

The single-head refactor (branch `single-head-refactor`, commit c80b467) changed
behaviour for all runs with `lambda_same > 0`. Runs with `lambda_same = 0` are
unaffected since no same-image loss term was ever active.

## Keep (lambda_same = 0, unaffected)

| Run | Notes |
|---|---|
| `plain_bce_baseline` | adj-only baseline |
| `adj_only_seed_0..4` | ANOVA group, fully valid |
| `plain_bce`, `wbce_beta_03`, `wbce_beta_1` | Pre-date lambda fields; implicitly lambda_same=0 |

## Delete and rerun (lambda_same > 0, behaviour changed)

| Run | Notes |
|---|---|
| `multitask_01/05/10/15` | Lambda sweep with same-image term |
| `same_only` | lambda_adj=0, lambda_same=1 |
| `noise_seed_0..4` | Multi-task ANOVA group (lambda_same=1) |
| `same_only_seed_0..4` | Same-only ANOVA group |

## Delete only (no rerun needed)

| Run | Notes |
|---|---|
| `test_single_head` | Smoke test, only 200 iters |

## Rerun script

`scripts/run_overnight.sh` handles deletion and rerun of all 15 affected runs.
First validate with `test_single_head_full` (currently running), then trigger
the full overnight run.
