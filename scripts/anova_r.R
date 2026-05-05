#!/usr/bin/env Rscript
# =============================================================================
# anova_r.R — Welch one-way ANOVA for the 5-condition hyperparameter study
# =============================================================================
#
# Compares best-checkpoint ARI across five training configurations (5 seeds
# each) to assess whether the same-image auxiliary loss and the loss-tilt
# parameter beta have a statistically significant effect on clustering quality.
#
# Inputs:
#   results.jsonl   — one JSON line per run, written by main.py
#
# Outputs (all written to doc/):
#   tab_anova_summary.tex   — per-condition mean / SD / min / max
#   tab_anova_omnibus.tex   — Welch one-way ANOVA F-test result
#   tab_anova_contrasts.tex — two planned Welch t-tests with Cohen's d
#   fig_anova_ari.pdf/.png  — jitter + mean ± SD plot
#
# Run from the project root:
#   Rscript scripts/anova_r.R
# =============================================================================

suppressPackageStartupMessages({
  library(jsonlite)
  library(dplyr)
  library(ggplot2)
  library(xtable)
})

# ── config -------------------------------------------------------------------- ----

# Significance threshold; 0.10 rather than 0.05 because n=5 per condition
# gives low power, and we treat this as an exploratory study.
ALPHA <- 0.10

RESULTS_PATH <- "results.jsonl"
OUT_DIR      <- "doc"

# Map condition label -> run names
COND_RUNS <- list(
  a = paste0("adj_only_seed_",     0:4),
  b = paste0("same_only_seed_",    0:4),
  c = paste0("multitask_10_seed_", 0:4),
  d = paste0("multitask_02_seed_", 0:4),
  e = paste0("adj_beta1_seed_",    0:4)
)

# Plain-text labels (for plots)
COND_LABELS_PLAIN <- c(
  a = "(a) adj-only b=1/52",
  b = "(b) same-only",
  c = "(c) multi-task ls=1.0",
  d = "(d) multi-task ls=0.2",
  e = "(e) adj-only b=1"
)

# LaTeX labels (for tables)
COND_LABELS_TEX <- c(
  a = "(a) adj-only, $\\beta=1/52$",
  b = "(b) same-only",
  c = "(c) multi-task, $\\lambda_s=1.0$",
  d = "(d) multi-task, $\\lambda_s=0.2$",
  e = "(e) adj-only, $\\beta=1$"
)

# ── load data ----------------------------------------------------------------- ----

all_run_names <- unlist(COND_RUNS)

# Read results.jsonl line by line; NULL fields (optional keys absent from some
# runs) are replaced with NA before coercing to a data frame so bind_rows works.
raw <- lapply(readLines(RESULTS_PATH), fromJSON)
df_all <- bind_rows(lapply(raw, function(x) {
  x[sapply(x, is.null)] <- NA
  as.data.frame(x, stringsAsFactors = FALSE)
}))

# Keep only ANOVA runs; if a run was resumed or re-logged, take the last entry.
df <- df_all |>
  filter(run %in% all_run_names) |>
  group_by(run) |>
  slice_tail(n = 1) |>
  ungroup() |>
  mutate(condition = case_when(
    run %in% COND_RUNS$a ~ "a",
    run %in% COND_RUNS$b ~ "b",
    run %in% COND_RUNS$c ~ "c",
    run %in% COND_RUNS$d ~ "d",
    run %in% COND_RUNS$e ~ "e"
  )) |>
  mutate(condition = factor(condition, levels = names(COND_RUNS)))

# Completeness check
n_per_cond <- df |> count(condition, .drop = FALSE)
cat("Runs loaded per condition:\n")
print(n_per_cond, n = Inf)

incomplete <- n_per_cond |> filter(n < 5) |> pull(condition) |> as.character()
if (length(incomplete) > 0) {
  warning("Incomplete data for conditions: ", paste(incomplete, collapse = ", "),
          ". Results may be unreliable.")
}

# ── summary statistics -------------------------------------------------------- ----

summ <- df |>
  group_by(condition) |>
  summarise(
    n       = n(),
    mean    = mean(best_ari),
    sd      = sd(best_ari),
    min     = min(best_ari),
    max     = max(best_ari),
    .groups = "drop"
  )

cat("\nSummary statistics (best ARI):\n")
print(summ)

# ── pooled within-condition SD ------------------------------------------------ ----

# Pooled SD (hat_sigma) estimates the common within-condition variability.
# It is used in the report to compute the minimum detectable difference (MDD):
#   delta_min = 1.43 * hat_sigma  (80% power, two-sided t-test, alpha=0.10, n=5)
pooled_sd <- summ |>
  summarise(s = sqrt(sum((n - 1) * sd^2) / sum(n - 1))) |>
  pull(s)

delta_min <- 1.43 * pooled_sd   # MDD at 80% power

cat(sprintf("\nPooled within-condition SD (hat_sigma): %.4f\n", pooled_sd))
cat(sprintf("MDD at 80%% power (1.43 * hat_sigma):   %.4f\n", delta_min))
cat("  -> paste these values into the TODO in doc/approach.tex\n")

# ── Welch one-way ANOVA (omnibus) --------------------------------------------- ----

# Welch's variant (var.equal = FALSE) is used because group variances may differ;
# it is more robust than the classical F-test when homoscedasticity is uncertain.
aov_res <- oneway.test(best_ari ~ condition, data = df, var.equal = FALSE)
cat("\nWelch one-way ANOVA:\n")
print(aov_res)

omnibus_sig <- aov_res$p.value < ALPHA
cat(sprintf("Omnibus significant at alpha=%.2f: %s\n", ALPHA, omnibus_sig))

# ── planned contrasts (Welch two-sample t-tests) ------------------------------ ----

# Two pre-specified contrasts against the adjacency-only baseline (a):
#   (c) vs (a) — does adding the same-image loss improve ARI?
#   (e) vs (a) — does a larger beta (stronger positive reweighting) hurt ARI?
# Using Welch t-tests (var.equal = FALSE) for the same robustness reason as above.
ari_a <- df$best_ari[df$condition == "a"]
ari_c <- df$best_ari[df$condition == "c"]
ari_e <- df$best_ari[df$condition == "e"]

ct1 <- t.test(ari_c, ari_a, var.equal = FALSE)   # contrast 1: (c) vs (a)
ct2 <- t.test(ari_e, ari_a, var.equal = FALSE)   # contrast 2: (e) vs (a)

#' Compute pooled Cohen's d effect size between two groups.
#'
#' @param x Numeric vector for group 1.
#' @param y Numeric vector for group 2.
#' @return Scalar Cohen's d (positive if mean(x) > mean(y)).
cohens_d <- function(x, y) {
  sp <- sqrt(((length(x) - 1) * var(x) + (length(y) - 1) * var(y)) /
               (length(x) + length(y) - 2))
  (mean(x) - mean(y)) / sp
}

contrasts_df <- data.frame(
  Contrast = c(
    "(c) vs (a): same-image loss",
    "(e) vs (a): loss tilt $\\beta$"
  ),
  Estimate = round(c(
    ct1$estimate[[1]] - ct1$estimate[[2]],
    ct2$estimate[[1]] - ct2$estimate[[2]]
  ), 4),
  CI95 = c(
    sprintf("[%.4f,\\;%.4f]", ct1$conf.int[1], ct1$conf.int[2]),
    sprintf("[%.4f,\\;%.4f]", ct2$conf.int[1], ct2$conf.int[2])
  ),
  t     = round(c(ct1$statistic, ct2$statistic), 3),
  df    = round(c(ct1$parameter, ct2$parameter), 1),
  p     = round(c(ct1$p.value,   ct2$p.value),   4),
  d     = round(c(cohens_d(ari_c, ari_a), cohens_d(ari_e, ari_a)), 3),
  sig   = ifelse(c(ct1$p.value, ct2$p.value) < ALPHA, "*", ""),
  check.names = FALSE
)
names(contrasts_df)[3] <- "95\\% CI"

cat("\nPlanned contrasts:\n")
print(contrasts_df)

# ── LaTeX tables -------------------------------------------------------------- ----

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# sanitize.text.function = identity prevents xtable from escaping LaTeX math
# (dollar signs, backslashes) that we have deliberately embedded in the labels.

## 1. Summary table
summ_tex <- summ |>
  mutate(Condition = COND_LABELS_TEX[as.character(condition)]) |>
  transmute(
    Condition,
    `$n$`       = n,
    `Mean ARI`  = sprintf("%.4f", mean),
    SD          = sprintf("%.4f", sd),
    Min         = sprintf("%.4f", min),
    Max         = sprintf("%.4f", max)
  )

print(
  xtable(summ_tex,
    caption = paste0(
      "Per-condition summary statistics for best-checkpoint ARI ",
      "(5 seeds each). $\\hat{\\sigma} = ", sprintf("%.4f", pooled_sd),
      "$; $\\Delta_{\\min} \\approx ", sprintf("%.4f", delta_min), "$."
    ),
    label = "tab:anova-summary",
    align = "llccccc"
  ),
  file                    = file.path(OUT_DIR, "tab_anova_summary.tex"),
  include.rownames        = FALSE,
  booktabs                = TRUE,
  sanitize.text.function  = identity,
  caption.placement       = "top"
)

## 2. Omnibus ANOVA table
omnibus_tex <- data.frame(
  Source   = "Between conditions",
  F        = round(aov_res$statistic, 3),
  `$df_1$` = round(aov_res$parameter[1], 1),
  `$df_2$` = round(aov_res$parameter[2], 1),
  p        = round(aov_res$p.value, 4),
  sig      = ifelse(aov_res$p.value < ALPHA, "*", ""),
  check.names = FALSE
)

print(
  xtable(omnibus_tex,
    caption = sprintf(
      "Welch one-way ANOVA omnibus test ($\\alpha = %.2f$). * $p < \\alpha$.",
      ALPHA
    ),
    label = "tab:anova-omnibus",
    align = "llrrrrl"
  ),
  file                    = file.path(OUT_DIR, "tab_anova_omnibus.tex"),
  include.rownames        = FALSE,
  booktabs                = TRUE,
  sanitize.text.function  = identity,
  caption.placement       = "top"
)

## 3. Contrasts table
print(
  xtable(contrasts_df,
    caption = sprintf(
      paste0("Planned contrasts (Welch $t$-test, $\\alpha = %.2f$). ",
             "Estimate $=$ mean(cond) $-$ mean(a). $d$ = Cohen's $d$. ",
             "* $p < \\alpha$."),
      ALPHA
    ),
    label = "tab:anova-contrasts",
    align = "llrcrrrrl"
  ),
  file                    = file.path(OUT_DIR, "tab_anova_contrasts.tex"),
  include.rownames        = FALSE,
  booktabs                = TRUE,
  sanitize.text.function  = identity,
  caption.placement       = "top"
)

cat(sprintf("\nLaTeX tables written to %s/\n", OUT_DIR))
cat("  tab_anova_summary.tex\n")
cat("  tab_anova_omnibus.tex\n")
cat("  tab_anova_contrasts.tex\n")

# ── plot ---------------------------------------------------------------------- ----

# Individual seed results as jittered points; mean ± 1 SD as a diamond + error bar.
plot_df <- df |>
  mutate(label = COND_LABELS_PLAIN[as.character(condition)])

means_df <- plot_df |>
  group_by(condition, label) |>
  summarise(m = mean(best_ari), s = sd(best_ari), .groups = "drop")

p <- ggplot(plot_df, aes(x = condition, y = best_ari)) +
  geom_jitter(width = 0.08, size = 2.2, alpha = 0.55, colour = "#2C7BB6") +
  geom_errorbar(
    data    = means_df,
    mapping = aes(x = condition, ymin = m - s, ymax = m + s),
    width   = 0.18, linewidth = 0.7, inherit.aes = FALSE
  ) +
  geom_point(
    data    = means_df,
    mapping = aes(x = condition, y = m),
    size = 3.5, shape = 18, inherit.aes = FALSE
  ) +
  scale_x_discrete(labels = COND_LABELS_PLAIN) +
  labs(
    x     = NULL,
    y     = "Best-checkpoint ARI",
    title = "ARI by condition (n=5 seeds; diamond = mean, bar = \u00b1 1 SD)"
  ) +
  theme_bw(base_size = 11) +
  theme(axis.text.x = element_text(angle = 15, hjust = 1))

ggsave(file.path(OUT_DIR, "fig_anova_ari.pdf"), p, width = 7, height = 4)
ggsave(file.path(OUT_DIR, "fig_anova_ari.png"), p, width = 7, height = 4, dpi = 150)
cat(sprintf("Plot written to %s/fig_anova_ari.{pdf,png}\n", OUT_DIR))
