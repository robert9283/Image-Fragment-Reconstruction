# Open work items

- ablate the comparison-head input. Three runs at otherwise-identical settings:
    - Run 1: current four-term combination `[a-b, a*b, a, b]` (1024-dim head)
    - Run 2: plain concatenation `[a, b]` only (512-dim head)
    - Run 3: difference + product `[a-b, a*b]` only (512-dim head)
  This will tell us whether the InferSent four-term construction actually helps
  on images or whether a simpler combination is just as good. ~75 min of
  training time total.
- Add a purity column to the multi-task sweep table (the largest table, where
  it's most informative). Currently purity is only logged in per-eval log
  entries, not in the per-run summary in results.jsonl. Adding a
  `final_purity` field to the summary is a one-line change in main.py;
  picking it up in the table is another one-liner.
- After tonight's ANOVA runs finish, write up Section 9.5 (multi-seed comparison
  with t-tests) and update the headline numbers in Section 9.1 to use the
  proper means.
- clean the git up a bit and make it submission ready. Remember this task is
  for a job interview. Everything should be concise, clean and easy to
  understand for someone that is not familiar with the code. After
  submission they will need to run the pipeline to generate the model file.
  It should be clear how the models are generated and how the report was
  generated.
- in the further suggestions section can you remove the paragraph "Differentiable surrogate for ARI."

## Done

- Weighted BCE bug fixed (`BCELoss(reduction='none')`, then weight, then mean).
- `n_neg / n_pos` is computed once from `(n_images, GRID)` in `main.n_neg_over_n_pos`
  and passed into the model as `pos_weight = beta * ratio`. Tilt parameter `beta`
  now exposed in `config.yaml` and propagated into `results.jsonl`.
- Section "Why Adjacency Prediction Helps" omitted; content was redundant.
- Architecture TikZ figure in section 5 (separate `fig_architecture.tex`).
- "Siamese Network" removed from the title.
- Section 6.2 (Class Imbalance) folded into 6.3 (Loss Function).
- Plain-BCE special case ($\beta = 1/52$) explicitly mentioned in the loss section.
- Recall-tilt paragraph shortened.
- Figure 1 (similarity graph) legend overlap fixed.
- Section 7: $p_{ij}$ described as a confidence score, not a calibrated probability.
- Spectral-clustering paragraphs merged with formula; library names moved to footnote.
- "Metric consistency" paragraph removed.
- ARI section retitled "Evaluation Metric for the Clustering Task" with scope clarification.
- ARI section: clean definition with $a/b/c/d$ pair counts; purity example in a quote block.
- Random_state details moved to a footnote.
- `approach_siamese.tex` renamed to `approach.tex`.
- `summary.tex` and `summary.pdf` untracked from git, kept locally via `.gitignore`.
- Further Suggestions section pruned: items already implemented are removed; new
  forward-looking items added (differentiable ARI surrogate, pretrained encoder,
  topology-aware clustering, color-histogram baseline).
- 4x4 grid TikZ figure in section 6.1 (`fig_grid_labels.tex`).
- Failure-modes script (`scripts/failure_modes.py`) auto-selecting the current
  best checkpoint, with hyperparameters in the figure caption.
- Failure-modes script wired into `debug/generate_report.sh` so the figure
  is refreshed alongside the rest of the debug plots.
- Section 4 renamed from "Architecture: Siamese Network" to just "Architecture".
- ARI definition prefix: "Definition (Adjusted Rand Index)".
- Problem Statement simplified: dropped the heavy formal definitions
  ($\mathcal{I}$, $f_k$, $\hat{\mathcal{C}}$, two enumerated constraints,
  bulleted "Known structure"). Replaced with two short paragraphs that state
  the same content directly with concrete numbers.
- "Probability" replaced with "confidence" wherever it referred to model output.
- Multi-task and same-only ablations finished and incorporated into the Results
  section; consistency pass between doc and code completed.
- `seed` config knob added to `main.py` (`torch.manual_seed` and
  `np.random.seed` are called when `seed` is set in `config.yaml`).
- "Multi-batch averaging" paragraph moved out of the ARI section and into
  the Training section as the new "Validation: multi-batch averaging"
  subsection.
- Purity definition added to Section 7 alongside the ARI definition, with
  the formal expression $\frac{1}{M}\sum_n \max_m |\hat{C}_n \cap C_m|$ and
  a note that balanced clustering removes the over-fragmentation inflation.
- Decision Points Summary section removed.
- Per-head pretext-task metrics: same-image AUROC/AUPRC are now logged
  alongside adjacency AUROC/AUPRC whenever the same-image head is active
  (lambda_same > 0). New keys: same_auroc, same_auprc per eval step,
  final_same_auroc / final_same_auprc per-run summary.
- Single-seed result discussion in the Results section shortened where
  differences between configurations are within noise (beta sweep, full
  lambda_same sweep, same-only ablation now read more cautiously, with
  forward-reference to the multi-seed Section 9.5).

## Historical context (kept for the report)

### Weighted BCE was not actually weighted
**File:** `src/fragment_adjacency_predictor.py`, around line 99.

Current code:
```python
self.loss_fn = nn.BCELoss()                       # default reduction='mean'
...
loss = (self.loss_fn(preds, targets) * weight).mean()
```

`BCELoss(reduction='mean')` returns a **scalar** — the mean BCE over all pairs.
Multiplying that scalar by the per-pair `weight` vector and then taking `.mean()`
is mathematically equivalent to multiplying the unweighted loss by `weight.mean()`,
which is just a constant rescaling. A constant multiplier on the loss is
equivalent to a constant rescaling of the learning rate; **the class imbalance is
not being addressed at all**.

**Fix:** use `BCELoss(reduction='none')` so the loss returns a per-pair vector,
then multiply by weight, then take the mean:

```python
self.loss_fn = nn.BCELoss(reduction='none')
...
losses = self.loss_fn(preds, targets)             # shape (n_pairs,)
loss   = (losses * weight).mean()
```

This will be the headline change for the next training round once the current
ablation cycle finishes.

## Code simplifications

### Hard-code the n-/n+ ratio
For our fixed configuration (10 images, 4×4 grid), the ratio of negative to
positive pairs is always exactly `12480 / 240 = 52` — it does not depend on the
batch. The runtime computation in `train_step` should be replaced with a constant
derived from `N_IMAGES` and `GRID`, making the dependency on grid size explicit
and removing one source of confusion. While we are there, expose the tilt
parameter `beta` in `config.yaml`.
