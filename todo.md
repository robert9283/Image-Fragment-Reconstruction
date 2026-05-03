# Open work items

- check consistency of document and also consistency with code
- the problem statement at the beginning of the tex file is a bit too formal. That is overkill. Just state the problem very precisiely. You can use a bit of mathematical terminology but do not formlaize fully.
- please call section 4 just architecture and not architecure siamese
- ablate the comparison-head input. Three runs at otherwise-identical settings:
    - Run 1: current four-term combination `[a-b, a*b, a, b]` (1024-dim head)
    - Run 2: plain concatenation `[a, b]` only (512-dim head)
    - Run 3: difference + product `[a-b, a*b]` only (512-dim head)
  This will tell us whether the InferSent four-term construction actually helps
  on images or whether a simpler combination is just as good. ~75 min of
  training time total.
- In Fig 1 "Schematic of the model." you talk about probabilites. I think probability is the wrong word. That mistake you made at severak positions.
- the problem statement at the beginning of the tex file is a bit too formal. That is overkill. Just state the problem very precisiely. You can use a bit of mathematical terminology but do not formlaize fully
.
- please call section 4 just architecture and not architecure siamese
- in section 7 "Evaluation Metric for the Clustering Task" in the definition of ARI please write explictiley in the definition in brackets at the start that the definition refers to ari.

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
