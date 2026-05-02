# Open work items

- in the tex the section "Why Adjacency Prediction Helps" can be ommitted.
- in section 5 "Architecture: Siamese Network" can you include a tikz picture depicting the architecture of the model in a schemadic way. Pleaase let the tikz picture live in a seperate tex file which is included.
- please remove "Siamese Network" from the title
- sectio 6.2 is not anymore needed. It can be shortend and included in the next section "Loss function"
-in section 6.3 of the loss function please also mention which value of the tilt parameter corresponds to the normal case where the weighted binary cross entropy correspinds to binary cross entropy
- shorten the paragraph: "Why we lean toward a moderate-to-aggressive tilt. " to a few sentences. No seperate paragraph is necessary
- in fig 1 "Figure 1: Schematic of the predicted similarity graph. " the two label text for the purple and orange node in the caption overlapp so that the text is cutoff.

## Done

- Weighted BCE bug fixed (`BCELoss(reduction='none')`, then weight, then mean).
- `n_neg / n_pos` is computed once from `(n_images, GRID)` in `main.n_neg_over_n_pos`
  and passed into the model as `pos_weight = beta * ratio`. Tilt parameter `beta`
  now exposed in `config.yaml` and propagated into `results.jsonl`.

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
