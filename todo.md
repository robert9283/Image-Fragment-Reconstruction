# Open work items

## Bugs

### Weighted BCE is not actually weighted
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
