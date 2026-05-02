# Open work items

- in the tex the section "Why Adjacency Prediction Helps" can be ommitted.
- in section 5 "Architecture: Siamese Network" can you include a tikz picture depicting the architecture of the model in a schemadic way. Pleaase let the tikz picture live in a seperate tex file which is included.
- please remove "Siamese Network" from the title
- sectio 6.2 is not anymore needed. It can be shortend and included in the next section "Loss function"
-in section 6.3 of the loss function please also mention which value of the tilt parameter corresponds to the normal case where the weighted binary cross entropy correspinds to binary cross entropy
- shorten the paragraph: "Why we lean toward a moderate-to-aggressive tilt. " to a few sentences. No seperate paragraph is necessary
- in fig 1 "Figure 1: Schematic of the predicted similarity graph. " the two label text for the purple and orange node in the caption overlapp so that the text is cutoff.
- in section 7 you write that p reflects the models probability. I think that is wrong it is just a measure of confidence.
- in section 7 "From Pretext Task to Clustering" the two paragraph "Decision point: clustering on the graph. Spectral clustering" and "Balanced spectral clustering." can be merged since we do not use anymore spectral clustering by itself. Do not mention the libraries used in the text. You can add them into footnotes. Add the formula of spectral clustering and explain very briefly baanced spectral clustering. The paragraph "Metric consistency." can be removed.
- the title "Evaluation Metric: Adjusted Rand Index" is a bit misleading since we have two task. You shoud specifiy that you mean the evaluation metric for the clustering of the image fragments. Please make that also in the section clear.
- in section "Experimental Setup" the text "random_state = 42 insklearn.cluster.Spec
KMeans, andSpectralEmbedding)" gets cutoff. Also I think this can be moved into a foot note.
- i would like that file siamese_approach.tex gets renamed into something more meaningful. The siamese_approach is not so telling.
- can you take the summary.tex out of the git but leave it in the folder so I wont publish this later accidedently
- please also give a clean definition of ARI. Encapsulate the example you give for ARI and puri into an example block.
- check consistency of document and also consistency with code
- update section 14: further suggestions. Some of this stuff has been done already
- in section 6.1 "Constructing Training Labels" i would like to have a tikz picture of a 4 times 4 grid of fragments.
- i would like to have a script that extract me a few failure modes. That is a set of 10 pictures where the clustering didnt work very well. In the caption of the image we should write all the hyperparameters used so we dont confuse models easily.

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
