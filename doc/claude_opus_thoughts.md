# Claude's thoughts on the fragment reconstruction problem

*Generated 2026-05-02 11:49:45*

---

## 1. Possible approaches to the problem

The problem is: given 160 mixed fragments from 10 images, recover which fragments belong
to which image, with no labels and no positional info at inference.

### A. Classical / non-learning baselines

These are worth at least implementing as baselines:

- **Color / texture histogram clustering.** Each fragment has a color distribution.
  Same-image fragments share lighting, palette, and overall tonality. Compute pairwise
  histogram distances (chi-square, EMD), cluster the resulting affinity matrix
  spectrally. No training required. Surprisingly competitive on this kind of problem.
- **Boundary matching (jigsaw solver).** For each fragment pair, score how well their
  borders could meet — sum of squared differences along candidate edges. This is the
  classical jigsaw puzzle approach (Pomeranz, Sholomon, Paikin & Tal). Works even
  better in the *un-augmented* case where boundaries are pixel-aligned.
- **Pre-trained feature extractor.** Run each fragment through a frozen CNN (ResNet,
  DINO, CLIP) and cluster the features. Caveat: 16×16 is well below the scale these
  models were trained on, so features may not be meaningful; would need upsampling.

### B. Self-supervised representation learning

- **Contrastive (SimCLR / MoCo).** Use augmentations as positives. Doesn't directly
  solve the task because two augmentations of the same fragment is not the same as
  two fragments from the same image — but the learned features can still cluster.
- **Adjacency prediction (current approach).** Predict whether two fragments are
  spatially adjacent in the original grid; cluster on the resulting similarity matrix.
- **Jigsaw permutation prediction (Noroozi 2016).** Train the model to recover the
  permutation that scrambled the grid. The encoder learns transferable features.
- **DeepCluster / SwAV-style.** Alternate between clustering current embeddings and
  using cluster assignments as pseudo-labels for the next training round. Naturally
  matched to the task.

### C. End-to-end / structured prediction

- **Set-equivariant transformer over all 160 fragments.** Feed all fragments at once;
  let attention figure out which belong together. Output: cluster assignment logits
  per fragment. Heavy but conceptually clean.
- **Graph neural network.** Build a fully connected graph; learn edge weights and
  cluster jointly via end-to-end spectral relaxation.
- **Iterative EM.** Alternate between (i) assigning fragments to images and (ii)
  refining a per-image generative model. Beautiful but slow.

### D. Hybrid

- **Multi-task pretext.** Combine adjacency prediction with same-image classification
  (treat the 10 images per batch as a 10-way classification task — labels exist within
  the batch).
- **Classical features + learned head.** Use color histogram + pre-trained features as
  inputs to a small Siamese network.
- **Boundary-aware loss.** Add an explicit pixel-level boundary continuity term to the
  adjacency prediction loss.

---

## 2. Quality of the current solution

### What works

- The pretext task is principled and self-supervised by construction. No leakage of
  source labels into training.
- Spectral clustering on the predicted similarity matrix is the right inference choice
  for this output format — no metric mismatch between training and clustering.
- The infrastructure is solid: training log with full timing breakdown, automatic
  early stopping, debug report, separate scripts for adjacency and clustering metrics.
- Achieves **ARI ≈ 0.78** with around 18 minutes of training. That's decent for a
  10-way clustering with no labels.

### What's weak

1. **The adjacency F1 is low (~0.26).** This is the crucial diagnostic. The model is
   not actually solving the adjacency problem — it's learning a softer "do these come
   from the same image?" signal. That's still useful (it's why ARI is high), but it
   means the chosen pretext task is partly redundant. A simpler pretext (same-image
   prediction) might work as well or better.
2. **Augmentation breaks the adjacency signal.** Rotation up to 20° plus channel shifts
   destroy pixel-level boundary continuity. The whole *theoretical* justification for
   adjacency prediction (edge continuity) is partially undermined by the augmentation.
   We never quantified this.
3. **High evaluation variance.** ARI is computed on a single fresh validation batch
   each step. Some batches are easier than others. Running 30+ batches and averaging
   would give much tighter curves.
4. **The fragment size is brutal.** 16×16 patches are tiny — many will be near-uniform
   regions (sky, grass, walls) that genuinely cannot be distinguished. There's a
   theoretical ceiling well below 1.0 ARI.
5. **No baselines were measured.** We have no idea whether a color-histogram baseline
   gets ARI 0.3 or ARI 0.7. Without baselines we cannot claim the learned approach
   is doing significant work.
6. **No ablations.** What's the effect of dropout? Weight decay? Architecture size?
   The 4-term combination in the comparison head? We don't know.
7. **The whole 160-fragment batch is fed each step.** This makes each step expensive
   (~350 ms) and limits how many training samples the model sees per minute.

### Easy improvements (ordered by ROI)

1. **Implement a color-histogram baseline.** 30 minutes of work, gives a critical
   reference number for the report.
2. **Average ARI over 20+ validation batches** at each eval point. Removes the
   variance, makes curves readable, no extra training cost.
3. **Add same-image prediction as a multi-task head.** One extra binary head:
   "are these from the same source?" The labels are free (we know `s(k)`). This is
   what the model is implicitly learning anyway; making it explicit may speed up
   convergence.
4. **Balanced spectral clustering.** Already in TODO. Each cluster must have exactly
   16 fragments. Should bump ARI noticeably.
5. **Hungarian-matched purity reporting** for the visualization plots.
6. **Quantify the augmentation effect.** Train one model with augmentation off and
   compare. If adjacency F1 jumps from 0.26 to 0.7 without augmentation, that's a
   killer plot for the report.

---

## 3. Originality vs. other applicants

### What other applicants will probably do

Most applicants given this task will, with Claude's help, gravitate to one of three
buckets:

- **(60%) Contrastive learning / SimCLR-style.** It's the canonical self-supervised
  answer and the first thing Claude suggests when prompted with "self-supervised
  image grouping."
- **(20%) Pre-trained feature extractor + k-means.** The lazy / pragmatic answer.
- **(15%) Adjacency / jigsaw pretext (our approach).** This is the second-most
  obvious answer, especially if someone reads the Noroozi 2016 paper. Not unique.
- **(5%) Something unusual** — boundary matching, GNN, iterative EM.

So our approach is **not particularly original**. A senior reviewer who has seen
several submissions will recognize the Noroozi pretext immediately.

### What *would* differentiate us

The technical choice itself is unlikely to be the differentiator. What can
differentiate us is:

1. **Baselines + honest evaluation.** Most applicants will not run a
   color-histogram baseline. If we show:
   - Color histogram: ARI = X
   - Pre-trained features: ARI = Y
   - Our adjacency model: ARI = 0.78
   ... we look like someone who actually thinks about the problem rather than just
   reaches for a fashionable method. This is *the* highest-leverage move.

2. **Acknowledge the augmentation problem.** Few applicants will spot that the
   augmentation pipeline destroys the very signal their pretext task relies on.
   Explicitly diagnosing this — with an ablation showing F1 with/without
   augmentation — demonstrates real understanding of the data, not just the method.

3. **Diagnose the F1=0.26 / ARI=0.78 gap.** This gap is interesting. The model is
   *not* solving the stated pretext but is still useful for clustering. Why?
   Because adjacency is a sufficient-but-not-necessary signal for source identity.
   Discussing this in the report shows analytical maturity.

4. **A theoretical framing.** Cast the problem precisely (we already did this in
   Section 1 of the tex). Discuss the identifiability ceiling: which fragments
   *cannot* be correctly grouped (uniform sky tiles, etc.)? Estimate this ceiling
   empirically by examining mis-clustered fragments. This kind of analysis is rare
   in submissions.

5. **Code and infrastructure quality.** Clean separation of concerns, configurable
   training, automated debug reports, training logs with timing breakdowns. This
   says "I would be safe to hire" louder than any single algorithmic choice.

6. **A small architectural twist.** The 4-term comparison vector
   `[a−b, a⊙b, a, b]` is slightly non-standard (most jigsaw papers just
   concatenate). If we can show empirically (small ablation) that this beats plain
   concatenation, that's a concrete original contribution we can point to.

### Where to invest the remaining time

If we have, say, half a day before the deadline, the highest-leverage uses are:

- 1 hour: implement color-histogram baseline, plot the comparison
- 1 hour: average ARI over 20 batches at each eval, regenerate curves
- 1 hour: train one ablation with augmentation off, plot the comparison
- 30 min: balanced spectral clustering (already specced in TODO)
- 1 hour: write the failure-mode analysis section
- 30 min: polish the report, make sure the narrative is clear

Doing *any two* of these would put the submission ahead of most applicants. Doing
all of them would make the submission unusually thorough.

### Bottom line

Our algorithmic choice is not original. Our framing, evaluation rigor, and
diagnostic honesty *can* be. The differentiator is not the method but the depth of
engagement with the problem.
