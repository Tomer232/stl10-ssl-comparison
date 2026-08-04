---
name: parity-check
description: Audit that Tomer's VAE+LabelProp arm and Lior's SimCLR arm are actually comparable — same trunk, same embedding dim, same splits, same seeds, same budget, test set touched once. Use before any accuracy number goes into the report or slides, after changing either arm's encoder, splits, normalization, or training budget, and whenever the user asks "is this a fair comparison", "can we compare these", "are the numbers comparable", or is about to write up results.
---

# Parity check: is the comparison actually fair?

The whole project answers one question — *does a generative or a contrastive latent space
propagate labels better?* That question is only answerable if the **pretraining objective
is the only thing that differs** between the two arms. Every item below is a way for
something else to differ and silently become the real explanation for the gap.

Run this before any headline number ships. Report each item as **PASS / FAIL / UNKNOWN**,
with the file and line you checked. `UNKNOWN` is a legitimate and useful verdict — never
guess a PASS.

Lior's arm is `simclr_stl10_abilation_f.ipynb`. **Read it, do not modify it.**

## The checklist

### 1. Embedding dimensionality — highest risk

Lior's encoder outputs **512-d** pooled features with the projection head discarded. Our
VAE's `mu` must be 512-d too.

Different dims are not a small discrepancy: KNN graph density and cosine-similarity
distributions both change with dimension, so LP quality shifts for reasons that have
nothing to do with the latent space being generative or contrastive.

Check: our latent dim in the VAE config, against his encoder's output dim. Confirm the
projection head is discarded on his side, not tapped.

### 2. Tap point

Both arms must read features from the **same layer** — after global average pool, before
any head. If he taps post-pool and we tap the projection or a pre-pool activation, the
comparison is void.

### 3. Trunk identity

Same SE-ResNet on both sides, ours reimplemented from scratch (build-from-0):

- 2-conv stem: Conv3x3 s2 3→48, BN, ReLU, Conv3x3 s1 48→64, **no maxpool**
- pre-activation SE blocks, SE reduction 16
- depths 2-3-4-3, widths 64/128/256/384
- 1×1 conv to 512, global average pool
- 96×96 native input

Diff our implementation against his cell 12 layer by layer. Parameter count is a fast
sanity check but not proof — a matching count with mismatched ordering still fails.

### 4. Splits and held-out indices

The protocol is his, transcribed from cells 8 and 16: the **10 official folds** from
`fold_indices.txt` (1000 indices each into the 5000 labeled train images), with an
**800/200 stratified development split** inside each fold for all model selection, seeded
`SEED + fold_index`. Final evaluation fits each fold's full 1000 and scores all 8000 test
images, mean over the 10 folds.

Check ours uses the same folds, the same 800/200 protocol, and the same seeding rule.

**Known asymmetry, documented not fixed:** his stratified split comes from sklearn's
`train_test_split`; ours is hand-rolled because sklearn is banned. Same protocol and same
seed, but not the identical partition. Report it; do not claim equivalence.

### 4b. C selection

He picks **one C for all ten folds** — every fold's inner CV averaged across folds, ranked
by log loss, then accuracy, then macro-F1, then smaller C. Ours must default to the same.
If a run used `--c-selection per-fold`, that row is a sensitivity check and is **not**
comparable; FAIL any report that presents it as the headline.

### 5. Normalization constants

Same per-channel mean/std, and the same convention (computed on which split?). A mismatch
shifts the embedding geometry before anything else happens.

### 6. Test-set discipline

The test split is touched **once, at the end, by both arms**. Model selection — K, beta,
epoch count, anything — happens on the held-out labeled subset only.

FAIL this loudly if a test-set number appears anywhere in a sweep, a plot over epochs, or
an early-stopping criterion. It is the one error that invalidates the whole report rather
than one row of it.

### 7. Pretraining budget

Lior: 100 epochs, batch 256. Ours must match the epoch count on the same 100k unlabeled
split. If we converge earlier, still train the full budget — an unequal budget is the
first thing a reader will attack.

### 8. Seeds and the spread convention

≥3 seeds per arm, reported as mean ± std, with **ddof=1** (the project-wide convention).
A single-seed difference between two SSL methods on STL-10 is well within noise, which is
why the spread matters. Lior's notebook computes no std at all, so that column has no
counterpart on his side and must not be presented as a like-for-like comparison.

**Known open gap: Lior's notebook is single-seed (`SEED=42`).** Until that is resolved, any
comparison is one seed against three. Surface it every run — do not let it quietly become
"fine". The fix is his to make; our job is to keep raising it and to report the asymmetry
honestly if it stays.

### 9. Legitimate differences — document, do not force-match

These *should* differ and forcing them equal would be its own distortion. They belong in a
"differences we did not control for" paragraph:

- augmentations (SimCLR needs its strong two-view pipeline; a VAE does not)
- batch size (contrastive learning is batch-size sensitive in a way a VAE is not)
- optimizer and LR schedule

### 10. Downstream capacity confound

His downstream is a **linear probe**; ours is a **CNN trained on pseudo-labels**. A
straight accuracy comparison is confounded by classifier capacity.

This is why the pipeline includes a **linear probe on our own embeddings** — that row, not
the CNN row, is the apples-to-apples comparison against Lior. Check it exists and that the
report leads with it.

## Output format

A table of item / verdict / evidence, then:

- **Blocking** — anything that makes a number wrong (dim mismatch, split mismatch, test
  leakage). These must be fixed before the number is used.
- **Reportable** — real asymmetries that stay, and go in the writeup as caveats (the
  single-seed gap, the capacity confound, the augmentation difference).

Never soften a FAIL into a caveat to make a result publishable.
