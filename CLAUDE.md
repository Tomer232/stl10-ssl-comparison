# STL-10 Semi-Supervised Comparison — Afeka "Advanced Methods In ML" final project

Two-arm comparison on STL-10 by **Tomer Karmazin** and **Lior Reznik**.

- **Lior's arm:** SimCLR + linear probe (`simclr_stl10_abilation_f.ipynb`, in this repo for reference).
- **Tomer's arm:** VAE + Label Propagation. This is the code we write here.

**Research question: does a generative or a contrastive latent space propagate labels better?**

---

## Non-negotiable constraints

### "Build from 0" — binds Tomer's arm only

No `sklearn`, no `scipy`, no `faiss`, no `networkx`, no `torchvision.models`, no
approximate-NN libraries. PyTorch and NumPy are allowed. Hand-write:

- the KNN graph construction
- label propagation
- all metrics, from a `bincount` confusion matrix
- union-find for connected components
- the ResNet trunk

Lior keeps sklearn in his notebook — the rule does not apply to him, and his notebook
is **not to be modified** unless strictly necessary.

A `PostToolUse` hook (`.claude/hooks/check-imports.sh`) blocks these imports in `src/`,
`scripts/`, and `notebooks/`. The escape hatch is `sandbox/` (gitignored, exempt from the
hook) — that is where the one-off "validate my hand-rolled metrics against sklearn on a
fixed array" check lives. It is a throwaway and does not ship.

### VAE, not a plain autoencoder

`beta` must be **~0.1 to 0.001, with warmup**. At `beta=1` the KL flattens exactly the
class structure LP depends on; posterior collapse degenerates it entirely. Beta
sensitivity is a required ablation.

### Justify K

Sweep `K ∈ {5, 10, 15, 20, 30, 50}`, select on the held-out labeled subset. Log
connected-component coverage and edge purity alongside accuracy. **Best-LP-accuracy K is
not automatically best-final-CNN-accuracy K** — report both.

---

## The evaluation protocol (taken from Lior's notebook, not invented)

STL-10 ships **10 official folds** in `fold_indices.txt`, each 1000 indices into the
5000-image labeled train split. Lior fits one linear classifier per fold and evaluates it
on all 8000 test images, reporting the mean over the folds. Inside each fold, an **800/200
stratified development split** does all model selection. Our arm mirrors this exactly.

Two conventions are settled — do not silently change them:

- **One global C, not per-fold.** Every fold's inner CV is averaged across all ten folds
  and all ten are fitted with the single winner, ranked by log loss first (his cell 16).
  `--c-selection per-fold` exists as a sensitivity check and is **not** the comparable row.
- **`±std` is the sample std (ddof=1)** everywhere. Lior's notebook computes no standard
  deviation at all, so that column is ours alone and must never be presented as a
  like-for-like spread against his.

---

## Fairness constraints (make or break the comparison)

Run `/parity-check` before any headline number goes in the report.

- Same trunk, same embedding tap point, **same embedding dim**. Highest-risk item. Lior's
  encoder outputs 512-d pooled features (projection head discarded) — 512-d is the target.
- Same splits: the 10 official folds, the same 800/200 development split protocol, same
  seed, same normalization constants. **The test set is touched once, at the end, by both
  arms** — only `scripts/final_benchmark.py` may read it, behind `--confirm-test-evaluation`.
- Same pretraining budget: 100 epochs, batch 256.
- ≥3 seeds, mean ± std. **Lior's notebook is currently single-seed (SEED=42) — unresolved
  gap, raise it rather than quietly paper over it.**
- Legitimately different, so document rather than force-match: augmentations, batch size,
  optimizer schedule.

### The trunk

Reimplement Lior's SE-ResNet from scratch (no torchvision). This satisfies build-from-0
*and* keeps the trunk identical across both arms, so the pretraining objective stays the
only variable. Verified from his cell 12:

- custom 2-conv stem: Conv3x3 s2 3→48, BN, ReLU, Conv3x3 s1 48→64, **no maxpool**
- pre-activation SE blocks, SE reduction 16
- depths 2-3-4-3, widths 64/128/256/384
- 1x1 conv to 512, global average pool
- input 96×96 native

---

## Pipeline

VAE pretrain on the 100k unlabeled split → freeze → extract **mu** (never a sample, or the
graph changes between runs) → L2-normalize → KNN graph (cosine, heat-kernel weights,
symmetrize by union `W = max(W, Wᵀ)`) → LP (row-normalize to `P`, iterate `F ← PF`,
re-clamp labeled rows every step) → CNN classifier on the pseudo-labels.

**Plus a linear probe on the same embeddings** — that is the honest apples-to-apples row
against Lior, since his downstream is a linear probe and ours is a CNN.

---

## Engineering

- **Shared `.py` modules in `src/`, thin notebooks that call them stage by stage.** Not one
  big notebook — a kernel disconnect loses hours and notebooks merge horribly.
- The similarity matrix must be **chunked**. 105k × 105k fp32 is ~44 GB. Do 1024 rows at a
  time against all 105k columns (~430 MB), mask the diagonal, topk, keep `[N, K]` indices
  and values. Worth a paragraph in the report.
- 100k unlabeled images = 2.76 GB as uint8 → load fully into VRAM, augment on-GPU, skip the
  DataLoader entirely.
- Cache frozen embeddings once (105k × 512 fp16 ≈ 107 MB) so the K sweep runs off the array
  instead of re-encoding.
- Parse `stl10_binary` directly with `numpy.fromfile` rather than pulling in
  `torchvision.datasets` — fewer dependencies and disk is scarce on the server.

---

## Compute

**All training runs on the lab RTX 5090 server. Tomer's laptop has no CUDA GPU (Intel Iris
Xe only) — never suggest training locally.**

- **On the laptop**, driving the server remotely: use `/gpu-run`. It carries the
  connection details, the disk and GPU preflight, and the detach-safe launch pattern.
- **On the server itself**: follow `RUNBOOK.md`. It is the entry point for an agent
  running on `lab-server` — clone location, `scripts/bootstrap_server.sh`, stage order,
  and where results go. Do not use `/gpu-run` there; it would SSH the box into itself.

Local Python note: `python` on this laptop is **2.7**. Always use `py -3.12`. On the server
there is no `python` at all, only `python3` and venvs.

---

## Known findings to flag in the writeup, not fix

- STL-10's unlabeled split is intentionally broader than the 10 labeled classes, so LP
  diffuses labels through out-of-distribution nodes. **Quantify it — it is a finding, not a
  bug.**
- Lior's downstream is a linear probe, ours is a CNN on pseudo-labels, so a straight
  accuracy comparison is confounded by classifier capacity. The linear-probe row resolves it.

---

## Superseded — do not revert to these

Early calls made from the lecture slides alone, before the fuller plan existed:

- beta grid `{0, 0.25, 1.0, 4.0}` — way too high, contradicts the `beta ≤ 0.1` guidance
- 20k-node LP subsample — superseded by full 105k chunked
- one big notebook — superseded by modules
- reusing sklearn's `LogisticRegression`/`StandardScaler` for comparability — forbidden by
  build-from-0
