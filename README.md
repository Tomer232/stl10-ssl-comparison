# STL-10 semi-supervised comparison — VAE + Label Propagation vs SimCLR

Final project for Afeka's *Advanced Methods in Machine Learning*, by **Tomer
Karmazin** and **Lior Reznik**.

> **Does a generative or a contrastive latent space propagate labels better?**

Two arms, one trunk, one protocol, one variable:

| | Tomer's arm (this code) | Lior's arm (`simclr_stl10_abilation_f.ipynb`) |
|---|---|---|
| pretraining | VAE on the 100k unlabeled split | SimCLR on the 100k unlabeled split |
| embedding | `mu`, 512-d | pooled features, 512-d (projection head discarded) |
| downstream | label propagation over a KNN graph → CNN on pseudo-labels, **plus a linear probe** | linear probe |
| libraries | hand-written, no sklearn/scipy/faiss/networkx | sklearn |

The trunk is the same SE-ResNet on both sides — ours reimplemented from scratch
and verified **state-dict compatible** with his (13,385,544 parameters, identical
keys and shapes, bit-identical forward pass after `load_state_dict(strict=True)`).
That is what makes the pretraining objective the only variable.

---

## Running it

**All training happens on the lab RTX 5090 server.** See **[RUNBOOK.md](RUNBOOK.md)**
— clone location, one-command bootstrap, stage order, and the rules about the
test split and the shared GPU.

```bash
bash scripts/bootstrap_server.sh   # setup: venv, torch+cu128, data, splits, smoke test
bash scripts/run_all.sh plan       # what would run, and roughly how long
```

Tomer's laptop has no CUDA GPU; nothing here trains locally. The CPU smoke test
does run anywhere:

```bash
py -3.12 tests/smoke_test.py       # 41 asserts, ~90 s, no pytest needed
```

---

## Layout

```
src/          the pipeline, hand-written under the build-from-0 constraint
  trunk.py        SE-ResNet encoder, state-dict compatible with Lior's
  vae.py          VAE + beta warmup + the loss, with the KL term logged separately
  data.py         stl10_binary parsed with numpy.fromfile, folds, stratified split
  knn.py          chunked cosine KNN graph, heat-kernel weights, union symmetrization
  labelprop.py    F <- PF with the labeled rows re-clamped every iteration
  probe.py        multinomial logistic regression matching sklearn's C convention
  metrics.py      every metric from one bincount confusion matrix
  unionfind.py    connected components
  classifier.py   the CNN trained on propagated pseudo-labels
scripts/      thin drivers: prepare, pretrain, embed, sweep, downstream, benchmark
configs/      run configurations
tests/        smoke_test.py — the whole pipeline on tiny synthetic data
results/      committed summaries (the report reads these)
```

`CLAUDE.md` holds the binding constraints. `.claude/hooks/check-imports.sh`
enforces the build-from-0 rule mechanically.

---

## The constraints that shape the code

- **Build from 0.** No sklearn, scipy, faiss, networkx, or `torchvision.models`
  in Tomer's arm. PyTorch and NumPy only. Lior's notebook keeps sklearn — the
  rule binds one arm, not both.
- **A VAE, not an autoencoder.** `beta` in 0.001–0.1 with warmup. At `beta=1` the
  KL flattens exactly the class structure label propagation depends on.
- **`mu`, never a sample.** Sampling would change the graph between runs.
- **Chunked similarity.** 105k × 105k fp32 is ~44 GB; the graph is built 1024
  rows at a time against all 105k columns (~430 MB).
- **The test set is touched once**, at the end, by both arms, after every
  hyperparameter is locked on development data.

---

## Known asymmetries — reported, not hidden

- Lior's encoder is single-seed (`SEED=42`); ours runs three. The 10 official
  folds give both arms a spread on the downstream, but his *pretraining*
  variance is unmeasured.
- His downstream is a linear probe, ours is a CNN on pseudo-labels, so a
  straight accuracy comparison is confounded by classifier capacity. **The
  linear-probe row on our own embeddings is the apples-to-apples number**, and
  the report leads with it.
- Our stratified 80/20 development split is hand-rolled (sklearn is banned), so
  it follows the same protocol as his at the same seed but is not the identical
  partition.
- Augmentations, batch size and optimizer schedule differ where the objectives
  legitimately require it; forcing them equal would be its own distortion.
