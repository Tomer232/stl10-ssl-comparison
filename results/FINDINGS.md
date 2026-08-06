# STL-10 semi-supervised comparison — results

Generated 2026-08-06. Every number below is reproducible from `runs/`; the paths
are given so each can be traced to the file that produced it.

**Research question:** does a generative or a contrastive latent space propagate
labels better?

**Answer:** contrastive, by a very large margin — 0.793 vs 0.359 test accuracy on
the like-for-like row, against fold spreads of ±0.005 and ±0.006.

---

## Headline — test split, 10 official folds, 8000 test images each

The test split was opened once, at the end, by each arm.

| arm | downstream | test accuracy | macro-F1 | folds |
|---|---|---|---|---|
| **SimCLR** (P4_full_policy, epoch 100, C=0.1) | linear probe | **0.7933 ± 0.0050** | 0.7934 | 10 |
| **VAE** (β=0.1) | linear probe | **0.3592 ± 0.0060** | 0.3619 | 10 |
| VAE (β=0.1) + label spreading | CNN on ~100k pseudo-labels | 0.4013 ± 0.0060 | 0.3844 | **3** |

The first two rows are the comparison. The classifier, trunk, folds, budget,
optimizer and normalization are all held constant; the pretraining objective is
the only variable. **Gap: 43.4 points.**

The CNN row is confounded by classifier capacity and is reported beside the probe
row, never instead of it. It also covers only folds 0–2, not all ten — do not
present its ±std as comparable to the other two.

Sources:
- ours — `runs/final_benchmark_20260806_101606/summary.json`
- his — `runs/simclr_seed42/results/20260805_124142_984352/final_benchmark/label_efficiency_summary.json`

**On the ± column for his arm:** his notebook reports only means. The ±0.0050
above is computed here, with ddof=1, from his own per-fold test results
(`label_efficiency_fold_results.csv`), using the identical rule applied to ours.
It is therefore comparable — but it is our calculation, not a number he reported.

---

## Development results — three seeds per arm

Selection happened here, on the 200 held-out development-validation images of
each fold. The test split was untouched at this point.

| arm | seed 42 | seed 43 | seed 44 | across seeds |
|---|---|---|---|---|
| SimCLR (P4, ep100) | 0.7630 | 0.7770 | 0.7815 | **0.7738 ± 0.0096** |
| VAE + label spreading (K=15, α=0.9) | 0.4145 | 0.3985 | 0.4015 | **0.4048 ± 0.0085** |
| VAE + linear probe | 0.3540 | 0.3420 | 0.3485 | **0.3482 ± 0.0060** |

Seeds 43/44 were evaluated **at the locked** K/α, not re-selected per seed.

**Seed spread (±0.006–0.010) is roughly four times smaller than fold spread
(±0.021–0.037) in both arms.** The dominant source of variance is which images
land in the development split, not initialization. This matters for reading every
small difference in this report.

---

## Label propagation: the central methodological finding

Hard-clamped Zhu–Ghahramani propagation — `F ← PF` with the labeled rows reset
each step — **fails on this graph**, and the failure is not a bug in the
iteration:

| method | dev accuracy (β=0.1, best config) | mean confidence | iterations to converge |
|---|---|---|---|
| hard clamp (`--method clamp`) | 0.2360 ± 0.0312 | 0.157 | 1133–1326 |
| **label spreading (`--method spread`, α=0.9)** | **0.4145 ± 0.0373** | 0.266 | 20–32 |
| *no graph at all* — cosine 20-NN vote vs the 800 seeds | *0.370* | — | — |

With 800 seeds among 105,000 nodes on a well-connected graph, the random walk
mixes long before it is absorbed, so absorption probabilities stop depending on
where the walk started. The fixed point is nearly flat: mean winning-class share
0.157 against a uniform floor of 0.10. The iteration reaches it correctly (max
delta 6e-8) — the fixed point itself is uninformative.

Two consequences worth stating explicitly in the writeup:

1. **Dev accuracy peaks early and decays *into* the fixed point.** At K=15 it
   peaks at 0.401 around iteration 8, then falls to 0.218 by iteration 300 and
   0.208 by 1000.
2. **An iteration cap silently becomes a hyperparameter.** The original run
   capped at 300 and reported K=5 as best, because small K diffuses more slowly
   and so landed less far down the decay curve. At each K's own optimum the
   ranking reverses: K=20 best (0.404), K=5 worst (0.375). That K selection was
   an artifact of the cap, not a property of the graph.

`labelprop.spread` adds the `(1−α)Y` restart term, whose fixed point is the
discounted random-walk-with-restart score. It converges geometrically at rate α,
unconditionally. Validated in `sandbox/check_spread.py` against the closed form
`(1−α)(I−αP)⁻¹Y` (max error 3e-8) and against sklearn's `LabelSpreading` (100%
argmax agreement at three α values).

### Selected hyperparameters

β=0.1, K=15, α=0.9, C=0.1 — `results/selection.json`, locked 2026-08-06T00:24:09,
before the test split was opened.

α and K optima are both **interior** to their grids ({0.5…0.99}, {5…50}), so
neither grid needs extending. β=0.1 won at the **boundary** of its grid
({0…0.1}); where it turns over is unmeasured. Given the size of the arm gap this
cannot change the conclusion, but it is an open question about the VAE arm alone.

---

## Augmentation ablation (his arm, seed 42)

| policy | dev accuracy | log loss | selected? |
|---|---|---|---|
| **P4_full_policy** | **0.7630 ± 0.0212** | **0.7312** | ✓ (log loss first) |
| P1_crop_color | 0.7005 ± 0.0290 | 0.9378 | |
| P3_crop_flip_color | 0.6935 ± 0.0259 | 0.9367 | |
| P2_crop_color_blur | 0.6885 ± 0.0290 | 0.9533 | |

Paired per-fold tests (both policies scored on identical folds):

- **P4 − P3 = +0.0695, 10/10 folds.** P4 − P1 = +0.0625, 10/10 folds. Unanimous.
- P1 − P2 = +0.0120, 6/10 folds, paired sd 0.0401. **Not separable.**

P1→P2 isolates blur and shows nothing; P3→P4 adds grayscale and blur. So the
entire effect is attributable to `RandomGrayscale(p=0.2)`. Without it the encoder
can satisfy the contrastive objective from colour statistics, which two views of
an image share and other images do not.

**The other three policies are indistinguishable from one another** — they span
0.012, well inside the ±0.026 fold spread. The P1/P3 ordering also flips between
accuracy and log loss, and the selection rule (log loss first) picks P3 over P1
despite P1 having higher accuracy. Report the ablation as "grayscale matters,
nothing else is resolvable at n=10 folds", not as a four-way ranking.

---

## Parity audit

Run 2026-08-06. Full checklist in `.claude/skills/parity-check/SKILL.md`.

| item | verdict | evidence |
|---|---|---|
| Trunk identity | **PASS, proven** | His checkpoint loads into `src/trunk.py` with `strict=True`, 0 missing/unexpected; both 13,385,544 params; forward outputs differ by **0.000e+00** (`sandbox/parity_trunk.py`) |
| Embedding dim | PASS | 512 both |
| Tap point | **FAIL, quantified** | see below |
| Splits | PASS, asymmetric | see below |
| C selection | PASS | one global C, log-loss-first; both arms selected C=0.1 |
| Normalization | PASS, bit-identical | his cell 6 output = `data/normalization.json` to the last digit |
| Test discipline | PASS | test read once per arm, both behind explicit confirmation flags |
| Budget | PASS | 100 epochs, batch 256 |
| Optimizer / schedule | PASS | AdamW, LR 5e-4→1e-5, wd 1e-4, 10 warmup epochs — **identical**, not merely comparable |
| Seeds | PASS | 3 seeds per arm |
| Capacity confound | PASS | probe row leads; CNN row reported beside it |

### Tap point — the one genuine deviation

His tap: trunk → global-avg-pool → 512-d, projection head discarded.
Ours: that same vector → `to_mu` Linear(512→512) → `mu` (`src/vae.py:222`).

Same dimensionality, but not the same layer. Measured impact (β=0.1, seed 42):

| tap | linear probe | label propagation (best K, α) |
|---|---|---|
| `mu` (ours) | 0.3540 | 0.4145 (K=15, α=0.90) |
| pre-`mu` pooled (his layer) | 0.3555 | 0.3935 (K=20, α=0.80) |

Negligible for the probe, as expected — a linear classifier absorbs a linear map.
Worth **0.021 for label propagation**, and it shifts the best K, because cosine
similarity is not invariant under a linear map. Crucially the deviation **favours
our arm**: tapping his layer would make our LP number worse and the gap wider. It
cannot explain the result. (`sandbox/premu_vs_mu_lp.py`)

### Splits — same protocol, effectively independent partitions

Both arms use the 10 official folds and an 800/200 stratified split seeded
`SEED + fold_index`. His comes from sklearn's `train_test_split`; ours is
hand-rolled (sklearn is banned in our arm). Per-fold class distributions match
exactly. The validation sets themselves overlap on **19.2%** of images — against
**20.0% expected for independent draws**.

Consequence: **the two arms' development numbers are not paired.** Paired
per-fold tests are valid within an arm but not between arms. Cheaply fixable if
wanted (our arm reading his partition costs a sweep + probe re-run, ~30 min, no
retraining) — left as-is because the checklist specifies documenting it.

### Remaining asymmetries, documented not fixed

- **Augmentation** — his strong two-view SimCLR policy vs our mild crop/flip.
  Legitimate: SimCLR's objective requires it, a VAE's does not.
- **Determinism** — our arm enables deterministic CUDA kernels
  (`src/seeding.enable_deterministic`); his notebook does not. Part of his
  seed-to-seed spread is kernel nondeterminism, so his ± is slightly
  conservative and cannot be cleanly decomposed.
- **CNN row fold count** — 3 folds, not 10.

---

## Known findings, reported rather than fixed

- **STL-10's unlabeled split is broader than the 10 labeled classes**, so label
  propagation necessarily diffuses mass through out-of-distribution nodes. Per
  `CLAUDE.md` this is quantified, not filtered — any OOD heuristic would be a
  second uncontrolled variable between the arms.
- **The CNN memorizes its pseudo-labels.** It reaches 99.98% agreement with them
  (training cross-entropy 0.0002) while its test cross-entropy is 13.86 — it is
  confidently wrong. It lands at 0.4013 test, below the 0.4145 dev accuracy of
  the label propagation that generated its labels. A CNN on 100k noisy labels
  does not beat the propagation it came from; the capacity confound runs the
  opposite way to the direction anticipated.

---

## Reproducing

```bash
bash scripts/run_all.sh plan          # stage list
bash scripts/run_all.sh all           # prepare -> pretrain -> embed -> sweep -> probe
SELECTED_BETA=0.1 SELECTED_K=15 SELECTED_C=0.1 SELECTED_ALPHA=0.9 \
    bash scripts/run_all.sh lock
SEEDS="43 44" bash scripts/run_all.sh seeds
bash scripts/run_all.sh cnn
CNN_CHECKPOINTS=runs/cnn_.../checkpoints bash scripts/run_all.sh final

# His arm (never modifies the notebook):
python scripts/run_simclr_arm.py                      # 4 policies, ~7 h
python scripts/run_simclr_arm.py --seed 43 --policies P4_full_policy
python scripts/run_simclr_arm.py --test-benchmark runs/simclr_seed42/results/<RUN_ID> \
    --confirm-test-evaluation
```

Long runs must be launched with `setsid nohup … < /dev/null &` — see `RUNBOOK.md`
for why `nohup` alone is not enough.

---

## Files in this directory

```
selection.json                    the locked hyperparameters (β, K, α, C)
FINDINGS.md                       this file

test_benchmark/                   THE TEST SPLIT — read once per arm
  vae_arm/                        scripts/final_benchmark.py output
    summary.json                  both rows, mean ± std over folds
    fold_metrics.csv              per (arm, fold)
    per_class_metrics.csv         per (arm, fold, class)
    confusion_long.csv            long format; fold = -1 is the fold-summed matrix
  simclr_arm/                     his notebook cell 26 output, verbatim
    label_efficiency_summary.json
    label_efficiency_fold_results.csv
    label_efficiency_per_class.csv
    label_efficiency_confusions.csv
    selection.json                policy / epoch / C his development numbers chose

development/                      selection happened here; test never read
  spread_selection.json           best (K, α) per β, and where the diagnostics disagree
  spread_summary.csv              every (β, K, α) cell, mean ± std over the 10 folds
  spread_pivot_dev_accuracy.csv   β × K at the selected α
  spread_pivot_alpha_dev_accuracy.csv   β × α at the selected K
  spread_pivot_edge_purity.csv    β × K edge purity
  clamp_*                         the same for hard clamping — the negative result
  simclr_policy_ablation.csv      his four augmentation policies
```

Raw artifacts not committed (too large, all reproducible): VAE checkpoints,
SimCLR encoders, the 105000 × 512 cached embeddings, per-fold pseudo-labels, and
`test_embeddings.npy`. They live under `runs/` on the lab server.
