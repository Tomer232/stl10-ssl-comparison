# Does a generative or a contrastive latent space propagate labels better?

Semi-supervised classification on STL-10. Final project for *Advanced Methods in Machine
Learning*, Afeka College — **Tomer Karmazin** and **Lior Reznik**.

Everything lives in one notebook: **[`stl10_semi_supervised_comparison.ipynb`](stl10_semi_supervised_comparison.ipynb)**.

---

## The question

STL-10 provides 100,000 unlabeled images and only 1,000 labeled ones per fold. We compare
two ways of learning a representation from the unlabeled images, holding everything else
fixed so that the pretraining objective is the only variable.

| | Generative arm | Contrastive arm |
|---|---|---|
| Objective | β-VAE (evidence lower bound) | SimCLR (NT-Xent) |
| Encoder | SE-ResNet, 13.4M parameters | the same SE-ResNet |
| Embedding | `mu`, 512-d | pooled features, 512-d |
| Downstream | KNN graph → label propagation → CNN, plus a linear probe | linear probe |

The two linear-probe rows are the comparison. The CNN row answers a narrower question and is
reported beside it, never instead of it.

## Two constraints

**Built from zero — for the method.** No SciPy, FAISS, NetworkX, `torchvision.models` or
`torchvision.transforms` anywhere in the pipeline. The KNN graph, label propagation,
union-find, every metric, the SE-ResNet and both augmentation pipelines are written out in
the notebook in PyTorch and NumPy. Section 14 verifies each of them against the
implementation it replaces; those are the only cells that import torchvision, and the
notebook runs to completion without them.

The **evaluation probe** is a deliberate exception. It is hand-written too, but
scikit-learn's `LogisticRegression` is also fitted, on both arms, under the identical
protocol, and section 12 reports the two side by side. A downstream linear classifier is not
part of the method under study, and demonstrating that the conclusion does not depend on
whose logistic regression produced it is worth more than excluding the library.

`tests/check_imports.py` enforces the rule mechanically.

**The test split is opened once.** Every hyperparameter is selected on a held-out 800/200
development split inside each official fold. The selection is written to
`results/selection.json` and hashed before section 12 reads the test set for the first and
only time.

## Protocol

STL-10 ships ten official folds of 1,000 labeled images each. One classifier is fitted per
fold and evaluated on all 8,000 test images; the headline number is the mean over the ten
folds, and every `±` is the sample standard deviation (ddof = 1) across them. Both arms use
seed 42.

## Running it

```bash
git clone https://github.com/Tomer232/stl10-ssl-comparison.git
cd stl10-ssl-comparison
bash scripts/setup.sh                 # virtual environment, PyTorch, STL-10
python tests/check_imports.py         # build-from-zero check
jupyter lab stl10_semi_supervised_comparison.ipynb
```

A cold run needs a CUDA GPU with at least 16 GB and roughly two days — four 100-epoch SimCLR
pretrains, eight 100-epoch VAE pretrains, the sweeps, and ten downstream CNNs. Every heavy
stage caches its output under `runs/`, so an interrupted run resumes rather than starting
over, and a second pass takes minutes. Set `FORCE_RETRAIN = True` in section 0 to ignore the
cache.

See **[RUNBOOK.md](RUNBOOK.md)** for running it unattended on a remote GPU machine.

## Layout

```
stl10_semi_supervised_comparison.ipynb   the project
reference/simclr_stl10_abilation_f.ipynb Lior's original SimCLR implementation, unmodified
scripts/setup.sh                         environment and data setup
tests/check_imports.py                   enforces the build-from-zero rule
results/                                 committed metrics, selection and figures
runs/                                    checkpoints and caches (gitignored, reproducible)
data/                                    stl10_binary (gitignored)
```

The contrastive arm is reimplemented inside the notebook so that one file runs the whole
comparison, and so that both arms share the same augmentation primitives and the same probe.
Lior's original notebook is preserved unmodified under `reference/` as the provenance for
that arm.

## Documented asymmetries

- **Augmentation policies differ by design.** SimCLR's objective requires aggressive views;
  a VAE's does not, because for a VAE the augmented view *is* the reconstruction target.
  Section 3 argues why forcing them to match would distort the comparison rather than fix it.
- **The tap point differs by one linear layer.** The contrastive arm reads the pooled trunk
  feature; the generative arm passes it through `to_mu` first. Section 13.1 measures what
  that is worth to both the probe and the graph.
- **One seed.** Every `±` is the spread over the ten official folds, which is the variance
  this protocol is built around, but neither arm's pretraining variance is measured.
