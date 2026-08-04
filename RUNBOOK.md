# RUNBOOK — running this project on the lab RTX 5090 server

**If you are an agent running ON the server (`lab-server`, Ubuntu, one RTX 5090):
this file is your entry point. Read `CLAUDE.md` too — it carries the constraints
— but take your operating instructions from here.**

If you are running on Tomer's Windows laptop instead, stop: there is no CUDA GPU
there, nothing in this project trains locally, and the `/gpu-run` skill is the
laptop-side path. This file is for the machine with the GPU.

---

## 0. Where this lives

```
/home/labadmin/lab/Tomer_Karmazin/stl10-ssl-comparison/
```

Clone it there and nowhere else. Everything the project produces — `runs/`,
`results/`, `logs/`, `data/` — lands inside that directory, which is what keeps
all output inside Tomer's own folder on a machine whose `labadmin` account is
shared by the whole lab.

```bash
mkdir -p ~/lab/Tomer_Karmazin
cd ~/lab/Tomer_Karmazin
git clone <repo-url> stl10-ssl-comparison
cd stl10-ssl-comparison
```

---

## 1. Setup — one command

```bash
bash scripts/bootstrap_server.sh
```

Idempotent, safe to re-run, and it trains nothing. It checks disk and the GPU,
warns if the machine can still suspend itself mid-run, sets a repo-local git
identity, builds `.venv` with `torch==2.11.0+cu128`, downloads and extracts
STL-10 (deleting the tarball), builds the splits and normalization constants,
and finally runs the smoke test and the build-from-0 hook.

`bash scripts/bootstrap_server.sh --check` verifies without changing anything.

If the smoke test fails, **do not start training.** Fix it first — a failure
there means the pipeline is broken in a way that would waste hours of GPU time
and produce numbers nobody can defend.

---

## 2. What to run

```bash
bash scripts/run_all.sh plan
```

Prints the stage table and the rough wall-clock per stage, and runs nothing.
Read it before starting anything.

Stages, in order: `prepare` → `pretrain` → `embed` → `sweep` → `probe` → `lock`
→ `seeds` → `cnn`, and then `final` on its own at the very end.

**There is no tmux on this box.** A dropped SSH connection kills a foreground
job, so every long stage must be detached:

```bash
mkdir -p logs
nohup bash scripts/run_all.sh pretrain > logs/pretrain.log 2>&1 &
echo "pid $!"
tail -f logs/pretrain.log
```

Check on a run:

```bash
pgrep -af run_all.sh                                    # still alive?
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
df -h / | tail -1                                       # disk not filling?
tail -n 40 logs/pretrain.log
```

---

## 3. Rules that are not negotiable

**Never run `final` early.** Stage `final` is the only thing in this repo that
reads the test split. It is deliberately excluded from `all`, refuses to start
without `results/selection.json`, and requires `--confirm-test-evaluation`. Every
hyperparameter — beta, K, C — is chosen on the 200 held-out development images
inside each fold, never on test. Touching test before the selection is locked
does not weaken the report; it invalidates it entirely.

**Never install `sklearn`, `scipy`, `faiss`, or `networkx`.** The lecturer's
"build from 0" constraint requires the KNN graph, label propagation, the metrics,
union-find, the ResNet trunk and the logistic-regression probe to be hand-written.
A `PostToolUse` hook rejects those imports in `src/`, `scripts/`, `notebooks/` and
`tests/`. `sandbox/` is the deliberate, gitignored escape hatch for a throwaway
check; nothing there ships.

**Never edit `simclr_stl10_abilation_f.ipynb`.** It is Lior's arm, kept here as
the reference the fairness audit reads. Changing it breaks the comparison.

**Coordinate before taking the GPU.** One card, no scheduler, `labadmin` shared
by the lab. Check `nvidia-smi --query-compute-apps=...` first and, if someone
else's process is on it, say so rather than launching alongside them.

**Watch the disk.** One volume, no quotas. A job that fills `/` takes down the
whole lab's machine. Never delete other people's files to make room — ask.

---

## 4. Where the results go

| directory | tracked by git? | what it holds |
|---|---|---|
| `data/` | no | STL-10 plus `splits.npz` and `normalization.json` |
| `runs/` | no | everything a stage emits: checkpoints, embedding caches, logs, metrics |
| `results/` | **yes** | the small reportable artifacts, promoted from `runs/` |
| `logs/` | no | nohup output |

Promote the reportable artifacts when the numbers are final:

```bash
bash scripts/publish_results.sh              # dry run, shows what would change
bash scripts/publish_results.sh --commit     # copies CSV/JSON/PNG into results/ and commits
```

It refuses to commit anything over 5 MB, so checkpoints and the ~107 MB
embedding caches cannot leak into git history by accident. They stay on the
server and are reproducible from a checkpoint and a seed.

Pushing may fail from the server: the shared account has no git credential
helper and its `gh` login belongs to another lab member. The commit is made
regardless and can be pushed from the laptop. Publishing to GitHub is how Lior
gets access to the numbers.

---

## 5. Two decisions already made — do not silently change them

**One global C, not per-fold.** The linear probe averages every fold's inner CV
across all ten folds and fits all ten with the single winner, which is exactly
what Lior's cell 16 does. `--c-selection per-fold` exists as a sensitivity check
and is *not* the row comparable to his arm; the default is `global`.

**`±std` is the sample std (ddof=1)** everywhere in this project. Lior's notebook
reports fold means only and computes no standard deviation at all, so that column
is ours alone and must not be presented as a like-for-like spread against his.

---

## 6. If something looks wrong

Report it rather than working around it. In particular:

- **KL term collapsing toward 0** during pretraining is posterior collapse — the
  failure mode the low beta grid exists to avoid. It is visible per-epoch in the
  log on purpose.
- **Label propagation hitting the iteration cap** without converging means the
  reported accuracies are a snapshot of a still-moving iteration. The cap is 300
  because convergence measured at 107–118 iterations; if it is being hit, say so.
- **Best-K by accuracy disagreeing with best-K by edge purity or coverage** is
  expected and reportable, not a bug to tune away.
- **Label mass diffusing through out-of-distribution nodes** is expected too:
  STL-10's unlabeled split is deliberately broader than the 10 labeled classes.
  Quantify it; it is a finding.
