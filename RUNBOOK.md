# Running this on a remote GPU machine

The notebook is written to be run once, cold, on a machine with a CUDA GPU. A full run is
roughly two days, so it needs to survive a dropped connection. This is how to do that.

## Requirements

| | |
|---|---|
| GPU | CUDA, 16 GB or more. Developed on an RTX 5090 (Blackwell, `sm_120`), which needs **CUDA 12.8 or newer** |
| Disk | at least 12 GB free after the dataset. `setup.sh` checks and refuses to continue otherwise |
| Python | 3.12 |
| RAM | 16 GB is enough; the unlabeled split is memory-mapped rather than loaded |

## Setup

```bash
git clone https://github.com/Tomer232/stl10-ssl-comparison.git
cd stl10-ssl-comparison
bash scripts/setup.sh
```

`setup.sh` is idempotent and trains nothing. It creates `.venv`, installs PyTorch with the
right CUDA build, downloads and extracts STL-10 (deleting the ~2.5 GB tarball immediately),
and runs the build-from-zero check. `bash scripts/setup.sh --check` re-verifies without
changing anything.

## Running unattended

There is no notebook server to keep alive if you execute it headless, which is the
recommended way for the cold run:

```bash
mkdir -p logs
setsid nohup .venv/bin/jupyter nbconvert \
    --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 \
    stl10_semi_supervised_comparison.ipynb \
    > logs/run_$(date +%Y%m%d_%H%M%S).log 2>&1 < /dev/null &
disown
echo "started pid $!"
```

**Use `setsid`, not bare `nohup`.** `nohup` only makes the job ignore `SIGHUP`; the process
stays in the launching shell's process group, so a `kill -- -PGID` — or anything that tears
down the shell it was launched from — still kills a two-day run. `setsid` gives the job its
own session and process group. Verify before walking away:

```bash
ps -o pid,ppid,pgid,sid,stat,cmd -p <pid>
```

You want `PPID 1`, `PGID == SID == PID`, and `Ss` in `STAT`. A process's session cannot be
changed after it starts, so a job launched without `setsid` has to be restarted to be made
safe.

### Suspend

On a desktop-class Linux install, check that the machine will not suspend itself partway
through:

```bash
systemctl status sleep.target suspend.target
```

If they are not masked, an idle window during a two-day run will stop it. Masking them
changes machine-wide behaviour, so on a shared machine agree it with whoever owns the box
first:

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

## Monitoring

```bash
tail -f logs/<logfile>                                    # progress
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv
df -h /                                                   # the notebook writes ~8 GB
ls -la runs/                                              # which stages have completed
```

Each cached stage prints `[cache hit]` or `[building]`, so the log says exactly where the run
is. The order is: normalization → four SimCLR pretrains → eight VAE pretrains → embeddings →
graph survey → propagation studies → sweeps → selection → test benchmark.

## If it stops

Nothing is lost. Re-running the notebook reloads every completed stage from `runs/` and
resumes at the first one that has not finished. Only delete a `runs/*.pt` file if you
actually want that stage recomputed, or set `FORCE_RETRAIN = True` in section 0 to recompute
everything.

Disk is the most likely cause of a mid-run failure. Every stage that writes calls
`require_disk` first and aborts with a clear message rather than filling the volume — a
runaway job that fills the root filesystem takes the whole machine down, which on a shared
box affects other people's work.

## Getting results back

`results/` is small — metrics, the selection file, figures — and is committed. `runs/` holds
checkpoints and embedding caches, is gitignored, and stays on the machine; everything in it
is reproducible from a checkpoint and a seed.

```bash
rsync -avz <host>:<path>/stl10-ssl-comparison/results/ ./results/
```

Never rsync `runs/`, `data/` or `.venv`.
