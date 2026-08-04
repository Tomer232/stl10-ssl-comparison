---
name: gpu-run
description: Run training, embedding extraction, or any GPU work for this project on the lab RTX 5090 server (bguserver). Use whenever something needs a GPU — VAE pretraining, the K sweep, the beta ablation, embedding extraction, the CNN classifier — or when the user says "run it", "train", "start the pretrain", "on the server", "kick off the sweep", or asks to check on a run that is already going. Tomer's laptop has no CUDA GPU, so any GPU work at all means this skill.
---

# Running GPU work on bguserver

## First: are you already on the server?

This skill drives the server **from Tomer's laptop**. If you are running *on*
`lab-server` itself, every `ssh bguserver` below would be the machine connecting to
itself. Check before you do anything:

```bash
uname -s    # Linux + hostname lab-server  ->  you are ON the server
```

**If you are on the server, stop and follow `RUNBOOK.md` in the repo instead.** It has
the clone location, `scripts/bootstrap_server.sh`, and the stage order. Come back here
only when you are driving from the laptop.

---

The laptop has no CUDA GPU (Intel Iris Xe). Every GPU job runs on the lab server. You
cannot run these commands yourself in one shot — the SSH connection may prompt for a
password, and the box is shared. Read "Ground rules" before doing anything.

## The machine

| | |
|---|---|
| SSH alias | `bguserver` (already in `~/.ssh/config` → `[REDACTED — see ~/.ssh/config]`, Tailscale) |
| OS | Ubuntu 24.04.3, kernel 6.14, bare metal |
| GPU | 1× RTX 5090, 32 GB VRAM, driver 580.126.09, CUDA 13.0. Blackwell/sm_120 → **cu128 is the minimum** |
| CPU / RAM | Intel Core Ultra 7 265K, 20 cores, 125 GiB RAM |
| Python | `python3` = 3.12.3. **There is no `python` binary** — anything hardcoding `python` breaks |
| Project dir | `/home/labadmin/lab/Tomer_Karmazin/stl10-ssl-comparison/` |
| Multiplexer | **none** — no tmux, no screen |
| Scheduler | **none** — no Slurm, free-for-all on the GPU |

The account `labadmin` is **shared by the whole lab**. Do not touch `hri_env`,
`rrnlp_env`, `hri_sft`, or anyone else's directories — they belong to other people's work.
Our venv and our project dir only.

## Ground rules

1. **Never train on the laptop.** If a GPU is needed and the server is unreachable, stop
   and say so rather than falling back to CPU.
2. **Check the GPU is free before launching.** Nothing arbitrates access; two jobs will
   collide and both will be slow or OOM.
3. **Check disk before anything that writes.** See the disk warning below — it is the most
   likely way a run dies.
4. **Always detach.** There is no tmux, so a dropped SSH kills a foreground job. Use the
   `nohup` pattern below for anything longer than a couple of minutes.
5. **Ask before launching a long run.** A 100-epoch pretrain occupies the lab's only GPU
   for hours. Confirm with Tomer first, and mention that the lab shares the box.

## Disk — read this every time

The server has **one volume**. There is no `/data`, no `/scratch`, no second disk, and
no quotas. It was 99% full when first surveyed; Tomer freed roughly 100 GB on
2026-08-04, but that headroom is shared with the rest of the lab and can vanish without
warning.

Rough budget: a fresh venv with `torch+cu128` is ~4 GB, STL-10 extracted is ~2.5 GB (plus
the ~2.5 GB tarball, which `bootstrap_server.sh` deletes right after extraction), and the
checkpoints and embedding caches accumulate across the beta grid and the seed loop.

Preflight, every session:

```bash
ssh bguserver 'df -h / | tail -1'
```

If free space is under ~20 GB, **say so and stop**. Do not start a download, a `pip
install`, or a training run. A runaway job that fills `/` takes down the whole lab's box,
and there are no quotas to stop it. Freeing space is Tomer's call to make with the lab —
never delete other people's files to make room.

## Idle-suspend — fix before the first overnight run

GNOME idle-suspend is enabled and `sleep.target` / `suspend.target` are not masked, so
**the machine can suspend itself mid-training**. Before relying on any unattended run,
this needs (with sudo, by whoever owns the box):

```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
```

Flag this to Tomer rather than running it unprompted — it changes machine-wide behaviour
on a shared box.

## First-time setup

### Passwordless SSH

Right now `ssh bguserver` prompts for a password every time, which makes automated runs
painful. One-time fix, run by Tomer (the `ssh-copy-id` step needs the password
interactively, so it cannot be done from a tool call):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/bguserver_tomer -C "tomer-laptop"
ssh-copy-id -i ~/.ssh/bguserver_tomer.pub bguserver
```

then add `IdentityFile ~/.ssh/bguserver_tomer` to the `Host bguserver` block in
`~/.ssh/config`. The shared `authorized_keys` already holds one key (`[REDACTED]`); this
appends a second, it does not replace anything.

### Clone and bootstrap

The repo carries its own setup script, so this is two commands rather than a checklist.
`PROJECT` below is the canonical location — everything the project writes stays inside
Tomer's own folder on this shared account.

```bash
PROJECT=~/lab/Tomer_Karmazin/stl10-ssl-comparison

ssh bguserver "mkdir -p ~/lab/Tomer_Karmazin && cd ~/lab/Tomer_Karmazin && \
  git clone https://github.com/Tomer232/stl10-ssl-comparison.git"

ssh bguserver "cd $PROJECT && bash scripts/bootstrap_server.sh"
```

`bootstrap_server.sh` is idempotent and trains nothing. It checks disk and the GPU, warns
about idle-suspend, sets a repo-local git identity (the account's global identity belongs
to another lab member), builds `.venv` with `torch==2.11.0+cu128` — never touching
`hri_env` or `rrnlp_env` — downloads and extracts STL-10, builds the splits and
normalization constants, and finishes by running the smoke test and the build-from-0 hook.

`bash scripts/bootstrap_server.sh --check` re-verifies without changing anything.

If the smoke test fails, **do not launch training.**

## Getting code across

The laptop is the source of truth; the server only ever pulls.

```bash
git push                                              # laptop
ssh bguserver "cd $PROJECT && git pull --ff-only"     # server
```

For fast iteration on uncommitted work, `rsync` (installed on both ends) is fine:

```bash
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'data' --exclude 'runs' \
  ./src/ bguserver:~/lab/Tomer_Karmazin/stl10-ssl-comparison/src/
```

Never rsync `data/`, `runs/`, or `.venv` — disk is the binding constraint.

## Preflight, then launch

Always run the preflight as one command before launching:

```bash
ssh bguserver 'df -h / | tail -1; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv'
```

Interpretation:

- **Compute-apps list empty** → GPU free, safe to launch. (Xorg and gnome-shell hold ~127
  MiB for the desktop; that is normal and not a training job.)
- **Someone else's python holding VRAM** → do not launch. Tell Tomer who is on the GPU and
  let him coordinate.
- **Free disk under ~20 GB** → do not launch.

See what a stage would do before running it:

```bash
ssh bguserver "cd $PROJECT && bash scripts/run_all.sh plan"
```

Launch, detached (no tmux available):

```bash
ssh bguserver "cd $PROJECT && mkdir -p logs && \
  nohup bash scripts/run_all.sh pretrain > logs/pretrain_\$(date +%Y%m%d_%H%M%S).log 2>&1 & \
  echo \"started pid \$!\""
```

Stages, in order: `prepare` → `pretrain` → `embed` → `sweep` → `probe` → `lock` →
`seeds` → `cnn`. **`final` is separate, reads the test split, and runs once at the very
end** — never fold it into a batch.

`run_all.sh` invokes python with `-u`; without it stdout buffers and an 8 KB delay looks
exactly like a hung job.

Note the PID and the log path in your reply so the run can be found again.

## Monitoring

```bash
# progress
ssh bguserver 'tail -n 40 ~/lab/Tomer_Karmazin/final_project/runs/<logfile>'

# still alive?
ssh bguserver 'pgrep -af "pretrain_vae.py"'

# GPU actually working?
ssh bguserver 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'

# disk not filling up?
ssh bguserver 'df -h / | tail -1'
```

For a long run, prefer one check every several minutes over tight polling — each check is
a fresh SSH connection and, until keys are set up, a password prompt for Tomer.

Kill a run: `ssh bguserver 'kill <pid>'`, then confirm the VRAM was released with
`nvidia-smi`.

## Pulling results back

Bring back small artifacts — logs, metrics, the embedding cache (105k × 512 fp16 ≈ 107 MB),
figures. Leave large checkpoints on the server unless there is a reason to move them; the
laptop does not have a GPU to use them with anyway.

```bash
rsync -avz bguserver:~/lab/Tomer_Karmazin/stl10-ssl-comparison/runs/ ./runs/
```

`runs/` is gitignored. The repo has a script that promotes just the reportable artifacts
into the tracked `results/` directory, refusing anything over 5 MB so a checkpoint cannot
leak into git history:

```bash
ssh bguserver "cd ~/lab/Tomer_Karmazin/stl10-ssl-comparison &&   bash scripts/publish_results.sh"            # dry run first
```

Publishing to GitHub is how Lior gets access to the numbers. A push from the server may
fail (shared account, no credential helper) — the commit is still made and can be pushed
from the laptop.

## STL-10 data

`bootstrap_server.sh` downloads and extracts it (deleting the ~2.5 GB tarball immediately)
and builds `data/splits.npz` and `data/normalization.json`. There is nothing to do by hand.

It is parsed with `numpy.fromfile` rather than `torchvision.datasets` — fewer dependencies,
and the column-major transpose is handled explicitly in `src/data.py`. `data/` is
gitignored; the dataset never enters the repo.
