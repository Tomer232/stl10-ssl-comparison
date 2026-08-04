---
name: gpu-run
description: Run training, embedding extraction, or any GPU work for this project on the lab RTX 5090 server (bguserver). Use whenever something needs a GPU — VAE pretraining, the K sweep, the beta ablation, embedding extraction, the CNN classifier — or when the user says "run it", "train", "start the pretrain", "on the server", "kick off the sweep", or asks to check on a run that is already going. Tomer's laptop has no CUDA GPU, so any GPU work at all means this skill.
---

# Running GPU work on bguserver

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
| Project dir | `/home/labadmin/lab/Tomer_Karmazin/` |
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

The server has **one volume, and it was 99% full (~13 GB free)** when last surveyed.
There is no `/data`, no `/scratch`, no second disk.

Rough budget: a fresh venv with `torch+cu128` is ~4 GB, STL-10 extracted is ~2.5 GB (plus
the ~2.5 GB tarball, which should be deleted right after extraction). That is most of the
headroom gone before a single checkpoint is written.

Preflight, every session:

```bash
ssh bguserver 'df -h / | tail -1'
```

If free space is under ~15 GB, **say so and stop**. Do not start a download, a `pip
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

### The project venv

Ours alone — never install into `hri_env` or `rrnlp_env`.

```bash
ssh bguserver 'python3 -m venv ~/lab/Tomer_Karmazin/.venv && \
  ~/lab/Tomer_Karmazin/.venv/bin/pip install --upgrade pip'
ssh bguserver '~/lab/Tomer_Karmazin/.venv/bin/pip install \
  --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0'
ssh bguserver '~/lab/Tomer_Karmazin/.venv/bin/pip install numpy matplotlib'
```

`torch 2.11.0+cu128` is the version already proven working against this GPU in `hri_env`.
Do not install `sklearn` or `scipy` — build-from-0 forbids them, and `requirements.txt`
omits them deliberately.

Verify:

```bash
ssh bguserver '~/lab/Tomer_Karmazin/.venv/bin/python -c \
  "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

Expect `2.11.0+cu128 12.8 True NVIDIA GeForce RTX 5090`.

### Git identity on the server

The server's global git identity is **Yuval Zohar's** (`yuvalzohar12@gmail.com`, shared
account). Set a per-repo identity immediately after cloning, or commits made there will be
attributed to him:

```bash
ssh bguserver 'cd ~/lab/Tomer_Karmazin/final_project && \
  git config user.name "Tomer Karmazin" && git config user.email "tomer@jeepsea.co.il"'
```

## Getting code across

The laptop is the source of truth. The server holds a checkout that we only ever pull into.

```bash
# laptop: commit and push first
git push

# server: pull
ssh bguserver 'cd ~/lab/Tomer_Karmazin/final_project && git pull --ff-only'
```

For fast iteration on uncommitted work, `rsync` (installed on both ends) is fine:

```bash
rsync -avz --exclude '.git' --exclude '.venv' --exclude 'data' \
  ./src/ bguserver:~/lab/Tomer_Karmazin/final_project/src/
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
- **Free disk under ~15 GB** → do not launch.

Launch, detached (no tmux available):

```bash
ssh bguserver 'cd ~/lab/Tomer_Karmazin/final_project && \
  mkdir -p runs && \
  nohup .venv/bin/python -u scripts/pretrain_vae.py --config configs/vae_base.yaml \
    > runs/pretrain_$(date +%Y%m%d_%H%M%S).log 2>&1 & \
  echo "started pid $!"'
```

`-u` matters — without it Python buffers stdout and the log stays empty for ages, which
looks like a hung job.

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
rsync -avz bguserver:~/lab/Tomer_Karmazin/final_project/runs/ ./runs/
```

`runs/` is gitignored. Anything that belongs in the report gets copied into `results/` and
committed deliberately.

## STL-10 data

Not present on the server as of the last survey. It has to be downloaded there once —
check disk first, and delete the tarball immediately after extracting.

```bash
ssh bguserver 'cd ~/lab/Tomer_Karmazin/final_project && mkdir -p data && cd data && \
  curl -L -O http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz && \
  tar xzf stl10_binary.tar.gz && rm stl10_binary.tar.gz && du -sh stl10_binary'
```

Parse `stl10_binary` directly with `numpy.fromfile` — no `torchvision.datasets`. `data/` is
gitignored; the dataset never enters the repo.
