# results/

**Tracked by git, unlike `runs/`.** This is where the numbers that go in the
report live, so that Lior and anyone reading the repo can see them without
access to the server.

Populated by promoting artifacts out of `runs/`:

```bash
bash scripts/publish_results.sh            # dry run — shows what would change
bash scripts/publish_results.sh --commit   # copy + commit
```

Only `*.csv`, `*.json`, `*.md` and `*.png` are promoted — metrics, summaries,
selection files, figures. Kilobytes.

Checkpoints (`*.pt`) and embedding caches (`*.npy`/`*.npz`, ~107 MB each) stay on
the server: GitHub would reject them, a large blob cannot be removed from git
history without a force-push, and both are reproducible from a checkpoint and a
seed. `publish_results.sh` refuses to commit anything over 5 MB for that reason.

## The one file that gates the final benchmark

`results/selection.json` holds the locked hyperparameters — beta, K, C — all
chosen on development data. `scripts/final_benchmark.py` refuses to run without
it, and it is the only script in the repo that reads the test split. Locking
comes first, test second; the reverse order invalidates the report rather than
just weakening it.
