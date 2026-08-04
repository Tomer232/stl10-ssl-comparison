#!/usr/bin/env bash
#
# End-to-end pipeline for Tomer's arm (VAE -> mu -> KNN graph -> label
# propagation -> downstream), written for the lab RTX 5090 server.
#
#   bash scripts/run_all.sh plan          # print the stage table and quit
#   nohup bash scripts/run_all.sh all > logs/run_all.log 2>&1 &
#   bash scripts/run_all.sh sweep         # or one stage at a time
#
# NOTHING RUNS UNTIL YOU NAME A STAGE. Sourcing this file or running it with no
# argument prints the plan and exits, because "all" is measured in DAYS of GPU
# time and nobody should start it by accident.
#
# WHY A SHELL SCRIPT AND NOT A NOTEBOOK
# -------------------------------------
# The stages are hours apart and the connection to the server will drop at least
# once. Every stage writes its artifacts to a deterministic directory under
# runs/, so a stage that dies can be rerun on its own and the stage after it
# still knows where to look. A notebook would lose the kernel and the state.
#
# THE TEST SPLIT
# --------------
# Stage `final` is the only one that reads it, it is deliberately NOT part of
# `all`, and it refuses to start without results/selection.json. Run it once,
# by hand, at the end of the project. See scripts/final_benchmark.py.
#
# SERVER NOTES
# ------------
#   * there is no `python` on the server, only `python3` and venvs -- hence
#     PYTHON below, which defaults to the project venv;
#   * `-u` on every invocation so nohup's log updates live rather than in 8 KB
#     bursts, which is the difference between "it is training" and "it hung";
#   * disk is the binding constraint: one volume, no quotas, shared with the
#     rest of the lab. The VAE checkpoints are the bulk of what this writes --
#     see the note in stage `pretrain`. Check `df -h /` before a long stage.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration. Everything here is overridable from the environment, e.g.
#   SEED=43 BETAS="0.01" bash scripts/run_all.sh pretrain
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUNS="${RUNS:-$REPO_ROOT/runs}"
RESULTS="${RESULTS:-$REPO_ROOT/results}"
LOGS="${LOGS:-$REPO_ROOT/logs}"

# The protocol seed. The 10 official folds and their 800/200 development splits
# are built ONCE at this seed and never re-drawn: they are the evaluation
# protocol both arms share, not a source of randomness to average over.
SEED="${SEED:-42}"

# Seeds for the >= 3-seed requirement in CLAUDE.md's fairness section. Only the
# SELECTED beta is repeated -- three seeds x six betas of pretraining is a week
# of GPU time and answers nothing the beta sweep has not already answered.
# (Lior's arm is currently single-seed. That gap is real; raise it, do not
# quietly paper over it by reporting our spread against his point estimate.)
# NOTE: 42 is the protocol seed, so stage `seeds` repeats the run that stage
# `probe` already produced. That is deliberate -- it keeps the three-seed table
# self-contained and lets it be read without cross-referencing another run
# directory. Set SEEDS="43 44" to skip the repeat.
SEEDS="${SEEDS:-42 43 44}"

# The beta grid, from config.BETA_GRID. 0.0 is the plain-autoencoder control:
# it isolates how much of the result comes from the *variational* part rather
# than from reconstruction alone. Nothing above 0.1 -- by beta = 1 the KL has
# flattened exactly the class structure label propagation depends on.
BETAS="${BETAS:-0.0 0.001 0.003 0.01 0.03 0.1}"

# K grid for the graph sweep, from config.K_GRID.
K_GRID="${K_GRID:-5 10 15 20 30 50}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"

# Folds to train the downstream CNN on. NOT all ten by default: one fold is
# ~100 epochs over 100k images, so ten folds is more than a day. Three folds
# give a usable fold-to-fold spread for a row that is, by construction, the
# secondary one. Set CNN_FOLDS=all if there is time.
CNN_FOLDS="${CNN_FOLDS:-0,1,2}"

SPLITS_PATH="$DATA_ROOT/splits.npz"
NORMALIZATION_PATH="$DATA_ROOT/normalization.json"
SELECTION_PATH="$RESULTS/selection.json"

mkdir -p "$RUNS" "$RESULTS" "$LOGS"

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

banner() {
    echo ""
    echo "=============================================================================="
    echo "== $1"
    echo "== estimated wall clock: $2"
    echo "== started $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=============================================================================="
    echo ""
}

# Stages are idempotent by refusing to overwrite: every script errors out on an
# existing output directory rather than clobbering results someone is already
# reading. This lets a rerun skip the parts that finished.
skip_if_done() {
    local marker="$1"
    local what="$2"
    if [ -e "$marker" ]; then
        echo "SKIP $what -- $marker already exists"
        return 0
    fi
    return 1
}

require_file() {
    if [ ! -e "$1" ]; then
        echo "MISSING: $1" >&2
        echo "  $2" >&2
        exit 1
    fi
}

# Directory naming. These are deterministic (no timestamp) precisely so a later
# stage can find an earlier stage's output without parsing `ls`.
vae_dir()   { echo "$RUNS/vae_beta${1}_seed${2}"; }
embed_dir() { echo "$RUNS/emb_beta${1}_seed${2}"; }
sweep_dir() { echo "$RUNS/sweep_seed${1}"; }

# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

stage_plan() {
    cat <<'PLAN'

STAGES, IN ORDER, WITH ESTIMATED WALL CLOCK ON THE RTX 5090
===========================================================

  prepare    verify STL-10, channel stats, folds + dev splits     ~5 min
  pretrain   VAE pretraining, ONE RUN PER BETA (6 betas)          ~5 h each -> ~30 h
  embed      cache frozen mu for each beta (105k x 512 fp16)      ~5 min each -> ~30 min
  sweep      beta x K graph + label propagation, dev-scored       ~2 h total
  probe      linear probe per beta -- THE APPLES-TO-APPLES ROW    ~2 min each
  ---------- read the sweep, choose beta / K / C, then: ----------
  lock       write results/selection.json (needs your numbers)    instant
  seeds      repeat pretrain+embed+probe at the extra seeds       ~5 h each -> ~10 h
  cnn        downstream CNN on propagated pseudo-labels           ~2.5 h per fold
  ---------- once, at the very end, by hand: --------------------
  final      TEST SPLIT. Both rows. Requires selection.json.      ~10 min

  `all` runs prepare -> pretrain -> embed -> sweep -> probe and STOPS at the
  lock, because everything after it depends on a number a human has to choose.
  That is roughly 33 HOURS. Start it on a Friday.

THESE ARE ESTIMATES, NOT MEASUREMENTS
-------------------------------------
The pretraining figure assumes ~3 min/epoch for the SE-ResNet trunk plus the
decoder at 96x96, batch 256, with deterministic kernels on (they cost maybe
10-20%, and they are not optional -- if only one arm were deterministic, any
gap between the two latent spaces could just be run-to-run variance). Watch the
first two epochs in the log and multiply; if epoch 1 is 8 minutes, `all` is
three days and the beta grid needs trimming, not the epoch count -- the 100
epochs are budget parity with Lior and are not ours to change.

DISK
----
Each VAE checkpoint pair (best.pt + final.pt) is a few hundred MB and the
server has ~13 GB free on one volume. Six betas will be tight. Delete the
best.pt of a beta once its embeddings are cached -- the embeddings are what
every later stage reads, and 105k x 512 fp16 is only 107 MB.

PLAN
}

# ---------------------------------------------------------------------------
# Stage 1: data preparation
# ---------------------------------------------------------------------------

stage_prepare() {
    banner "prepare -- verify STL-10, channel statistics, folds and dev splits" "~5 min"

    if skip_if_done "$SPLITS_PATH" "prepare"; then return 0; fi

    # Computes the per-channel mean/std from the UNLABELED split only (using
    # train or test would leak evaluation data into preprocessing) and writes
    # the 10 official folds with their stratified 800/200 development splits.
    # Both arms read these files, so they are built once and never rewritten.
    "$PYTHON" -u scripts/prepare_data.py \
        --data-root "$DATA_ROOT" \
        --seed "$SEED" \
        --splits-path "$SPLITS_PATH" \
        --normalization-path "$NORMALIZATION_PATH"
}

# ---------------------------------------------------------------------------
# Stage 2: VAE pretraining -- the beta ablation
# ---------------------------------------------------------------------------

stage_pretrain() {
    banner "pretrain -- VAE, one run per beta (${BETAS}), seed ${SEED}" "~5 h per beta"

    require_file "$NORMALIZATION_PATH" "run: bash scripts/run_all.sh prepare"

    for beta in $BETAS; do
        local directory
        directory="$(vae_dir "$beta" "$SEED")"

        if skip_if_done "$directory/checkpoints/final.pt" "pretrain beta=$beta"; then
            continue
        fi

        echo ""
        echo "--- beta = $beta (seed $SEED) -> $directory"
        # beta is warmed up linearly over the first epochs so the KL term cannot
        # dominate before the decoder can reconstruct anything, which is the
        # usual cause of immediate posterior collapse. --beta is the MAXIMUM.
        "$PYTHON" -u scripts/pretrain_vae.py \
            --config configs/vae_base.json \
            --data-root "$DATA_ROOT" \
            --normalization-path "$NORMALIZATION_PATH" \
            --output-dir "$directory" \
            --beta "$beta" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --seed "$SEED"
    done
}

# ---------------------------------------------------------------------------
# Stage 3: cache the frozen embeddings
# ---------------------------------------------------------------------------

stage_embed() {
    banner "embed -- cache frozen mu for each beta (105k x 512 fp16)" "~5 min per beta"

    for beta in $BETAS; do
        local checkpoint directory
        checkpoint="$(vae_dir "$beta" "$SEED")/checkpoints/final.pt"
        directory="$(embed_dir "$beta" "$SEED")"

        if skip_if_done "$directory/embeddings.npy" "embed beta=$beta"; then
            continue
        fi
        require_file "$checkpoint" "run: BETAS=$beta bash scripts/run_all.sh pretrain"

        echo ""
        echo "--- beta = $beta -> $directory"
        # mu, never a posterior sample: a sample would perturb every pairwise
        # cosine similarity, reshuffle the top-K near ties, and change the graph
        # between runs -- the beta ablation would then be measuring noise.
        # The test split is not loaded here; extract_embeddings.py cannot read it.
        "$PYTHON" -u scripts/extract_embeddings.py \
            --checkpoint "$checkpoint" \
            --data-root "$DATA_ROOT" \
            --output-directory "$directory" \
            --output-stem embeddings \
            --no-memory-map \
            --seed "$SEED"
    done
}

# ---------------------------------------------------------------------------
# Stage 4: the beta x K sweep
# ---------------------------------------------------------------------------

stage_sweep() {
    banner "sweep -- KNN graph + label propagation over beta x K, scored on dev" "~2 h"

    local directory
    directory="$(sweep_dir "$SEED")"
    if skip_if_done "$directory/selection.json" "sweep"; then return 0; fi

    # One invocation over every beta: the script builds the graph once per
    # (beta, K) and propagates ten times off it, once per official fold, so a
    # single beta x K table comes out with the folds already aggregated.
    local embedding_arguments=()
    local label_arguments=()
    for beta in $BETAS; do
        local array
        array="$(embed_dir "$beta" "$SEED")/embeddings.npy"
        require_file "$array" "run: bash scripts/run_all.sh embed"
        embedding_arguments+=("$array")
        label_arguments+=("beta=$beta")
    done

    # Component coverage and edge purity are logged next to accuracy for every
    # K. They are not decoration: accuracy alone cannot tell a bad embedding
    # from a graph shattered into islands no label can reach, and CLAUDE.md is
    # explicit that the best-LP-accuracy K need not be the best K on either
    # diagnostic. When they disagree the script says so -- report that.
    "$PYTHON" -u scripts/sweep_knn_lp.py \
        --embeddings "${embedding_arguments[@]}" \
        --labels "${label_arguments[@]}" \
        --data-root "$DATA_ROOT" \
        --splits "$SPLITS_PATH" \
        --k-grid $K_GRID \
        --output-directory "$directory" \
        --seed "$SEED"

    echo ""
    echo "sweep tables:   $directory/summary.csv"
    echo "sweep choice:   $directory/selection.json  (dev-only, K per embedding)"
}

# ---------------------------------------------------------------------------
# Stage 5: the linear probe -- the row the report leads with
# ---------------------------------------------------------------------------

stage_probe() {
    banner "probe -- hand-rolled logistic regression per beta (APPLES-TO-APPLES ROW)" "~2 min per beta"

    # This is the honest comparison against Lior: his downstream is a linear
    # probe too, so this is the only row where the classifier is held constant
    # and the pretraining objective is the single variable. The CNN row in
    # stage `cnn` answers a narrower question and is confounded by capacity.
    for beta in $BETAS; do
        local array
        array="$(embed_dir "$beta" "$SEED")/embeddings.npy"
        require_file "$array" "run: bash scripts/run_all.sh embed"

        echo ""
        echo "--- beta = $beta"
        # Scored on each fold's 200 held-out development-validation images.
        # Never on test.
        "$PYTHON" -u scripts/train_downstream.py \
            --mode probe \
            --embeddings "$array" \
            --splits "$SPLITS_PATH" \
            --data-root "$DATA_ROOT" \
            --name "probe_beta${beta}_seed${SEED}" \
            --seed "$SEED"
    done

    echo ""
    echo "Each run wrote probe_selection_candidate.json with the C the development"
    echo "split chose. That C is an INPUT to the lock below, not a selection."
}

# ---------------------------------------------------------------------------
# Stage 6: the lock -- a deliberate human act
# ---------------------------------------------------------------------------

stage_lock() {
    banner "lock -- freeze beta, K and C into results/selection.json" "instant"

    # Deliberately NOT automatic. The final benchmark refuses to run without
    # this file, and the file exists to record that a person looked at the
    # development numbers and chose, BEFORE the test split was opened. A script
    # that derived it from the sweep's argmax would make the ceremony
    # decorative; a script that derived it after seeing test would invalidate
    # the entire report. So the numbers come from the environment, and whoever
    # sets them is the person on the hook for them.
    if [ -e "$SELECTION_PATH" ]; then
        echo "$SELECTION_PATH already exists -- refusing to relock."
        echo "If a hyperparameter genuinely changed, move the old file aside and say"
        echo "so in the report. Silently relocking after seeing a number is the one"
        echo "thing this file exists to prevent."
        cat "$SELECTION_PATH"
        return 0
    fi

    if [ -z "${SELECTED_BETA:-}" ] || [ -z "${SELECTED_K:-}" ] || [ -z "${SELECTED_C:-}" ]; then
        cat <<LOCK

Read these first:
  $(sweep_dir "$SEED")/summary.csv          accuracy / purity / coverage per (beta, K)
  $(sweep_dir "$SEED")/selection.json       best K per beta on dev, and any disagreement
  runs/probe_beta*_seed${SEED}_*/summary.json   the probe row, and the C each fold chose

Then lock them in, e.g.

  SELECTED_BETA=0.01 SELECTED_K=15 SELECTED_C=10 \\
      bash scripts/run_all.sh lock

LOCK
        exit 1
    fi

    "$PYTHON" - "$SELECTION_PATH" "$SELECTED_BETA" "$SELECTED_K" "$SELECTED_C" "$SEED" <<'PY'
import datetime
import json
import sys

path, beta, k, c, seed = sys.argv[1:6]

selection = {
    "selected_beta": float(beta),
    "selected_k": int(k),
    "selected_c": float(c),
    "selection_split": "development",
    "test_seen_during_selection": False,
    "locked_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "protocol_seed": int(seed),
    "note": "chosen on the 200 held-out development-validation images of each "
            "official fold, before the test split was opened",
}

with open(path, "w", encoding="utf-8") as handle:
    json.dump(selection, handle, indent=2, sort_keys=True)
    handle.write("\n")

print("locked -> " + path)
print(json.dumps(selection, indent=2, sort_keys=True))
PY
}

# ---------------------------------------------------------------------------
# Stage 7: the extra seeds
# ---------------------------------------------------------------------------

stage_seeds() {
    require_file "$SELECTION_PATH" "run the lock first: bash scripts/run_all.sh lock"

    local beta
    beta="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_beta"])' \
        "$SELECTION_PATH")"

    banner "seeds -- repeat pretrain + embed + probe at beta=$beta for seeds: $SEEDS" "~5 h per extra seed"

    # >= 3 seeds, mean +/- std, per CLAUDE.md. Only the selected beta is
    # repeated. The folds and development splits are NOT re-drawn per seed --
    # they are protocol -- so the seed-to-seed spread is training noise alone
    # and not split noise mixed in with it.
    for seed in $SEEDS; do
        local vae embeddings
        vae="$(vae_dir "$beta" "$seed")"
        embeddings="$(embed_dir "$beta" "$seed")"

        echo ""
        echo "--- seed $seed"

        if [ ! -e "$vae/checkpoints/final.pt" ]; then
            "$PYTHON" -u scripts/pretrain_vae.py \
                --config configs/vae_base.json \
                --data-root "$DATA_ROOT" \
                --normalization-path "$NORMALIZATION_PATH" \
                --output-dir "$vae" \
                --beta "$beta" \
                --epochs "$EPOCHS" \
                --batch-size "$BATCH_SIZE" \
                --seed "$seed"
        else
            echo "SKIP pretrain -- $vae/checkpoints/final.pt exists"
        fi

        if [ ! -e "$embeddings/embeddings.npy" ]; then
            "$PYTHON" -u scripts/extract_embeddings.py \
                --checkpoint "$vae/checkpoints/final.pt" \
                --data-root "$DATA_ROOT" \
                --output-directory "$embeddings" \
                --output-stem embeddings \
                --no-memory-map \
                --seed "$seed"
        else
            echo "SKIP embed -- $embeddings/embeddings.npy exists"
        fi

        "$PYTHON" -u scripts/train_downstream.py \
            --mode probe \
            --embeddings "$embeddings/embeddings.npy" \
            --splits "$SPLITS_PATH" \
            --data-root "$DATA_ROOT" \
            --name "probe_beta${beta}_seed${seed}" \
            --seed "$seed"
    done
}

# ---------------------------------------------------------------------------
# Stage 8: the downstream CNN
# ---------------------------------------------------------------------------

stage_cnn() {
    require_file "$SELECTION_PATH" "run the lock first: bash scripts/run_all.sh lock"

    local beta k embeddings vae
    beta="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_beta"])' \
        "$SELECTION_PATH")"
    k="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_k"])' \
        "$SELECTION_PATH")"

    embeddings="$(embed_dir "$beta" "$SEED")/embeddings.npy"
    vae="$(vae_dir "$beta" "$SEED")/checkpoints/final.pt"
    require_file "$embeddings" "run: bash scripts/run_all.sh embed"

    banner "cnn -- classifier on propagated pseudo-labels, beta=$beta K=$k folds=$CNN_FOLDS" \
           "~2.5 h per fold"

    # The pseudo-labels are propagated inside this call, at the LOCKED K, from
    # each fold's 800 development-train seeds -- never the full 1000, since the
    # other 200 are what this run is scored on. Nodes label propagation could
    # not reach carry the -1 sentinel and are dropped before the first epoch:
    # their "label" is the argmax of an all-zero row, so training on them is
    # training on noise.
    #
    # --confidence-weighting weights the loss by the winning class's share of
    # each node's mass. A node two hops from a single seed is a much weaker
    # supervision signal than a seed itself. Drop the flag for the unweighted
    # control; it is worth having both in the report.
    "$PYTHON" -u scripts/train_downstream.py \
        --mode cnn \
        --embeddings "$embeddings" \
        --k "$k" \
        --encoder-checkpoint "$vae" \
        --splits "$SPLITS_PATH" \
        --normalization "$NORMALIZATION_PATH" \
        --data-root "$DATA_ROOT" \
        --folds "$CNN_FOLDS" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --load-images-to-ram \
        --confidence-weighting \
        --log-every 5 \
        --name "cnn_beta${beta}_k${k}_seed${SEED}" \
        --seed "$SEED"

    echo ""
    echo "The checkpoints/ directory of that run is what stage \`final\` needs for"
    echo "the CNN row. Note the path."
}

# ---------------------------------------------------------------------------
# Stage 9: the final benchmark -- THE TEST SPLIT
# ---------------------------------------------------------------------------

stage_final() {
    banner "final -- TEST SPLIT, both rows, once" "~10 min"

    require_file "$SELECTION_PATH" "run the lock first: bash scripts/run_all.sh lock"

    local beta embeddings vae
    beta="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_beta"])' \
        "$SELECTION_PATH")"
    embeddings="$(embed_dir "$beta" "$SEED")/embeddings.npy"
    vae="$(vae_dir "$beta" "$SEED")/checkpoints/final.pt"

    require_file "$embeddings" "run: bash scripts/run_all.sh embed"
    require_file "$vae" "the VAE checkpoint the cached embeddings came from"

    # CNN_CHECKPOINTS must point at the checkpoints/ directory of the stage
    # `cnn` run. Without it only the probe row is produced, and the capacity
    # confound between a CNN on ~100k pseudo-labels and a linear probe on 1000
    # clean ones becomes invisible to the reader. Both rows or neither.
    if [ -z "${CNN_CHECKPOINTS:-}" ]; then
        echo "CNN_CHECKPOINTS is not set. The report needs BOTH rows -- set it to the"
        echo "checkpoints/ directory of the stage \`cnn\` run, e.g."
        echo "  CNN_CHECKPOINTS=$RUNS/cnn_beta${beta}_k15_seed${SEED}_20260804_120000/checkpoints \\"
        echo "      bash scripts/run_all.sh final"
        echo ""
        echo "Continuing with the probe row only in 10 seconds; Ctrl-C to stop."
        sleep 10
        "$PYTHON" -u scripts/final_benchmark.py \
            --confirm-test-evaluation \
            --selection "$SELECTION_PATH" \
            --embeddings "$embeddings" \
            --vae-checkpoint "$vae" \
            --splits "$SPLITS_PATH" \
            --normalization "$NORMALIZATION_PATH" \
            --data-root "$DATA_ROOT" \
            --seed "$SEED"
    else
        "$PYTHON" -u scripts/final_benchmark.py \
            --confirm-test-evaluation \
            --selection "$SELECTION_PATH" \
            --embeddings "$embeddings" \
            --vae-checkpoint "$vae" \
            --cnn-checkpoints "$CNN_CHECKPOINTS" \
            --splits "$SPLITS_PATH" \
            --normalization "$NORMALIZATION_PATH" \
            --data-root "$DATA_ROOT" \
            --seed "$SEED"
    fi

    echo ""
    echo "Run /parity-check before any of these numbers goes into the report."
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

stage_all() {
    # Everything that does not need a human decision. Stops at the lock.
    stage_prepare
    stage_pretrain
    stage_embed
    stage_sweep
    stage_probe

    cat <<'DONE'

==============================================================================
== `all` is done. It STOPS HERE on purpose.
==
== Read the sweep tables, choose beta / K / C on the DEVELOPMENT numbers, then:
==   SELECTED_BETA=... SELECTED_K=... SELECTED_C=... bash scripts/run_all.sh lock
== followed by  seeds, cnn, and finally (once)  final.
==============================================================================
DONE
}

main() {
    local stage="${1:-plan}"

    echo "repo     $REPO_ROOT"
    echo "python   $PYTHON"
    echo "data     $DATA_ROOT"
    echo "seed     $SEED"
    echo "stage    $stage"

    if [ "$stage" != "plan" ] && [ ! -x "$PYTHON" ]; then
        echo "" >&2
        echo "No interpreter at $PYTHON." >&2
        echo "The server has no bare \`python\` -- create the venv, or point PYTHON at" >&2
        echo "the right python3:   PYTHON=/usr/bin/python3 bash scripts/run_all.sh $stage" >&2
        exit 1
    fi

    case "$stage" in
        plan)     stage_plan ;;
        prepare)  stage_prepare ;;
        pretrain) stage_pretrain ;;
        embed)    stage_embed ;;
        sweep)    stage_sweep ;;
        probe)    stage_probe ;;
        lock)     stage_lock ;;
        seeds)    stage_seeds ;;
        cnn)      stage_cnn ;;
        final)    stage_final ;;
        all)      stage_all ;;
        *)
            echo "unknown stage: $stage" >&2
            stage_plan
            exit 1
            ;;
    esac

    echo ""
    echo "stage '$stage' finished $(date '+%Y-%m-%d %H:%M:%S')"
}

main "$@"
