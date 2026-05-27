#!/usr/bin/env bash

# ============================
#  Real-data batch entry point.
#  Launches realdata_train_bunch.py for drug/activation jobs.
#  Seed derivation: job_seed = BASE_SEED + JOB_INDEX.
# ============================

MAX_JOBS=35    # <<< set your maximum parallel jobs here
BASE_SEED=20260527
SAVE_ROOT="${SAVE_ROOT:-4idata_results_seed${BASE_SEED}}"

DRUGS=(
  crizotinib
)

ACTS=("softplus_scaled_4")

# ============================
#  Create log directory
# ============================

mkdir -p logs

# ============================
#  Function to limit jobs
# ============================
wait_for_free_slot() {
    # Count active background jobs
    while [ "$(jobs -r | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 1
    done
}

# ============================
#  Launch jobs
# ============================

JOB_INDEX=0
for drug in "${DRUGS[@]}"; do
  for act in "${ACTS[@]}"; do

    SEED=$((BASE_SEED + JOB_INDEX))
    LOGFILE="logs/${drug}_${act}_seed${SEED}.log"
    echo "Launching: python realdata_train_bunch.py --drug $drug --act $act --seed $SEED"
    echo "  Log: $LOGFILE"

    # Wait until a slot is free
    wait_for_free_slot

    # Launch job
    python realdata_train_bunch.py \
        --drug "$drug" \
        --act "$act" \
        --seed "$SEED" \
        --run-index "$JOB_INDEX" \
        --save-root "$SAVE_ROOT" \
        > "$LOGFILE" 2>&1 &

    PID=$!
    echo "  PID: $PID"
    echo
    JOB_INDEX=$((JOB_INDEX + 1))

  done
done

echo "All jobs launched. Monitor with: tail -f logs/<drug>_<act>_seed<seed>.log"
