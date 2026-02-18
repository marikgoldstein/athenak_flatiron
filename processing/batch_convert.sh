#!/bin/bash
# Batch convert seeds to npy, 10 at a time
# Usage: bash batch_convert.sh <base_dir> <start_seed> <end_seed> <snap_start> <snap_end> <label>

set -e
module load python/3.11.11

BASE_DIR="$1"
START_SEED="$2"
END_SEED="$3"
SNAP_START="$4"
SNAP_END="$5"
LABEL="$6"
BATCH_SIZE=10
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Batch convert: $LABEL ==="
echo "Base dir: $BASE_DIR"
echo "Seeds: $START_SEED to $END_SEED"
echo "Snaps: $SNAP_START to $SNAP_END"
echo "Batch size: $BATCH_SIZE"
echo ""

completed=0
failed=0
total=$(( END_SEED - START_SEED + 1 ))

for (( batch_start=START_SEED; batch_start<=END_SEED; batch_start+=BATCH_SIZE )); do
    batch_end=$(( batch_start + BATCH_SIZE - 1 ))
    if [ $batch_end -gt $END_SEED ]; then
        batch_end=$END_SEED
    fi

    echo "[$(date '+%H:%M:%S')] Starting batch: seeds $batch_start-$batch_end"
    pids=()

    for (( s=batch_start; s<=batch_end; s++ )); do
        seed_dir=$(printf "%s/seed_%04d" "$BASE_DIR" "$s")
        if [ ! -d "$seed_dir/bin" ]; then
            echo "  WARNING: $seed_dir/bin not found, skipping"
            failed=$((failed + 1))
            continue
        fi

        python3 "$SCRIPT_DIR/convert_to_npy.py" \
            --bindir "$seed_dir/bin" \
            --outdir "$seed_dir/npy" \
            --snap-start "$SNAP_START" --snap-end "$SNAP_END" \
            > "$seed_dir/convert.log" 2>&1 &
        pids+=($!)
    done

    # Wait for this batch
    for pid in "${pids[@]}"; do
        if wait $pid; then
            completed=$((completed + 1))
        else
            failed=$((failed + 1))
        fi
    done

    echo "[$(date '+%H:%M:%S')] Batch done. Progress: $completed/$total completed, $failed failed"
done

echo ""
echo "=== DONE: $LABEL ==="
echo "Completed: $completed/$total"
echo "Failed: $failed"
