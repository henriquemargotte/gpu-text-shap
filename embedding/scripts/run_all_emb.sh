#!/usr/bin/env bash
set -euo pipefail

# Orchestrates:
# 1) Python train + export
# 2) Build CUDA binary
# 3) Run CUDA emb shap
# 5) Detokenize SHAP for sample 0


SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT_DIR_DEFAULT="$ROOT_DIR/out"

DATASET="imdb"
MAX_LEN=128
HIDDEN_DIM=64
NUM_LAYERS=4
BATCH_SIZE=64
EPOCHS=20
LR=1e-3
TRAIN_SAMPLES=2000
TEST_SAMPLES=256
DEVICE="auto"

THREADS=128
PRINT=10

N_SAMPLES=1
SAMPLE_ID=0
N_PERMUTATIONS=129

OUT_DIR="$OUT_DIR_DEFAULT"
WEIGHTS_FILE="mlp_weights.txt"
DATASET_FILE="tokenized_dataset.txt"
META_FILE="export_meta.json"

usage() {
  cat <<EOF
Usage: $0 [options]

Python/training options:
  --dataset NAME            (default: $DATASET)
  --max-len N               (default: $MAX_LEN)
  --hidden-dim N            (default: $HIDDEN_DIM)
  --num-layers N            (default: $NUM_LAYERS)
  --batch-size N            (default: $BATCH_SIZE)
  --epochs N                (default: $EPOCHS)
  --lr FLOAT                (default: $LR)
  --train-samples N         (default: $TRAIN_SAMPLES)
  --test-samples N          (default: $TEST_SAMPLES)
  --device auto|cpu|cuda    (default: $DEVICE)

Export/output options:
  --out-dir PATH            (default: $OUT_DIR)
  --weights-file NAME       (default: $WEIGHTS_FILE)
  --dataset-file NAME       (default: $DATASET_FILE)
  --meta-file NAME          (default: $META_FILE)

CUDA options:
  --threads N               (default: $THREADS)
  --print N                 (default: $PRINT)

SHAP options:
  --sample N                (default: $SAMPLE_ID)
  --nsamples N              (default: $N_SAMPLES)
  --npermutations N         (default: $N_PERMUTATIONS)
Other:
  --skip-build              skip nvcc build
  -h|--help

Example:
  $0 --dataset imdb --max-len 64 --hidden-dim 128 --num-layers 3 --epochs 3
EOF
}

SKIP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --max-len) MAX_LEN="$2"; shift 2;;
    --hidden-dim) HIDDEN_DIM="$2"; shift 2;;
    --num-layers) NUM_LAYERS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --lr) LR="$2"; shift 2;;
    --train-samples) TRAIN_SAMPLES="$2"; shift 2;;
    --test-samples) TEST_SAMPLES="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;

    --out-dir) OUT_DIR="$2"; shift 2;;
    --weights-file) WEIGHTS_FILE="$2"; shift 2;;
    --dataset-file) DATASET_FILE="$2"; shift 2;;
    --meta-file) META_FILE="$2"; shift 2;;

    --sample) SAMPLE_ID="$2"; shift 2;;
    --nsamples) N_SAMPLES="$2"; shift 2;;
    --npermutations) N_PERMUTATIONS="$2"; shift 2;;
    --threads) THREADS="$2"; shift 2;;
    --print) PRINT="$2"; shift 2;;

    --skip-build) SKIP_BUILD=1; shift 1;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

mkdir -p "$OUT_DIR"

echo "[1/4] Python train+export -> $OUT_DIR"
TRAIN_LOG="$OUT_DIR/train_export.log"
python3 "$ROOT_DIR/embedding/python/train_export_emb.py" \
  --dataset "$DATASET" \
  --max-len "$MAX_LEN" \
  --hidden-dim "$HIDDEN_DIM" \
  --num-layers "$NUM_LAYERS" \
  --dropout 0.5 \
  --weight-decay 0.0 \
  --batch-size "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --train-samples "$TRAIN_SAMPLES" \
  --test-samples "$TEST_SAMPLES" \
  --device "$DEVICE" \
  --out-dir "$OUT_DIR" \
  --weights-file "$WEIGHTS_FILE" \
  --dataset-file "$DATASET_FILE" \
  --meta-file "$META_FILE" 2>&1 | tee "$TRAIN_LOG"

if [[ "$SKIP_BUILD" -eq 0 ]]; then
  echo "[2/4] Build emb_shap CUDA binary"
  nvcc -O3 -arch=sm_70 -o "$OUT_DIR/emb_shap" "$ROOT_DIR/embedding/cuda/emb_shap.cu"
else
  echo "[2/4] Skipping CUDA build"
fi

echo "[3/4] CUDA emb_shap"
EMB_BIN="$OUT_DIR/emb_shap"
if [[ ! -f "$EMB_BIN" ]]; then
  echo "emb_shap binary not found at $EMB_BIN" >&2
  exit 2
fi

# Read number of samples from exported dataset header
DATASET_HEADER="$OUT_DIR/$DATASET_FILE"
if [[ ! -f "$DATASET_HEADER" ]]; then
  echo "Dataset file not found: $DATASET_HEADER" >&2
  exit 2
fi
N_DATASET=$(head -n1 "$DATASET_HEADER" | awk '{print $1}')
echo "Dataset contains $N_DATASET samples; iterating over all samples"

for SAMPLE_ID in $(seq 0 $((N_DATASET - 1))); do
  echo "Running emb_shap for sample $SAMPLE_ID -> $OUT_DIR/shap_values_sample${SAMPLE_ID}.txt"
  "$EMB_BIN" \
    --weights "$OUT_DIR/$WEIGHTS_FILE" \
    --dataset "$OUT_DIR/$DATASET_FILE" \
    --n-permutations "$N_PERMUTATIONS" \
    --embeddings "$OUT_DIR/embedding_matrix.txt" \
    --threads "$THREADS" \
    --sample "$SAMPLE_ID" \
    --print "$PRINT" > "$OUT_DIR/shap_values_sample${SAMPLE_ID}.txt" 2>&1 || true

  # Move per-sample SHAP CSV produced by emb_shap to a sample-specific filename
  SRC_CSV="$OUT_DIR/$DATASET_FILE.shap_values.csv"
  DST_CSV="$OUT_DIR/${DATASET_FILE}.sample${SAMPLE_ID}.shap_values.csv"
  if [[ -f "$SRC_CSV" ]]; then
    mv "$SRC_CSV" "$DST_CSV"
    echo "moved SHAP CSV to $DST_CSV"
  else
    echo "Warning: expected SHAP CSV not found at $SRC_CSV" >&2
  fi

  FINAL_ACC=$(grep -o 'final_test_acc=[0-9.]*' "$TRAIN_LOG" | tail -n1 | cut -d= -f2 || true)
  if [[ -n "$FINAL_ACC" ]]; then
    echo "Final test accuracy: $FINAL_ACC"
  else
    echo "Could not parse final test accuracy from train log: $TRAIN_LOG"
  fi

  # Run Python SHAP computation (permutation + linear) using the exported files
  PERM_OUT="$OUT_DIR/${DATASET_FILE}.sample${SAMPLE_ID}.compute_shap.permutation.txt"
  LINEAR_OUT="$OUT_DIR/${DATASET_FILE}.sample${SAMPLE_ID}.compute_shap.linear.txt"
  echo "[3.1] Compute SHAP (permutation) -> $PERM_OUT"
  python3 "$ROOT_DIR/embedding/python/compute_shap_emb.py" \
    --weights "$OUT_DIR/$WEIGHTS_FILE" \
    --dataset "$OUT_DIR/$DATASET_FILE" \
    --meta "$OUT_DIR/$META_FILE" \
    --embeddings "$OUT_DIR/embedding_matrix.txt" \
    --sample "$SAMPLE_ID" \
    --explainer permutation \
    --npermutations "$N_PERMUTATIONS" \
    --out "$PERM_OUT" 2>&1 | tee "$OUT_DIR/compute_shap_permutation.sample${SAMPLE_ID}.log" || true

  # Detokenize SHAP for this sample -> per-sample detokenized output
  echo "[4] Detokenize SHAP for sample $SAMPLE_ID -> $OUT_DIR/sample${SAMPLE_ID}_shap.txt"
  # use the per-sample CSV produced/moved above
  SAMPLE_CSV="$OUT_DIR/${DATASET_FILE}.sample${SAMPLE_ID}.shap_values.csv"
  python3 "$ROOT_DIR/embedding/python/detokenize_shap_emb.py" \
    --dataset "$OUT_DIR/$DATASET_FILE" \
    --vocab "$OUT_DIR/vocab.txt" \
    --shap-csv "$SAMPLE_CSV" \
    --sample "$SAMPLE_ID" \
    --out "$OUT_DIR/sample${SAMPLE_ID}_shap.txt" 2>&1 | tee "$OUT_DIR/detokenize_shap.sample${SAMPLE_ID}.log" || true

  # Also make the compute_shap_emb outputs easy to find (they already include detokenized tokens)
  if [[ -f "$PERM_OUT" ]]; then
    cp "$PERM_OUT" "$OUT_DIR/sample${SAMPLE_ID}_shap.permutation.txt"
  fi

  # Collect SHAP timing information (CUDA and Python permutation) and save one CSV row per sample
  TIMES_FILE="$OUT_DIR/shap_times.csv"
  if [[ ! -f "$TIMES_FILE" ]]; then
    echo "sample,cuda_time,permutation_time" > "$TIMES_FILE"
  fi

  # CUDA kernel time (ms -> s) from per-sample shap_values log
  cuda_ms=$(grep -o 'cuda_kernel_time_ms=[0-9.]*' "$OUT_DIR/shap_values_sample${SAMPLE_ID}.txt" | tail -n1 | cut -d= -f2 || true)
  if [[ -n "$cuda_ms" ]]; then
    cuda_s=$(awk "BEGIN{printf \"%.6f\", $cuda_ms/1000}")
  else
    cuda_s=""
  fi

  # Python permutation explainer time (seconds)
  perm_s=$(grep -o 'permutation_explainer_eval_time=[0-9.]*s' "$OUT_DIR/compute_shap_permutation.sample${SAMPLE_ID}.log" 2>/dev/null | tail -n1 | sed 's/.*=//' | sed 's/s$//' || true)
  if [[ -z "$perm_s" ]]; then
    perm_s=""
  fi

  echo "${SAMPLE_ID},${cuda_s},${perm_s}" >> "$TIMES_FILE"
  echo "Saved SHAP timing(s) for sample ${SAMPLE_ID} to: $TIMES_FILE"

  # Generate detailed summaries (top/bottom tokens) for CUDA and permutation explainers
  GEN_SCRIPT="$ROOT_DIR/embedding/scripts/generate_shap_summaries.py"
  if [[ -f "$GEN_SCRIPT" ]]; then
    echo "Generating detailed SHAP summaries for sample ${SAMPLE_ID} (CUDA + permutation)"
    python3 "$GEN_SCRIPT" \
      --out-dir "$OUT_DIR" \
      --sample "$SAMPLE_ID" \
      --vocab "$OUT_DIR/vocab.txt" \
      --cuda-shap-csv "$DST_CSV" \
      --cuda-log "$OUT_DIR/shap_values_sample${SAMPLE_ID}.txt" \
      --perm-detok "$OUT_DIR/sample${SAMPLE_ID}_shap.permutation.txt" \
      --perm-log "$OUT_DIR/compute_shap_permutation.sample${SAMPLE_ID}.log" \
      --logits-file "$OUT_DIR/test_logits_sorted.csv" || true
    echo "Wrote summaries: $OUT_DIR/cuda_shap_summary.csv and $OUT_DIR/permutation_shap_summary.csv (appended)"
  else
    echo "Summary generator not found: $GEN_SCRIPT" >&2
  fi

done
