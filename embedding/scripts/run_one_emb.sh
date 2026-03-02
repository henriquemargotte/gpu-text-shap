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

echo "Running emb_shap on exported dataset (this may take time). Output -> $OUT_DIR/shap_values.txt"
"$EMB_BIN" \
  --weights "$OUT_DIR/$WEIGHTS_FILE" \
  --dataset "$OUT_DIR/$DATASET_FILE" \
  --npermutations "$N_PERMUTATIONS" \
  --embeddings "$OUT_DIR/embedding_matrix.txt" \
  --threads "$THREADS" \
  --sample "$SAMPLE_ID" \
  --print "$PRINT" > "$OUT_DIR/shap_values.txt" 2>&1 || true

echo "SHAP output saved to: $OUT_DIR/shap_values.txt"

FINAL_ACC=$(grep -o 'final_test_acc=[0-9.]*' "$TRAIN_LOG" | tail -n1 | cut -d= -f2 || true)
if [[ -n "$FINAL_ACC" ]]; then
  echo "Final test accuracy: $FINAL_ACC"
else
  echo "Could not parse final test accuracy from train log: $TRAIN_LOG"
fi

echo "[4/4] Detokenize SHAP for sample $SAMPLE_ID -> $OUT_DIR/sample${SAMPLE_ID}_shap.txt"
python3 "$ROOT_DIR/embedding/python/detokenize_shap_emb.py" \
  --dataset "$OUT_DIR/$DATASET_FILE" \
  --vocab "$OUT_DIR/vocab.txt" \
  --shap-csv "$OUT_DIR/$DATASET_FILE.shap_values.csv" \
  --sample "$SAMPLE_ID" \
  --out "$OUT_DIR/sample${SAMPLE_ID}_shap.txt"
