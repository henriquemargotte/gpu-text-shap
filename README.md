# Documentação não atualizada. Código atual no diretório embedding.

# linear_mlp_bert_cuda

End-to-end demo:

1) Python: downloads a public text dataset, tokenizes with a BERT tokenizer, trains an MLP (linear layers with identity activations), and exports:
   - tokenized dataset to a plain-text file
   - network weights to a plain-text file

2) CUDA: loads the exported dataset + weights and runs a GPU feedforward kernel that also computes permutation-based SHAP-style feature contributions for a single sample.

## Quickstart

```bash
cd linear_mlp_bert_cuda
./scripts/run_all.sh \
  --dataset imdb \
  --max-len 64 \
  --hidden-dim 128 \
  --num-layers 3 \
  --epochs 1 \
  --train-samples 2000 \
  --test-samples 256
```

Artifacts are written under `out/`.

## What changed / important notes

- The pipeline now defaults to a binary dataset (`imdb`) so the exported model is a binary classifier (single scalar logit per input).
- `./scripts/run_all.sh` exports the dataset then builds the CUDA binary and creates a one-sample dataset file containing the first exported sample. The CUDA binary is run on that single sample.
- The CUDA kernel runs `2 * n_permutations` blocks: the first `n_permutations` blocks compute the model using a random permutation of the input token order, and the next `n_permutations` blocks compute the same permutation in reverse order. This lets the kernel approximate permutation-based SHAP contributions.
- Per-block work:
  - Each block builds its permutation (Fisher–Yates with a small LCG) in shared memory and loads the permuted inputs.
  - The kernel computes cumulative-model outputs for prefixes of features (for i = 0..seq_len-1) by running the MLP repeatedly but only including inputs up to `i` in the first layer; the per-prefix outputs are stored in a per-block shared buffer.
  - The block then differences those cumulative outputs to recover per-feature marginal contributions and atomically accumulates them into a global `shap` buffer.
- Host-side: after the kernel finishes the host copies the aggregated `shap` buffer, divides by the number of blocks to produce averaged `shap_values` (one value per input feature), and writes a CSV file next to the single-sample dataset containing: `feature_idx,token_id,shap_value` (`out/tokenized_dataset.txt.single.shap_values.csv`).

## Implementation notes

- Feature transform: `x = token_id / vocab_size` (same as before).
- The MLP still uses identity activation between layers.
- Shared memory footprint now includes two float buffers for layer inputs/outputs, a permutation `int` buffer of length `seq_len`, and a per-block float buffer of length `seq_len` used to accumulate per-feature prefix outputs.
- If `seq_len` grows large the shared-memory usage per block may limit occupancy; consider reducing `n_permutations` or using a reduction-based approach if you encounter shared-memory limits.

## Output

- Model weights: `out/mlp_weights.txt`
- Tokenized dataset (single sample for CUDA run): `out/tokenized_dataset.txt.single`
- SHAP CSV: `out/tokenized_dataset.txt.single.shap_values.csv` (columns: `feature_idx,token_id,shap_value`)

## Reproduce / debug

- Build only:
```bash
cd cuda && make -j
```
- Run CUDA feedforward on the already-exported single-sample file:
```bash
OUT_DIR=out ./out/feedforward --weights out/mlp_weights.txt --dataset out/tokenized_dataset.txt.single --vocab-size $(python3 -c "import json;print(json.load(open('out/export_meta.json'))['vocab_size'])") --threads 128 --print 10
```

If you'd like a different behavior (e.g., compute SHAP on multiple samples or precompute permutations on the host), let me know and I can update the scripts and kernel accordingly.
