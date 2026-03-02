#!/usr/bin/env python3
"""Compute SHAP values for a sample using the exported model and dataset.

This script loads the plain-text weights exported by `train_export.py`, builds
an equivalent PyTorch linear MLP, loads a sample from the exported dataset
file, and computes per-feature SHAP values using one of the supported
explainers: `linear`, `kernel`, or `permutation`.

Notes for fair comparison with the CUDA implementation:
- Inputs are normalized as `token_id / vocab_size` (same as CUDA and exporter).
- Missing features are masked by replacing with zeros (CUDA treats omitted
  features as absent/zero when computing marginal contributions).
- The script can truncate the sample to `--sample-size` to match the CUDA
  single-sample dataset produced by `scripts/run_all.sh`.

Example:
  python python/compute_shap.py \
    --weights out/mlp_weights.txt \
    --dataset out/tokenized_dataset.txt \
    --meta out/export_meta.json \
    --sample 0 --sample-size 64 --explainer permutation --nsamples 100
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple, Optional

import numpy as np
import torch
import time


def parse_args():
    p = argparse.ArgumentParser(description="Compute SHAP values for an exported MLP and a dataset sample")
    p.add_argument("--weights", type=str, required=True, help="Path to exported weights (text)")
    p.add_argument("--dataset", type=str, required=True, help="Exported dataset file (token ids)")
    p.add_argument("--meta", type=str, required=True, help="Exported meta JSON (contains tokenizer, vocab_size)")
    p.add_argument("--sample", type=int, default=0, help="Index of sample to explain (0-based)")
    p.add_argument("--sample-size", type=int, default=None, help="Truncate the input to this many features (must be <= exported seq_len)")
    p.add_argument("--explainer", type=str, default="permutation", choices=["linear", "kernel", "permutation"], help="Explainer algorithm to use")
    p.add_argument("--nsamples", type=int, default=1000, help="Number of samples (for Kernel/Permutation) or budget for explainer")
    p.add_argument("--target-index", type=int, default=0, help="Target output index/class to explain (default 0)")
    p.add_argument("--out", type=str, default=None, help="Output text path (defaults to <dataset>.sampleX.shap.txt)")
    p.add_argument("--tokenizer", type=str, default=None, help="Override tokenizer name in meta")
    p.add_argument("--npermutations", type=int, default=1000, help="Number of permutations for permutation explainer (if supported by shap version)")
    p.add_argument("--embeddings", type=str, default=None, help="Path to exported embedding matrix (text)")
    return p.parse_args()


def read_dataset(dataset_path: str) -> Tuple[int, int, List[List[int]]]:
    # Returns (n, seq_len, token_rows)
    with open(dataset_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        if len(header) < 2:
            raise RuntimeError("Invalid dataset header")
        n = int(header[0]); seq_len = int(header[1])
        rows = []
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # first entry is label
            ids = [int(x) for x in parts[1:1+seq_len]]
            rows.append(ids)
    return n, seq_len, rows


def load_weights_text(path: str):
    # Reads weights file and returns a list of (W,b) numpy arrays where W.shape=(out_dim,in_dim)
    layers = []
    with open(path, "r", encoding="utf-8") as f:
        L = int(f.readline().strip())
        for _ in range(L):
            line = f.readline()
            if not line:
                raise RuntimeError("Unexpected EOF in weights file")
            in_dim, out_dim = [int(x) for x in line.strip().split()]
            w_count = out_dim * in_dim
            w_vals = []
            while len(w_vals) < w_count:
                parts = f.readline().strip().split()
                w_vals.extend([float(x) for x in parts])
            W = np.array(w_vals, dtype=np.float32).reshape((out_dim, in_dim))
            # read biases (may be on one line)
            b_vals = []
            while len(b_vals) < out_dim:
                parts = f.readline().strip().split()
                b_vals.extend([float(x) for x in parts])
            b = np.array(b_vals, dtype=np.float32)
            layers.append((W, b))
    return layers


def build_torch_model_from_weights(layers: List[Tuple[np.ndarray, np.ndarray]], input_cut: Optional[int] = None) -> torch.nn.Module:
    # If input_cut is provided and smaller than first layer in_dim, slice the first W accordingly.
    modules = []
    for i, (W, b) in enumerate(layers):
        out_dim, in_dim = W.shape
        if i == 0 and input_cut is not None and input_cut < in_dim:
            in_dim_use = input_cut
            W_use = W[:, :in_dim_use]
        else:
            in_dim_use = in_dim
            W_use = W

        lin = torch.nn.Linear(in_dim_use, out_dim, bias=True)
        lin.weight.data = torch.from_numpy(W_use).to(torch.float32)
        lin.bias.data = torch.from_numpy(b).to(torch.float32)
        modules.append(lin)
    # Build a sequential model applying linears in order with no activations
    model = torch.nn.Sequential(*modules)
    model.eval()
    return model


def compute_effective_input_weights(layers: List[Tuple[np.ndarray, np.ndarray]], input_cut: Optional[int] = None) -> np.ndarray:
    # Compute W_eff = W_L @ W_{L-1} @ ... @ W_1  (shape: out_dim_last, in_dim_first)
    mats = [W for (W, b) in layers]
    # Possibly slice first matrix columns
    if input_cut is not None and input_cut < mats[0].shape[1]:
        mats[0] = mats[0][:, :input_cut]
    W_eff = mats[0]
    for M in mats[1:]:
        W_eff = M @ W_eff
    return W_eff


def main():
    args = parse_args()

    # Load meta
    with open(args.meta, "r", encoding="utf-8") as mf:
        meta = json.load(mf)
    tokenizer_name = args.tokenizer if args.tokenizer else meta.get("tokenizer")
    # tokenizer_name is optional; detokenization will use the exported `vocab.txt`
    vocab_size = int(meta.get("vocab_size", 30522))

    # Resolve embedding matrix path
    emb_path = args.embeddings
    if emb_path is None:
        emb_rel = meta.get("export", {}).get("embedding_file")
        if emb_rel:
            emb_path = os.path.join(os.path.dirname(args.dataset), emb_rel)
    if emb_path is None:
        raise RuntimeError("Embedding matrix path not provided and not found in meta (use --embeddings)")

    # Load embedding matrix (text format: vocab_size embed_dim followed by floats)
    def load_embedding_text(path: str):
        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split()
            if len(header) < 2:
                raise RuntimeError("Invalid embedding header")
            vs = int(header[0]); ed = int(header[1])
            vals = []
            while len(vals) < vs * ed:
                parts = f.readline().strip().split()
                vals.extend([float(x) for x in parts])
        arr = np.array(vals, dtype=np.float32).reshape((vs, ed))
        return arr

    emb_matrix = load_embedding_text(emb_path)

    # Read dataset and sample
    n, seq_len, rows = read_dataset(args.dataset)
    if args.sample < 0 or args.sample >= n:
        raise IndexError("sample index out of range")

    token_ids = rows[args.sample]
    # Determine sample size (truncate to requested)
    if args.sample_size is None:
        sample_size = seq_len
    else:
        if args.sample_size > seq_len:
            raise ValueError("--sample-size cannot be larger than dataset seq_len")
        sample_size = args.sample_size

    token_ids = token_ids[:sample_size]

    # Load MLP weights and construct Embedding+MLP model matching train_export_emb
    layers = load_weights_text(args.weights)

    import torch.nn as nn

    class EmbeddingMLP(nn.Module):
        def __init__(self, emb_matrix: np.ndarray, layers_weights: List[Tuple[np.ndarray, np.ndarray]], padding_idx: int = 0, dropout: float = 0.0):
            super().__init__()
            vs, ed = emb_matrix.shape
            self.embedding = nn.Embedding(vs, ed, padding_idx=padding_idx)
            # load embedding weights
            with torch.no_grad():
                self.embedding.weight.data = torch.from_numpy(emb_matrix).to(torch.float32)
            modules = []
            for i, (W, b) in enumerate(layers_weights):
                out_dim, in_dim = W.shape
                lin = nn.Linear(in_dim, out_dim, bias=True)
                lin.weight.data = torch.from_numpy(W).to(torch.float32)
                lin.bias.data = torch.from_numpy(b).to(torch.float32)
                modules.append(lin)
            self.layers = nn.ModuleList(modules)
            self.dropout = nn.Identity()
            self.padding_idx = padding_idx
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len) token ids
            emb = self.embedding(x)
            mask = (x != self.padding_idx).unsqueeze(-1).to(emb.dtype)
            summed = (emb * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            out = summed / lengths
            for i, layer in enumerate(self.layers):
                out = layer(out)
                if i < (len(self.layers) - 1):
                    out = self.relu(out)
            return out

    model = EmbeddingMLP(emb_matrix, layers, padding_idx=0, dropout=0.0)
    model.eval()

    # Prepare input token ids (integers). shape (1, sample_size)
    x = np.array(token_ids, dtype=np.int64).reshape(1, -1)

    # Target index
    target = args.target_index

    # Output path
    out_path = args.out if args.out else os.path.splitext(args.dataset)[0] + f".sample{args.sample}.compute_shap.txt"

    # For fair comparison, use zero baseline / masker (missing features -> 0 token id)
    background = np.zeros((1, x.shape[1]), dtype=np.float32)

    # All explainers require the shap library
    try:
        import shap
    except Exception as e:
        raise RuntimeError("shap library is required for explainers. Install via 'pip install shap'") from e

    # If explainer is linear, use shap's LinearExplainer on the effective linear model
    if args.explainer == "linear":
        # Compute effective weight matrix
        W_eff = compute_effective_input_weights(layers, input_cut=sample_size)
        # W_eff shape (out_dim_last, input_dim)
        if target >= W_eff.shape[0]:
            raise IndexError("target-index out of range for model outputs")

        # Compute effective bias by running the model on zero input
        zero_in = torch.zeros((1, x.shape[1]), dtype=torch.float32)
        with torch.no_grad():
            b_eff = model(zero_in).cpu().numpy().reshape(-1)

        # Simple linear model wrapper providing coef_ and intercept_ and predict
        class _SimpleLinearModel:
            def __init__(self, W, b):
                self.coef_ = W
                self.intercept_ = b
            def predict(self, X: np.ndarray) -> np.ndarray:
                return X.dot(self.coef_.T) + self.intercept_

        lm = _SimpleLinearModel(W_eff.astype(np.float32), b_eff.astype(np.float32))
        expl = shap.LinearExplainer(lm, background)
        t0 = time.perf_counter()
        shap_out = expl.shap_values(x)
        t1 = time.perf_counter()
        print(f"linear_explainer_eval_time={t1-t0:.3f}s")
        # shap_out may be list (per output) or array
        if isinstance(shap_out, list):
            shap_vals = np.array(shap_out[target]).reshape(-1)[:x.shape[1]]
        else:
            shap_vals = np.array(shap_out).reshape(-1)[:x.shape[1]]

    else:
        # Define a prediction function returning scalar model output for target
        def f(X: np.ndarray) -> np.ndarray:
            # X shape: (nsamples, input_dim) may be floats; round/clip to valid token ids
            Xi = np.clip(np.rint(X), 0, x.max()).astype(np.int64)
            xt = torch.from_numpy(Xi)
            with torch.no_grad():
                out = model(xt).cpu().numpy()
            if out.ndim == 1:
                return out.reshape(-1)
            return out[:, target]

        if args.explainer == "kernel":
            expl = shap.KernelExplainer(f, background)
            # nsamples controls the number of evaluations (approx)
            shap_out = expl.shap_values(x, nsamples=args.nsamples)
            # shap_values may be array shape (1, input_dim) or list; unify
            shap_vals = np.array(shap_out).reshape(-1)[:x.shape[1]]

        elif args.explainer == "permutation":
            # Use permutation explainer with an independent masker (zero baseline)
            try:
                masker = shap.maskers.Independent(background)
                expl = shap.Explainer(f, masker, algorithm="permutation")
                # Use max_evals to control the budget; time the evaluation
                t0 = time.perf_counter()
                exp = expl(x, max_evals=args.npermutations)
                t1 = time.perf_counter()
                print(f"permutation_explainer_eval_time={t1-t0:.3f}s")
                shap_vals = np.array(exp.values).reshape(-1, x.shape[1])[0]
            except Exception:
                # Fallback: run shap.Explainer without masker kwargs
                expl = shap.Explainer(f, algorithm="permutation")
                t0 = time.perf_counter()
                exp = expl(x, max_evals=args.npermutations)
                t1 = time.perf_counter()
                print(f"permutation_explainer_eval_time={t1-t0:.3f}s")
                shap_vals = np.array(exp.values).reshape(-1, x.shape[1])[0]

        else:
            raise ValueError("Unsupported explainer")

    # Detokenize using the exported `vocab.txt` (matches `train_export_emb` tokenization)
    # Try to locate vocab file from meta export or next to the dataset
    vocab_path = None
    emb_export = meta.get("export", {})
    if emb_export:
        # try common name
        possible = emb_export.get("vocab_file")
        if possible:
            vocab_path = os.path.join(os.path.dirname(args.dataset), possible)
    if vocab_path is None:
        # fallback to dataset directory / vocab.txt
        vocab_path = os.path.join(os.path.dirname(args.dataset), "vocab.txt")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found for detokenization: {vocab_path}")

    # load vocab list: line index == token id
    vocab_list = []
    with open(vocab_path, "r", encoding="utf-8") as vf:
        for line in vf:
            vocab_list.append(line.rstrip("\n"))

    # Map token ids to strings
    tokens = [vocab_list[tid] if 0 <= tid < len(vocab_list) else "<unk>" for tid in token_ids]
    # Reconstruct a simple detokenized text by joining tokens with spaces
    text = " ".join(tokens).strip()

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"sample={args.sample} explainer={args.explainer} target={target} sample_size={sample_size}\n")
        f.write(f"detokenized_full_text: {text}\n")
        f.write("idx\ttoken_id\ttoken_str\tshap_value\n")
        for i, tid in enumerate(token_ids):
            tok_str = tokens[i]
            sv = float(shap_vals[i]) if i < len(shap_vals) else 0.0
            f.write(f"{i}\t{tid}\t{tok_str}\t{sv}\n")

    print(f"Wrote SHAP output to: {out_path}")


if __name__ == "__main__":
    main()
