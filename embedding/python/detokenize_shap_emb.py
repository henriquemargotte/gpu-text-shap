#!/usr/bin/env python3
"""Detokenize SHAP outputs using the embedding-word vocabulary exported by the embedding pipeline.

This script reads:
- exported dataset file (token ids): header `N seq_len` then `label id0 id1 ...`
- SHAP CSV produced by `emb_shap` (dataset_file.shap_values.csv)
- `vocab.txt` produced by `train_export_emb.py` where line number == token id

Outputs a text file listing tokens and their SHAP values.
"""

import argparse
import csv
import os
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--vocab", type=str, required=True, help="Path to vocab.txt (one token per line, index == line number)")
    p.add_argument("--shap-csv", type=str, default=None)
    p.add_argument("--sample", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def read_dataset_tokens(dataset_path: str, sample_idx: int) -> List[int]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        n = int(header[0])
        seq_len = int(header[1])
        if sample_idx < 0 or sample_idx >= n:
            raise IndexError("sample out of range")
        for i, line in enumerate(f):
            if i == sample_idx:
                parts = line.strip().split()
                ids = [int(x) for x in parts[1:1+seq_len]]
                return ids
    raise RuntimeError("sample not found")


def read_shap_values(shap_csv_path: str, seq_len: int) -> List[float]:
    vals = [0.0] * seq_len
    with open(shap_csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            idx = int(row.get("feature_idx") or row.get("feature") or row.get("idx"))
            val = float(row.get("shap_value") or row.get("shap") or row.get("value"))
            if 0 <= idx < seq_len:
                vals[idx] = val
    return vals


def load_vocab_list(path: str) -> List[str]:
    toks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            toks.append(line.rstrip("\n"))
    return toks


def main():
    args = parse_args()
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not os.path.exists(args.vocab):
        raise FileNotFoundError(f"Vocab not found: {args.vocab}")

    with open(args.dataset, "r", encoding="utf-8") as f:
        header = f.readline().strip().split()
        seq_len = int(header[1])

    shap_csv = args.shap_csv if args.shap_csv else args.dataset + ".shap_values.csv"
    if not os.path.exists(shap_csv):
        raise FileNotFoundError(f"SHAP CSV not found: {shap_csv}")

    token_ids = read_dataset_tokens(args.dataset, args.sample)
    shap_vals = read_shap_values(shap_csv, seq_len)
    vocab = load_vocab_list(args.vocab)

    # Build rows (original_index, token_id, token_str, shap_value)
    rows = []
    for i, tid in enumerate(token_ids):
        tok = vocab[tid] if 0 <= tid < len(vocab) else "<unk>"
        val = shap_vals[i] if i < len(shap_vals) else 0.0
        rows.append((i, tid, tok, val))

    # Sort by shap value descending (largest first)
    rows.sort(key=lambda r: r[3], reverse=True)

    out_path = args.out if args.out else os.path.splitext(shap_csv)[0] + f".sample{args.sample}.txt"
    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write(f"sample={args.sample}\n")
        outf.write("orig_idx\ttoken_id\ttoken_str\tshap_value\n")
        for orig_idx, tid, tok, val in rows:
            outf.write(f"{orig_idx}\t{tid}\t{tok}\t{val}\n")

    print(f"Wrote embedding-aware detokenized SHAP to: {out_path}")


if __name__ == "__main__":
    main()
