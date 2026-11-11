"""
Stream FineWeb (HuggingFace) and tokenize with the GLM-5.3 tokenizer into a flat HDF5.

FineWeb is high-quality web text. Streaming avoids downloading multi-GB shards: documents
are pulled one at a time, tokenized, and appended to the HDF5 until --max_tokens is hit.

Examples:
    # 3B tokens (~12 GB HDF5) for a small base model
    python scripts/prepare_fineweb_data.py --out data/fineweb_train.h5 --max_tokens 3000000000
    # dev split from the same stream (skip the first N tokens, take a slice)
    python scripts/prepare_fineweb_data.py --out data/fineweb_dev.h5 --max_tokens 10000000 --skip_tokens 3000000000
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from datasets import load_dataset

from src.tokenizer.hf_tokenizer import HFTokenizer

WRITE_CHUNK = 8_000_000
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_CONFIG = "sample-10BT"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output HDF5 path")
    p.add_argument("--max_tokens", type=int, default=3_000_000_000)
    p.add_argument("--skip_tokens", type=int, default=0, help="skip this many tokens before writing")
    p.add_argument("--min_chars", type=int, default=200, help="drop documents shorter than this")
    args = p.parse_args()

    tok = HFTokenizer("data/tokenizers/tokenizer.json")
    eot = tok.eot_token
    print(f"tokenizer: n_vocab={tok.n_vocab} eot={eot}")

    ds = load_dataset(DATASET_NAME, name=DATASET_CONFIG, split="train", streaming=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    total = 0
    skipped = 0
    buf: list[int] = []
    with h5py.File(args.out, "w") as f:
        dset = f.create_dataset("tokens", (0,), maxshape=(None,), dtype="i4", chunks=(WRITE_CHUNK,))
        # Same metadata prepare_cyber_data.py writes: the trainer reads the tokenizer spec
        # from here (without it, sampling falls back to r50k_base).
        import json as _json

        dset.attrs["tokenizer"] = _json.dumps({
            "type": "hf", "version": 1,
            "tokenizer_path": "data/tokenizers/tokenizer.json",
            "eot_token": "<|endoftext|>", "n_vocab": tok.n_vocab,
        })
        dset.attrs["vocab_size"] = tok.n_vocab

        def flush() -> None:
            nonlocal total, buf
            if not buf:
                return
            arr = np.asarray(buf, dtype=np.int32)
            dset.resize(total + arr.size, axis=0)
            dset[total:total + arr.size] = arr
            total += arr.size
            buf = []

        for row in ds:
            text = (row.get("text") or "").strip()
            if len(text) < args.min_chars:
                continue
            ids = tok.encode_ordinary(text)
            if skipped < args.skip_tokens:
                skipped += len(ids) + 1
                continue
            buf.extend(ids)
            buf.append(eot)
            if len(buf) >= WRITE_CHUNK:
                flush()
                print(f"  {total / 1e6:.1f}M tokens written", flush=True)
            if total >= args.max_tokens:
                break
        flush()

    print(f"wrote {total:,} tokens -> {args.out}")


if __name__ == "__main__":
    main()
