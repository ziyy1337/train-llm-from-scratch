# USER KAGGLE RUNBOOK (CyberGYM 500M)

## Step 1 — Start Kaggle T4 session and upload files
1. Create/start a T4 (or 2xT4) notebook session. If using the 2-GPU path, enable **GPU x2**.
2. Upload these two files as Kaggle dataset inputs:
   - `/tmp/repo_with_tok.tgz` (sha256 `f073e53281bf310ea1db04ab7e63aa9c9dcabc5d3a191d4c02e433b2c4938bcd`, 17 MB)
   - `data/tokenizers/tokenizer.json` from the repo (sha256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`, 20217442 bytes)

## Step 2 — Run Cells 0–4 and 8 first
Run in order. Cells 0–4 are fast. Cell 8 runs syntax/help checks only (no training).

```python
# Cell runner (paste as one cell if you want):
!tar -xzf /kaggle/working/repo_with_tok.tgz -C /kaggle/working
%cd /kaggle/working/train-llm-from-scratch
!pip install -q -r requirements.txt
# verify
import os, sys, hashlib, pathlib
p = pathlib.Path('data/tokenizers/tokenizer.json')
assert p.is_file() and os.path.getsize(p) == 20217442
assert hashlib.sha256(p.read_bytes()).hexdigest() == '19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d'
print('tokenizer OK')
```

```python
# train TINY only:
!python scripts/train_transformer.py --preset tiny --device cuda --steps 10 --eval-every 10 \
  --batch-size 4 --seed 0
```

## Step 3 — Skip Cell 6
- Do **not** run the 2-GPU DDP smoke (`torchrun ... pretrain_base.py`) in a T4 session; it is optional.
- If you have 2 GPUs:
  ```bash
  torchrun --standalone --nproc_per_node=2 scripts/pretrain_base.py --config configs/pretrain_cybergym.json
  ```

## Step 4 — What to download vs recompute
- **Download locally:** checkpoint `.pt` files produced in `/kaggle/working` (they are ignored by git).
- **Recompute locally:** corpus in `data/raw/cybersec/` and any `.h5` files (ignored by git). Exact regeneration:
  ```bash
  python scripts/prepare_cyber_data.py --generate --target-mb 20 --seed 1337 --tokenizer r50k_base
  python scripts/prepare_cyber_data.py --input data/raw/cybersec --out-train data/train/pile_train.h5 \
    --out-dev data/val/pile_dev.h5 --tokenizer r50k_base --max-tokens 300000
  ```
- Do not run more than 10 training steps unless you have a persistent runtime and budget.
