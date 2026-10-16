# MANIFEST

Repo root: /home/ziyy1337/Projects/train-llm-from-scratch
Git HEAD: 7c433cf7c768cee1743fed1138857266a530fb45
Repo tarball (minus .git, data, models, logs): /tmp/repo_with_tok.tgz
  sha256 f073e53281bf310ea1db04ab7e63aa9c9dcabc5d3a191d4c02e433b2c4938bcd
  size 17 MB

Artifact hashes:
- data/tokenizers/tokenizer.json: 20217442 bytes, sha256 19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d
- notebooks/cybergym_pipeline.ipynb: sha256 b3c4ecf876c0dad5351156b5ae220e14f78b920c5a2020dde2659d38a30873b7
- notebooks/kaggle_cybergym.ipynb: sha256 81b00654826e52576c844c846ca30a2252de766c3d49ee6dfa0be72fda6e897c

Ignored (regenerate, not committed):
- data/raw/cybersec/corpus.jsonl (synthetic; regenerate with seed 1337)
- data/*.h5 (tokenized; regenerate via scripts/prepare_cyber_data.py)
- models/*.pt (checkpoints; regenerate via train_transformer.py)
- logs/

Checkpoint reference (local scratch only):
- /tmp/cyberverify20/ckpts/checkpoint_step_00000009.pt: 79236079 bytes; step=9; last_completed_step=9; optimizer state count 50
- Verified resume source `/tmp/cyberverify/killrun_ckpts/checkpoint_step_00000001.pt`: step=1, keys include model_state_dict, optimizer_state_dict, config, lr_state

Model code unchanged (no architecture edits): src/models untouched. Config edits limited to config/presets.py (500m) and config/config.py AMP_DTYPE.
No 12-hour run attempted. Maximum executed training: 10 steps.
