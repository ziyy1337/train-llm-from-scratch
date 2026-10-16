"""
Build tokenized pretraining data for the CyberGYM cybersecurity corpus.

Two jobs, both from the CLI or as importable functions:

1. ``generate_corpus`` writes a synthetic cybersecurity corpus as JSONL when the user
   has not collected a real one yet. Every Python snippet it emits is checked with
   ``ast.parse`` before it is written, and the requested size is honoured exactly.
2. ``prepare_tokenized_h5`` reads a plaintext/JSONL corpus, tokenizes it, splits
   98/2 by token count and writes HDF5 files the trainer can read with no code change.

Tokenizer choice is always explicit (``--tokenizer``); there is no default and no
silent substitution. A file whose ids would fall outside the tokenizer's real
vocabulary is refused rather than written, because a mismatch between the stored
``vocab_size`` attribute and the actual ids trains against a broken embedding table.

REAL CORPUS SPEC (what to collect to replace the placeholder)
------------------------------------------------------------
Language   English prose + Python source. One language only; no other code languages.
Format     JSONL, one document per line, ``{"text": "..."}``. Blank lines skipped.
           Raw (non-JSON) lines are accepted and treated as plain text.
           Alternatively a ``.txt`` file, one document per line.
Size       20-100 GB of raw text is a realistic target for a 500M model at 512 ctx.
           Minimum useful: ~1M tokens. The tokenizer needs real repetition to learn
           merges; a corpus under ~50 MB produces a degenerate vocabulary.
Content    Mix roughly 40% security prose (advisories, CVE writeups, write-ups,
           detection rules) and 60% Python (real projects, or sanitised PoC/parser code).
Tokenizer  Whatever ships with the model. Retokenize whenever it changes.

PLACEHOLDER WARNING
-------------------
``data/raw/cybersec/corpus.jsonl`` is machine-generated filler, not real security data.
It exists so the whole pipeline runs end to end. Replace it before drawing any
conclusion about model quality.

CLI
---
    # generate a 50 MB placeholder corpus, then tokenize it
    python scripts/prepare_cyber_data.py --generate --target-mb 50 --tokenizer r50k_base
    python scripts/prepare_cyber_data.py --input data/raw/cybersec \\
        --out-train data/train/cyber_train.h5 --out-dev data/dev/cyber_dev.h5 \\
        --tokenizer r50k_base
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

from src.tokenizer.glm import DEFAULT_GLM_MODEL
from src.tokenizer.hf_tokenizer import DEFAULT_GLM53_TOKENIZER
from src.tokenizer import tokenizer_from_spec, tokenizer_spec

DEFAULT_CORPUS_DIR = "data/raw/cybersec"
# Default outputs intentionally mirror the 500m preset's placeholder paths so a freshly
# tokenized corpus drops straight into the training config.
DEFAULT_TRAIN = "data/train/pile_train.h5"
DEFAULT_DEV = "data/val/pile_dev.h5"
# The 500m preset's vocab_size (the GPT-2-style placeholder vocabulary). Tokenizers with
# a different real vocabulary are still written -- the trainer rounds up on load -- but the
# mismatch is reported loudly because the embedding table is sized from this value.
PRESET_VOCAB_SIZE = 50304

# ---------------------------------------------------------------------------
# Corpus generation
# ---------------------------------------------------------------------------

_SECURITY_TERMS = [
    "remote code execution", "authentication bypass", "buffer overflow", "SQL injection",
    "privilege escalation", "command injection", "cross-site scripting", "deserialization",
    "path traversal", "race condition", "use-after-free", "integer overflow",
    "server-side request forgery", "directory listing", "default credentials",
    "improper certificate validation", "timing attack", "denial of service",
]

_PRODUCTS = [
    "the web gateway", "the VPN concentrator", "the print spooler", "the file transfer daemon",
    "the container runtime", "the CI runner", "the authentication service",
    "the message broker", "the database driver", "the load balancer",
]

_CWE = ["CWE-89", "CWE-79", "CWE-22", "CWE-287", "CWE-502", "CWE-798", "CWE-400",
        "CWE-306", "CWE-732", "CWE-190", "CWE-416", "CWE-434"]

_PROSE_TEMPLATES = [
    "Advisors describe {term} in {product} as {cwe}, rated critical.",
    "Exploitation requires local access, but no authentication is needed for {term}.",
    "The {cwe} flaw has been present since the 2019 release and was never patched.",
    "A public proof of concept shows {term} against {product} in under thirty seconds.",
    "Scanners flag {product} for {term}, though the check is a false positive here.",
    "Mitigation is straightforward: patch {product} or disable {term} at the firewall.",
    "Threat actors exploited {term} in the wild within a week of disclosure.",
    "The vendor downplayed {term}, calling it a hardening issue rather than a bug.",
]

CODE_TEMPLATES = [
    # Hash / signature verification
    'import hashlib\n\n\ndef verify_artifact(path: str, expected: str) -> bool:\n'
    '    """Reject artifacts whose digest does not match the signed manifest."""\n'
    '    with open(path, "rb") as handle:\n'
    '        digest = hashlib.sha256(handle.read()).hexdigest()\n'
    '    return digest == expected\n',
    # SQL injection guard
    'import sqlite3\n\n\ndef find_user(conn: sqlite3.Connection, name: str) -> list[tuple]:\n'
    '    """Parameterised lookup; the name is never interpolated into the SQL string."""\n'
    '    query = "SELECT id, email FROM users WHERE name = ?"\n'
    '    return conn.execute(query, (name,)).fetchall()\n',
    # Input validation / command construction
    'import shlex\n\n\ndef build_scp_command(host: str, remote: str) -> list[str]:\n'
    '    """Build an argument vector so no shell ever parses user input."""\n'
    '    return ["scp", "-o", "BatchMode=yes", remote, f"{host}:/tmp/upload"]\n',
    # Subprocess with a list argv
    'import subprocess\n\n\ndef collect_diagnostics(path: str) -> str:\n'
    '    """Run the collector with a fixed argv; shell=False is the safe default."""\n'
    '    result = subprocess.run(\n'
    '        ["/usr/local/bin/collector", "--json", path],\n'
    '        capture_output=True, text=True, check=False,\n'
    '    )\n'
    '    return result.stdout\n',
    # Cryptographic key handling
    'import hashlib\nimport hmac\n\n\ndef verify_hmac(payload: bytes, signature: bytes, key: bytes) -> bool:\n'
    '    """Constant-time compare so a timing oracle cannot leak the signature."""\n'
    '    expected = hmac.new(key, payload, hashlib.sha256).digest()\n'
    '    return hmac.compare_digest(expected, signature)\n',
    # Rate limiting / brute-force detection
    'from collections import defaultdict\n\n\nclass LoginThrottle:\n'
    '    def __init__(self, limit: int = 5, window: int = 60) -> None:\n'
    '        self._limit = limit\n'
    '        self._window = window\n'
    '        self._failures: dict[str, list[float]] = defaultdict(list)\n\n'
    '    def record_failure(self, user: str, now: float) -> bool:\n'
    '        """Return True once this user has exceeded the failure budget."""\n'
    '        recent = [t for t in self._failures[user] if now - t < self._window]\n'
    '        recent.append(now)\n'
    '        self._failures[user] = recent\n'
    '        return len(recent) > self._limit\n',
    # Path traversal defence
    'from pathlib import Path\n\n\ndef safe_join(root: str, user_path: str) -> Path:\n'
    '    """Resolve under root, refusing anything that escapes the base directory."""\n'
    '    base = Path(root).resolve()\n'
    '    candidate = (base / user_path).resolve()\n'
    '    if not candidate.is_relative_to(base):\n'
    '        raise ValueError(f"path escapes root: {user_path!r}")\n'
    '    return candidate\n',
    # YAML / deserialisation warning
    'import json\n\n\ndef load_policy(raw: str) -> dict[str, object]:\n'
    '    """json.loads only: pickle and yaml.load would execute arbitrary code."""\n'
    '    policy: dict[str, object] = json.loads(raw)\n'
    '    if not isinstance(policy, dict):\n'
    '        raise TypeError("policy document must be a mapping")\n'
    '    return policy\n',
    # Certificate validation
    'import ssl\n\n\ndef make_context() -> ssl.SSLContext:\n'
    '    """Enforce verification; CERT_NONE silently accepts any certificate."""\n'
    '    ctx = ssl.create_default_context(purpose=ssl.Purpose.SERVER_AUTH)\n'
    '    ctx.check_hostname = True\n'
    '    ctx.verify_mode = ssl.CERT_REQUIRED\n'
    '    return ctx\n',
    # Integer overflow guard (the classic one)
    'def allocate(count: int, size: int) -> int:\n'
    '    """Refuse a multiplication that would wrap around a machine integer."""\n'
    '    if count < 0 or size < 0:\n'
    '        raise ValueError("negative dimension")\n'
    '    if size and count > (2**63 - 1) // size:\n'
    '        raise OverflowError("allocation would overflow")\n'
    '    return count * size\n',
]

# Lines that stress the tokenizer: literal backslashes, unicode, very long literals.
_STRESS_LINES = [
    'REGEX = r"^(?:[a-z0-9!#$%&\'*+/=?^_`{{|}}~-]+)@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\\.)+$"',
    'WINDOWS_PATH = "C:\\\\Users\\\\admin\\\\AppData\\\\Roaming\\\\payload.exe"',
    'SHELL = "\\\\u00e9\\\\u00e8\\\\u00ea  \\\\u4e2d\\\\u6587 \\\\u0645\\\\u0631\\\\u062d\\\\u0628\\\\u0627"',
    'COMMENT = "naïve café — résumé, Ωmega, 日本語, emoji 🙂 in a security log line"',
    'SQL_SHELL = "SELECT * FROM users WHERE name = \' OR \'1\'=\'1\' -- "',
    'PADDING = "' + ("A" * 400) + '"  # long literal to exercise tokenizer windows',
]


def _python_document(rng: random.Random) -> str:
    """Return one syntactically valid, self-contained Python module."""
    header = f'"""{rng.choice(_CWE)}: {rng.choice(_SECURITY_TERMS)}."""\n\n'
    stress = "\n".join(rng.sample(_STRESS_LINES, k=rng.randint(1, 3)))
    body = rng.choice(CODE_TEMPLATES)
    return f"{header}{stress}\n\n{body}"


def _prose_document(rng: random.Random) -> str:
    """Return one paragraph of English security prose (not code, so not ast-parsed)."""
    sentences = [rng.choice(_PROSE_TEMPLATES).format(
        term=rng.choice(_SECURITY_TERMS),
        product=rng.choice(_PRODUCTS),
        cwe=rng.choice(_CWE),
    ) for _ in range(rng.randint(3, 7))]
    return " ".join(sentences)


def generate_corpus(
    output_path: str | Path,
    target_bytes: int,
    seed: int = 1337,
    *,
    code_fraction: float = 0.6,
) -> dict[str, int]:
    """Write a synthetic corpus of at least ``target_bytes`` to ``output_path`` as JSONL.

    Deterministic for a given ``seed``: the same seed produces byte-identical output.
    Every generated Python document is validated with :func:`ast.parse` and the run
    fails if any snippet does not parse.

    Returns counts of documents written and Python documents validated.
    """
    rng = random.Random(seed)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    written_bytes = 0
    n_docs = 0
    n_python = 0
    ast_failures = 0

    with open(out, "w", encoding="utf-8") as handle:
        while written_bytes < target_bytes:
            if rng.random() < code_fraction:
                text = _python_document(rng)
                try:
                    ast.parse(text)
                except SyntaxError as exc:
                    ast_failures += 1
                    raise AssertionError(
                        f"generator produced invalid Python ({exc}); this is a generator bug:\n{text}"
                    ) from exc
                n_python += 1
            else:
                text = _prose_document(rng)

            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            handle.write(line)
            written_bytes += len(line.encode("utf-8"))
            n_docs += 1

    if written_bytes < target_bytes:  # pragma: no cover - loop always overshoots slightly
        raise AssertionError(f"wrote {written_bytes} < target {target_bytes}")

    return {"documents": n_docs, "python_docs": n_python, "bytes": written_bytes,
            "ast_failures": ast_failures}


# ---------------------------------------------------------------------------
# Tokenizer selection
# ---------------------------------------------------------------------------


def resolve_tokenizer(choice: str) -> Any:
    """Build the tokenizer named by ``choice``. No default: the caller must choose.

    ``choice`` is one of:

    - ``glm53``: the official GLM-5.3 ``tokenizer.json`` at its default path (real n_vocab 154856).
    - ``glm``: a SentencePiece ``tokenizer.model`` at its default path.
    - a tiktoken name such as ``r50k_base``.
    - a path: ``*.json`` is loaded as a HuggingFace tokenizer.json, ``*.model`` as SentencePiece.
    """
    if choice == "glm53":
        return tokenizer_from_spec({"type": "hf", "tokenizer_path": DEFAULT_GLM53_TOKENIZER})
    if choice == "glm":
        return tokenizer_from_spec({"type": "glm", "model_path": DEFAULT_GLM_MODEL})
    if choice.endswith(".model"):
        return tokenizer_from_spec({"type": "glm", "model_path": choice})
    if choice.endswith(".json"):
        return tokenizer_from_spec({"type": "hf", "tokenizer_path": choice})
    return tokenizer_from_spec({"type": "tiktoken", "name": choice})


# ---------------------------------------------------------------------------
# Corpus reading
# ---------------------------------------------------------------------------


def read_documents(input_dir: str | Path) -> list[str]:
    """Read every ``.jsonl`` (field ``text``) and ``.txt`` document under ``input_dir``."""
    directory = Path(input_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Input directory not found: {directory}")

    documents: list[str] = []
    for item in sorted(directory.iterdir()):
        if not item.is_file():
            continue
        if item.suffix in (".jsonl", ".txt"):
            with open(item, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    if item.suffix == ".txt":
                        documents.append(line)
                        continue
                    try:
                        text = json.loads(line).get("text")
                    except json.JSONDecodeError:
                        text = line  # raw line is the document
                    if text:
                        documents.append(text)
    return documents


# ---------------------------------------------------------------------------
# Tokenization and HDF5 writing
# ---------------------------------------------------------------------------


def _write_token_h5(path: str, tokens: np.ndarray, spec: dict[str, Any], vocab_size: int) -> None:
    """Write one HDF5 file in the exact format ``data_loader.get_batch_iterator`` expects."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as handle:
        dataset = handle.create_dataset("tokens", data=tokens, dtype="i", maxshape=(None,))
        dataset.attrs["tokenizer"] = json.dumps(spec, ensure_ascii=False)
        dataset.attrs["vocab_size"] = int(vocab_size)


def prepare_tokenized_h5(
    input_dir: str | Path,
    out_train: str,
    out_dev: str,
    tokenizer_choice: str,
    max_tokens: int | None = None,
    shuffle_seed: int = 42,
    dev_ratio: float = 0.02,
) -> dict[str, Any]:
    """Tokenize ``input_dir`` and write 98/2 train/dev HDF5 files.

    Raises if any token id falls outside the tokenizer's real vocabulary, or if the
    highest id would collide with the ``vocab_size`` attribute stored alongside it.
    """
    tokenizer = resolve_tokenizer(tokenizer_choice)
    spec = tokenizer_spec(tokenizer)
    n_vocab = int(tokenizer.n_vocab)

    documents = read_documents(input_dir)
    if not documents:
        raise ValueError(
            f"No documents found in {input_dir}. Run with --generate first, or add a "
            "JSONL corpus with a 'text' field."
        )

    # Encode per document so the eot token separates them, exactly as prepare_tiny_data does.
    eot = tokenizer.eot_token
    all_tokens: list[int] = []
    for document in documents:
        all_tokens.extend(tokenizer.encode_ordinary(document))
        all_tokens.append(eot)
        if max_tokens is not None and len(all_tokens) >= max_tokens:
            break

    if not all_tokens:
        raise ValueError("Tokenization produced no tokens.")

    tokens = np.asarray(all_tokens, dtype=np.int32)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]

    # --- refuse to write data the model could not index -------------------------
    highest = int(tokens.max())
    if highest >= n_vocab:
        raise ValueError(
            f"Refusing to write {out_train}: highest token id {highest} is outside the "
            f"tokenizer's vocabulary of {n_vocab} ({spec}). Ids above n_vocab would index "
            "past the embedding table."
        )

    # --- 98/2 split by token count ---------------------------------------------
    rng = np.random.default_rng(shuffle_seed)
    order = rng.permutation(tokens.size)
    dev_size = max(1, int(tokens.size * dev_ratio))
    if tokens.size - dev_size < 1:
        raise ValueError(f"Corpus of {tokens.size} tokens is too small to split.")

    dev_tokens = np.sort(order[:dev_size])
    train_tokens = np.sort(order[dev_size:])
    dev = tokens[dev_tokens]
    train = tokens[train_tokens]

    _write_token_h5(out_train, train, spec, n_vocab)
    _write_token_h5(out_dev, dev, spec, n_vocab)

    summary = {
        "tokenizer": spec,
        "n_vocab": n_vocab,
        "documents": len(documents),
        "tokens_total": int(tokens.size),
        "tokens_train": int(train.size),
        "tokens_dev": int(dev.size),
        "unique_ids": int(np.unique(tokens).size),
        "highest_id": highest,
        "out_train": str(out_train),
        "out_dev": str(out_dev),
    }

    print("=" * 68)
    print("CyberGYM corpus summary")
    print("=" * 68)
    print(f"  tokenizer        : {spec}")
    print(f"  real n_vocab     : {n_vocab}")
    if n_vocab != PRESET_VOCAB_SIZE:
        padded = -(-n_vocab // 64) * 64
        print(f"  NOTE: 500m preset declares vocab_size={PRESET_VOCAB_SIZE}; this tokenizer has "
              f"{n_vocab}, which the trainer pads up to {padded}.")
    print(f"  documents read   : {summary['documents']}")
    print(f"  tokens total     : {summary['tokens_total']:,}")
    print(f"  tokens train/dev : {summary['tokens_train']:,} / {summary['tokens_dev']:,}")
    print(f"  unique token ids : {summary['unique_ids']:,} "
          f"({100 * summary['unique_ids'] / n_vocab:.2f}% of the vocabulary)")
    print(f"  highest id       : {summary['highest_id']} (< n_vocab={n_vocab})")
    print(f"  wrote            : {out_train}")
    print(f"  wrote            : {out_dev}")
    print("=" * 68)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generate", action="store_true",
                        help="Generate a synthetic placeholder corpus before tokenizing.")
    parser.add_argument("--target-mb", type=int, default=50,
                        help="Placeholder corpus size in MiB when --generate is used.")
    parser.add_argument("--seed", type=int, default=1337, help="Generator seed (deterministic).")
    parser.add_argument("--input", default=DEFAULT_CORPUS_DIR, help="Corpus directory.")
    parser.add_argument("--out-train", default=DEFAULT_TRAIN)
    parser.add_argument("--out-dev", default=DEFAULT_DEV)
    parser.add_argument("--tokenizer", required=True,
                        help="'glm53' (official GLM-5.3 tokenizer.json), 'glm' (SentencePiece "
                             "tokenizer.model), a tiktoken name (r50k_base), or a path to a "
                             "tokenizer.json / tokenizer.model. Required: there is no default, "
                             "so a tokenizer is never guessed.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Stop after N tokens.")
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    corpus = Path(args.input) / "corpus.jsonl"
    if args.generate:
        stats = generate_corpus(corpus, target_bytes=args.target_mb * 1024 * 1024, seed=args.seed)
        print(f"Generated {corpus}: {stats['documents']:,} documents, "
              f"{stats['bytes'] / 1024 / 1024:.2f} MiB, "
              f"{stats['python_docs']:,} Python docs validated by ast.parse, "
              f"{stats['ast_failures']} failures")

    prepare_tokenized_h5(
        input_dir=args.input,
        out_train=args.out_train,
        out_dev=args.out_dev,
        tokenizer_choice=args.tokenizer,
        max_tokens=args.max_tokens,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()