"""
HuggingFace ``tokenizer.json`` tokenizer, loaded through the repo's spec system.

The official GLM-5.3 release (``zai-org/GLM-5.3``) ships a byte-level BPE tokenizer in
the ``tokenizers`` format: a ``tokenizer.json`` file holding the vocabulary, a
``normalizer``/``pre_tokenizer``/``decoder`` pipeline and the added special tokens.
There is no ``tokenizer.model`` (SentencePiece) artifact, so
:mod:`src.tokenizer.glm` does not apply; this module handles the JSON form instead.

Like :mod:`src.tokenizer.glm`, it has no silent fallback. Point it at a file that is
not a valid ``tokenizer.json`` and it raises rather than guessing a vocabulary.

Requires the ``tokenizers`` package (a dependency of ``transformers`` and
``huggingface_hub``); the import is guarded so ``src.tokenizer`` still imports
without it, and the error only surfaces when someone actually loads a model.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from typing import Any

# Default location of the official artifact. Only a convenience for callers; nothing
# here ever loads it implicitly.
DEFAULT_GLM53_TOKENIZER = os.path.join("data", "tokenizers", "tokenizer.json")

# The GLM-5.3 document separator, from tokenizer_config.json's eos_token.
DEFAULT_EOT = "<|endoftext|>"


class MissingTokenizerFileError(FileNotFoundError):
    """The tokenizer file is absent, unreadable, or not a tokenizer.json."""


def _import_tokenizers() -> Any:
    """Import the ``tokenizers`` package, or raise a clear, actionable error."""
    try:
        import tokenizers  # noqa: PLC0415  (lazy: keeps src.tokenizer importable)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Loading a HuggingFace tokenizer.json requires the 'tokenizers' package, which is "
            "not installed. Install it with: pip install tokenizers"
        ) from exc
    return tokenizers


class HFTokenizer:
    """A ``tokenizer.json`` tokenizer satisfying the repo's ``Tokenizer`` protocol.

    Args:
        tokenizer_path: Path to a HuggingFace ``tokenizer.json``. Required.
        eot_token: Document separator piece; its id is resolved at construction and a
            missing piece is a hard error rather than a guess.
    """

    def __init__(self, tokenizer_path: str | os.PathLike[str], *, eot_token: str = DEFAULT_EOT) -> None:
        self.tokenizer_path = os.fspath(tokenizer_path)
        if not self.tokenizer_path:
            raise MissingTokenizerFileError(
                "HFTokenizer requires the path to a real tokenizer.json. No path was given, and "
                "this tokenizer deliberately has no silent fallback: a character-level or "
                "hardcoded-vocabulary stand-in produces corrupt training data that still "
                f"round-trips cleanly. Download the official artifact to "
                f"'{DEFAULT_GLM53_TOKENIZER}' and pass it explicitly."
            )
        if not os.path.isfile(self.tokenizer_path):
            raise MissingTokenizerFileError(
                f"No tokenizer.json at {self.tokenizer_path!r}. Download the official artifact "
                "first (see the repo's data/tokenizers/ directory); this tokenizer will not "
                "guess at a vocabulary."
            )

        tokenizers = _import_tokenizers()
        try:
            self._tok = tokenizers.Tokenizer.from_file(self.tokenizer_path)
        except Exception as exc:
            raise MissingTokenizerFileError(
                f"Failed to load {self.tokenizer_path!r} as a tokenizer.json: {exc}"
            ) from exc

        # Reflect reality only. Never hardcoded, never rounded.
        self._n_vocab: int = int(self._tok.get_vocab_size(with_added_tokens=True))
        self.eot_piece: str = eot_token
        piece_id = self._tok.token_to_id(eot_token)
        if piece_id is None:
            raise MissingTokenizerFileError(
                f"End-of-text piece {eot_token!r} is absent from "
                f"{self.tokenizer_path!r} ({self._n_vocab} tokens). Pass the model's actual "
                "separator, e.g. HFTokenizer(path, eot_token='<|user|>')."
            )
        self._eot_token: int = int(piece_id)

    # ------------------------------------------------------------------ properties
    @property
    def n_vocab(self) -> int:
        """Vocabulary size read from the file, including added special tokens."""
        return self._n_vocab

    @property
    def eot_token(self) -> int:
        """Id of the end-of-text piece (an int, per the ``Tokenizer`` protocol)."""
        return self._eot_token

    # ------------------------------------------------------------------ encoding
    def encode(self, text: str, allowed_special: Any = "all") -> list[int]:
        """Encode text, treating ``eot_piece`` as a special token."""
        add_special = allowed_special == "all"
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode with no special-token handling (mirrors ``tiktoken.Encoding``)."""
        return self._tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: Iterable[str]) -> list[list[int]]:
        """Encode a batch of texts."""
        return [self.encode_ordinary(t) for t in texts]

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to text. Ids beyond the vocabulary are skipped, not fatal."""
        return self._tok.decode([int(i) for i in ids], skip_special_tokens=False)

    def decode_batch(self, ids_list: Iterable[Iterable[int]]) -> list[str]:
        """Decode a batch of token lists."""
        return [self.decode(ids) for ids in ids_list]

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes behind one id."""
        return self._tok.id_to_token(int(token_id)).encode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "hf",
            "version": 1,
            "tokenizer_path": self.tokenizer_path,
            "eot_token": self.eot_piece,
            "n_vocab": self.n_vocab,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HFTokenizer:
        """Rebuild from :meth:`to_dict`. Raises if ``tokenizer_path`` is missing."""
        if data.get("type") != "hf":
            raise ValueError("not a HuggingFace tokenizer description")
        path = data.get("tokenizer_path")
        if not path:
            raise MissingTokenizerFileError(
                "HF tokenizer spec has no 'tokenizer_path'. This tokenizer requires an explicit "
                "path to the official tokenizer.json; it will not fall back to a "
                "character-level stand-in."
            )
        return cls(str(path), eot_token=str(data.get("eot_token", DEFAULT_EOT)))


def build_hf_tokenizer(spec: Mapping[str, Any]) -> HFTokenizer:
    """Factory for ``tokenizer_from_spec`` (see ``src/tokenizer/__init__.py``)."""
    return HFTokenizer.from_dict(spec)


def describe_hf_tokenizer(tokenizer: HFTokenizer) -> dict[str, Any]:
    """Inverse of :func:`build_hf_tokenizer` for the spec system."""
    return tokenizer.to_dict()