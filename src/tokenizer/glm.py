"""
SentencePiece-backed GLM tokenizer, wired into the repo's tokenizer spec system.

The official GLM release ships its tokenizer as a SentencePiece ``tokenizer.model``.
This module wraps one of those files so ``src.tokenizer.tokenizer_from_spec`` can build
it from ``{"type": "glm", "model_path": "data/tokenizers/tokenizer.model"}``.

Design rule, deliberately strict
-------------------------------
This tokenizer MUST be explicitly constructed from a real model file. There is no
silent fallback: constructing :class:`GLMTokenizer` without a usable
``tokenizer.model`` raises :class:`MissingTokenizerModelError` immediately. An earlier
version of this file fell back to character-level ids and a hardcoded ``n_vocab`` of
50304; because a character-level round-trip still returns the original text, that
failure was invisible in every smoke test and produced corrupt training data that
looked healthy. A tokenizer that does not exist must fail loudly, not degrade quietly.

There is also no module-level singleton that auto-loads on import: the presence of
``data/tokenizers/tokenizer.model`` must never change behaviour by itself. Callers
pass the path they want (``GLMTokenizer("path/to/tokenizer.model")``), so the same
process can never pick up a file that appeared after start-up.

``sentencepiece`` is imported lazily and guarded, so importing this module (and
therefore ``src.tokenizer``) works without it installed; the error only surfaces when
someone actually tries to load a model.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

# Default location of the official artifact, relative to the repo root. Provided only as
# a convenience default for callers; it is never used without an explicit request.
DEFAULT_GLM_MODEL = os.path.join("data", "tokenizers", "tokenizer.model")

# The end-of-text piece GLM models use to separate documents. Overridable per model,
# because the exact id depends on the vocabulary shipped with the release.
DEFAULT_EOT = "<|endoftext|>"


class MissingTokenizerModelError(FileNotFoundError):
    """No usable SentencePiece model was supplied. Never fall back silently."""


def _import_sentencepiece() -> Any:
    """Import sentencepiece, or raise a clear, actionable error."""
    try:
        import sentencepiece  # noqa: PLC0415  (deliberately lazy: keeps src.tokenizer importable)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Loading a GLM SentencePiece tokenizer requires the 'sentencepiece' package, "
            "which is not installed. Install it with: pip install sentencepiece"
        ) from exc
    return sentencepiece


class GLMTokenizer:
    """A SentencePiece tokenizer that satisfies the repo's ``Tokenizer`` protocol.

    Args:
        model_path: Path to a SentencePiece ``tokenizer.model``. Required.
        eot_token: The end-of-text piece; looked up in the vocabulary at construction.
            Falls back to the model's piece id 0 when the string is absent.
    """

    def __init__(self, model_path: str | os.PathLike[str], *, eot_piece: str = DEFAULT_EOT) -> None:
        self.model_path = os.fspath(model_path)
        if not self.model_path:
            raise MissingTokenizerModelError(
                "GLMTokenizer requires the path to a real SentencePiece 'tokenizer.model'. "
                "No path was given, and this tokenizer deliberately has no silent fallback: "
                "a character-level or hardcoded-vocabulary stand-in produces corrupt training "
                f"data that still round-trips cleanly. Download the official artifact to "
                f"'{DEFAULT_GLM_MODEL}' and pass it explicitly."
            )
        if not os.path.isfile(self.model_path):
            raise MissingTokenizerModelError(
                f"No GLM tokenizer model at {self.model_path!r}. Download the official "
                "SentencePiece artifact first (see the repo's data/tokenizers/ directory); "
                "this tokenizer will not guess at a vocabulary."
            )

        spm = _import_sentencepiece()
        try:
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(self.model_path)
        except Exception as exc:
            raise MissingTokenizerModelError(
                f"Failed to load {self.model_path!r} as a SentencePiece model: {exc}"
            ) from exc

        # Reflect reality only. No default, no rounding: an unloaded tokenizer raises above,
        # so by the time we are here the vocabulary is genuinely known.
        # sentencepiece exposes GetPieceSize(); there is no GetVocabSize.
        self._n_vocab: int = int(self._sp.GetPieceSize())
        self.eot_piece: str = eot_piece

        # Resolve the eot id once, and fail loudly if the model does not carry the piece:
        # a wrong eot id silently corrupts document boundaries in the packed token stream.
        self._eot_token: int | None = None
        for piece_id in range(self._n_vocab):
            if self._sp.IdToPiece(piece_id) == eot_piece:
                self._eot_token = piece_id
                break
        if self._eot_token is None:
            raise MissingTokenizerModelError(
                f"End-of-text piece {eot_piece!r} is absent from the vocabulary in "
                f"{self.model_path!r} ({self._n_vocab} pieces). Pass the model's actual "
                "document separator, e.g. GLMTokenizer(path, eot_piece='<|user|>')."
            )

    # ------------------------------------------------------------------ properties
    @property
    def n_vocab(self) -> int:
        """Vocabulary size read from the model file. Never hardcoded."""
        return self._n_vocab

    @property
    def eot_token(self) -> int:
        """Id of the end-of-text piece (an int, per the ``Tokenizer`` protocol)."""
        assert self._eot_token is not None  # guaranteed by __init__
        return self._eot_token

    # ------------------------------------------------------------------ encoding
    def encode(self, text: str, allowed_special: Any = "all") -> list[int]:
        """Encode text. Special tokens are not given special treatment by SentencePiece."""
        return self._sp.EncodeAsIds(text)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode without special-token handling (mirrors ``tiktoken.Encoding``)."""
        return self._sp.EncodeAsIds(text)

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        """Encode a batch of texts."""
        return self._sp.EncodeAsIds(list(texts))

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to text. Ids outside the vocabulary raise in SentencePiece."""
        return self._sp.DecodeIds([int(i) for i in ids])

    def decode_batch(self, ids_list: Sequence[Sequence[int]]) -> list[str]:
        """Decode a batch of token lists."""
        return [self.decode(ids) for ids in ids_list]

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes behind one id, mirroring ``BPETokenizer.token_bytes``."""
        piece = self._sp.IdToPiece(int(token_id))
        try:
            return piece.encode("utf-8")
        except (AttributeError, UnicodeEncodeError):
            return bytes(str(piece), "utf-8")

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "glm",
            "version": 1,
            "model_path": self.model_path,
            "eot_piece": self.eot_piece,
            "n_vocab": self.n_vocab,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GLMTokenizer:
        """Rebuild from :meth:`to_dict`. Raises if ``model_path`` is missing or unreadable."""
        if data.get("type") != "glm":
            raise ValueError("not a GLM tokenizer description")
        path = data.get("model_path")
        if not path:
            raise MissingTokenizerModelError(
                "GLM tokenizer spec has no 'model_path'. This tokenizer requires an explicit "
                "path to the official SentencePiece model; it will not fall back to a "
                "character-level stand-in."
            )
        return cls(str(path), eot_piece=str(data.get("eot_piece", DEFAULT_EOT)))


def build_glm_tokenizer(spec: Mapping[str, Any]) -> GLMTokenizer:
    """Factory for ``tokenizer_from_spec`` (see ``src/tokenizer/__init__.py``)."""
    return GLMTokenizer.from_dict(spec)


def describe_glm_tokenizer(tokenizer: GLMTokenizer) -> dict[str, Any]:
    """Inverse of :func:`build_glm_tokenizer` for the spec system."""
    return tokenizer.to_dict()