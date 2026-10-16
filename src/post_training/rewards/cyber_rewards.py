"""
Cybersecurity / CyberGYM reward functions for RLVR (GRPO / PPO).

This module is the extension point for the CyberGYM domain.  The repo's other
pipelines (GSM8K / arithmetic) keep using ``reward_gsm8k`` from
:mod:`src.post_training.rewards`; nothing in this file replaces that default.

Design
------
A concrete reward is a subclass of :class:`CyberReward` (also exported as
``CyberRewardBase``) that overrides :meth:`CyberReward.score`.  ``score(response,
prompt) -> float`` is the only method the trainers need to call.  Picking a
different reward means changing the import in ``scripts/train_grpo.py`` (or the
``--set`` override documented below), never editing this class.

The trivial default implemented here rewards *format adherence* only, so that a
dry-run of the GRPO scaffolding yields a finite, non-crashing reward signal.  It
is deliberately not a claim about security quality.

TODOs for real CyberGYM rewards (implement as subclasses or overrides):
    * exploitability scoring -- does a candidate exploit actually run?
    * vulnerability-detection accuracy -- does the answer identify the right CWE?
    * PoC validity -- does a produced proof of concept reproduce the issue?
    * false-positive penalty for over-reporting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


class CyberReward:
    """Base class for CyberGYM rewards.  Subclass and override :meth:`score`."""

    # Scale of the default format-adherence term (kept small to limit hacking).
    FORMAT_SCORE = 1.0

    def score(self, response: str, prompt: str) -> float:  # noqa: ARG002 (prompt informs subclasses)
        """Return a scalar reward for ``response`` to ``prompt`` (default: format only)."""
        return self._format_adherence(response) * self.FORMAT_SCORE

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _format_adherence(response: str) -> float:
        """Trivial default: 1.0 when the response is non-empty and contains code-like text."""
        text = (response or "").strip()
        if not text:
            return 0.0
        # A crude but deterministic stand-in for "looks like a PoC / code answer".
        if "```" in text or "def " in text or "import " in text or "\n" in text:
            return 1.0
        return 0.5

    # Batch helper used by trainers that score many responses at once.
    def score_batch(self, responses: Sequence[str], prompts: Sequence[str]) -> list[float]:
        return [self.score(r, p) for r, p in zip(responses, prompts, strict=False)]


# Backwards/alternative spellings some call sites or tests may expect.
CyberRewardBase = CyberReward
BaseCyberReward = CyberReward


def reward_cyber_format(response: str, prompt: str = "") -> float:
    """Module-level convenience wrapper around the default reward."""
    return CyberReward().score(response, prompt)


# Re-export friendly name for GRPO wiring: an instance with a ``score`` method.
default_cyber_reward = CyberReward()
