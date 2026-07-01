"""Logit bias helper for SEAR synthesis — nudges cite/connective tokens when eligible."""
from __future__ import annotations

import numpy as np
from llama_cpp import LogitsProcessor

from sear.processor import ExtractiveCopyConstraint, NEG_INF


class BiasedGINLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        constraint: ExtractiveCopyConstraint,
        *,
        dynamic_bias: bool = True,
        static_bias: dict[int, float] | None = None,
    ):
        self.constraint = constraint
        self.dynamic_bias = dynamic_bias
        self.static_bias = static_bias or {}

    def __call__(self, input_ids: list[int], scores: np.ndarray) -> np.ndarray:
        input_arr = np.array(input_ids, dtype=np.intc)
        masked = self.constraint(input_arr, scores)
        biases = dict(self.static_bias)
        if self.dynamic_bias:
            biases.update(self.constraint.eligible_bias_tokens())
        for tok, bias in biases.items():
            if tok < masked.shape[0] and masked[tok] > NEG_INF / 2:
                masked[tok] = masked[tok] + bias
        return masked
