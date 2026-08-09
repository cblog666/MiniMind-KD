from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from minimind_kd.protocol import THINK_CLOSE, THINK_OPEN, parse_tool_call


class RewardFunction(Protocol):
    def __call__(self, record: dict[str, Any], completion: str) -> float: ...


def _normalize_answer(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def exact_match_reward(record: dict[str, Any], completion: str) -> float:
    expected = record.get("answer")
    if expected is None:
        return 0.0
    normalized_completion = _normalize_answer(completion)
    normalized_expected = _normalize_answer(str(expected))
    return float(
        normalized_completion == normalized_expected or normalized_completion.endswith(normalized_expected)
    )


_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def numeric_answer_reward(record: dict[str, Any], completion: str) -> float:
    expected = record.get("answer")
    if expected is None:
        return 0.0
    found = _NUMBER.findall(completion.replace(",", ""))
    if not found:
        return 0.0
    try:
        actual = float(found[-1])
        target = float(expected)
    except (TypeError, ValueError):
        return 0.0
    return float(math.isclose(actual, target, rel_tol=1e-6, abs_tol=1e-6))


def reasoning_format_reward(record: dict[str, Any], completion: str) -> float:
    effort = str(record.get("effort", "high"))
    if effort == "none":
        return float(completion.lstrip().startswith(THINK_CLOSE))
    valid = completion.count(THINK_OPEN) <= 1 and completion.count(THINK_CLOSE) == 1
    if THINK_OPEN in completion:
        valid = valid and completion.index(THINK_OPEN) < completion.index(THINK_CLOSE)
    return float(valid)


def tool_schema_reward(record: dict[str, Any], completion: str) -> float:
    if "expected_tool" not in record:
        return 0.0
    try:
        name, _ = parse_tool_call(completion)
    except (ValueError, SyntaxError):
        return 0.0
    return float(name == record["expected_tool"])


REWARD_REGISTRY: dict[str, RewardFunction] = {
    "exact_match": exact_match_reward,
    "numeric": numeric_answer_reward,
    "reasoning_format": reasoning_format_reward,
    "tool_schema": tool_schema_reward,
}


@dataclass
class WeightedRewards:
    weights: dict[str, float]

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(REWARD_REGISTRY)
        if unknown:
            raise ValueError(f"Unknown rewards: {', '.join(sorted(unknown))}")

    def __call__(self, record: dict[str, Any], completion: str) -> float:
        return sum(
            REWARD_REGISTRY[name](record, completion) * weight for name, weight in self.weights.items()
        )


class LocalGenerativeReward:
    """Rubric judge backed only by a caller-supplied local generation function.

    The callable receives a prompt and must return JSON with a numeric `score`.
    No hosted API client, endpoint, or credential handling is included.
    """

    def __init__(self, generate_local: Callable[[str], str]) -> None:
        self.generate_local = generate_local

    def __call__(self, record: dict[str, Any], completion: str) -> float:
        rubric = str(record.get("rubric", "Score correctness and instruction following from 0 to 1."))
        judge_prompt = (
            f"Rubric: {rubric}\nPrompt: {record['prompt']}\nCandidate: {completion}\n"
            'Return only JSON: {"score": <number from 0 to 1>}'
        )
        try:
            score = float(json.loads(self.generate_local(judge_prompt))["score"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        return min(1.0, max(0.0, score))
