from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from enum import Enum
from typing import Any

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
DSML_TOKEN = "<|DSML|"
DSML_TOOL_CALLS_OPEN = f"{DSML_TOKEN}tool_calls>"
DSML_TOOL_CALLS_CLOSE = f"</{DSML_TOKEN[1:]}tool_calls>"
SPECIAL_TOKENS = [
    THINK_OPEN,
    THINK_CLOSE,
    DSML_TOKEN,
    "<|action|>",
    "<|title|>",
    "<|query|>",
    "<|authority|>",
    "<|domain|>",
    "<|extracted_url|>",
    "<|read_url|>",
]


class ReasoningEffort(str, Enum):
    NONE = "none"
    HIGH = "high"
    MAX = "max"


MAX_REASONING_INSTRUCTION = (
    "Use the available context and tools to reason as deeply as necessary. "
    "Verify intermediate conclusions and do not stop at the first plausible answer."
)


def effort_prompt(prompt: str, effort: ReasoningEffort | str) -> str:
    effort = ReasoningEffort(effort)
    if effort is ReasoningEffort.NONE:
        return f"{prompt}\nAssistant: {THINK_CLOSE}"
    if effort is ReasoningEffort.MAX:
        return f"System: {MAX_REASONING_INSTRUCTION}\nUser: {prompt}\nAssistant: {THINK_OPEN}"
    return f"User: {prompt}\nAssistant: {THINK_OPEN}"


def format_tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Serialize one invocation using DeepSeek V4's public DSML schema."""

    root = ET.Element("tool_calls")
    invocation = ET.SubElement(root, "invoke", name=name)
    for key in sorted(arguments):
        value = arguments[key]
        is_string = isinstance(value, str)
        item = ET.SubElement(
            invocation,
            "parameter",
            name=str(key),
            string="true" if is_string else "false",
        )
        item.text = value if is_string else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    xml = ET.tostring(root, encoding="unicode", short_empty_elements=False)
    return (
        xml.replace("<tool_calls>", DSML_TOOL_CALLS_OPEN)
        .replace("</tool_calls>", DSML_TOOL_CALLS_CLOSE)
        .replace("<invoke", f"{DSML_TOKEN}invoke")
        .replace("</invoke>", f"</{DSML_TOKEN[1:]}invoke>")
        .replace("<parameter", f"{DSML_TOKEN}parameter")
        .replace("</parameter>", f"</{DSML_TOKEN[1:]}parameter>")
    )


def parse_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    start = text.find(DSML_TOOL_CALLS_OPEN)
    end = text.find(DSML_TOOL_CALLS_CLOSE, start + len(DSML_TOOL_CALLS_OPEN))
    if start < 0 or end < 0:
        raise ValueError("Missing DSML tool_calls block")
    end += len(DSML_TOOL_CALLS_CLOSE)
    payload = text[start:end]
    xml = (
        payload.replace(DSML_TOOL_CALLS_OPEN, "<tool_calls>")
        .replace(DSML_TOOL_CALLS_CLOSE, "</tool_calls>")
        .replace(f"{DSML_TOKEN}invoke", "<invoke")
        .replace(f"</{DSML_TOKEN[1:]}invoke>", "</invoke>")
        .replace(f"{DSML_TOKEN}parameter", "<parameter")
        .replace(f"</{DSML_TOKEN[1:]}parameter>", "</parameter>")
    )
    root = ET.fromstring(xml)
    invocations = root.findall("invoke")
    if len(invocations) != 1:
        raise ValueError("Expected exactly one DSML invocation")
    invocation = invocations[0]
    name = invocation.attrib.get("name")
    if not name:
        raise ValueError("DSML invocation has no name")
    arguments: dict[str, Any] = {}
    for item in invocation.findall("parameter"):
        key = item.attrib.get("name")
        if not key or key in arguments:
            raise ValueError("Tool parameter names must be unique and non-empty")
        string_flag = item.attrib.get("string")
        if string_flag == "true":
            arguments[key] = item.text or ""
        elif string_flag == "false":
            try:
                arguments[key] = json.loads(item.text or "null")
            except json.JSONDecodeError as exc:
                raise ValueError(f"DSML parameter {key!r} is not valid JSON") from exc
        else:
            raise ValueError(f"DSML parameter {key!r} needs string=true or string=false")
    return name, arguments


_THINKING = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def manage_interleaved_thinking(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep traces for real tool rounds; prune them for ordinary chat turns.

    This mirrors the context-management behavior described for DeepSeek V4,
    without exposing or manufacturing hidden reasoning at inference time.
    """

    has_tool_round = any(message.get("role") == "tool" for message in messages)
    if has_tool_round:
        return [dict(message) for message in messages]
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        if copy.get("role") == "assistant" and isinstance(copy.get("content"), str):
            copy["content"] = _THINKING.sub("", copy["content"]).strip()
        cleaned.append(copy)
    return cleaned
