import pytest

from minimind_kd.protocol import (
    DSML_TOKEN,
    effort_prompt,
    format_tool_call,
    manage_interleaved_thinking,
    parse_tool_call,
)


def test_dsml_xml_round_trip_and_escaping():
    rendered = format_tool_call(
        "search",
        {"query": "a < b & c", "limit": 3, "filters": ["paper", "code"]},
    )
    assert rendered.startswith(f"{DSML_TOKEN}tool_calls>")
    assert f'{DSML_TOKEN}invoke name="search">' in rendered
    assert f'{DSML_TOKEN}parameter name="query" string="true">' in rendered
    name, arguments = parse_tool_call(rendered)
    assert name == "search"
    assert arguments == {
        "filters": ["paper", "code"],
        "limit": 3,
        "query": "a < b & c",
    }


def test_invalid_tool_call_is_rejected():
    with pytest.raises(ValueError):
        parse_tool_call("<tool_call />")


def test_reasoning_effort_and_history_management():
    assert "</think>" in effort_prompt("hello", "none")
    assert "System:" in effort_prompt("hello", "max")
    messages = [
        {"role": "assistant", "content": "<think>private scratch</think> answer"},
        {"role": "user", "content": "next"},
    ]
    assert manage_interleaved_thinking(messages)[0]["content"] == "answer"
    tool_messages = messages + [{"role": "tool", "content": "result"}]
    assert "<think>" in manage_interleaved_thinking(tool_messages)[0]["content"]
