"""Tests for the intent-classification gate."""

import pytest

from app.rag.gate import classify


@pytest.mark.parametrize(
    "query, expected_intent, should_retrieve",
    [
        ("What documents do I need for a personal loan?", "docs", True),
        ("What are the processing fees?", "fees", True),
        ("Is foreclosure allowed on floating rate loans?", "policy", True),
        ("What types of loans does Tata Capital offer?", "faq", True),
        ("Your interest rate is too high", "objection", True),
        ("Tell me a joke", "out_of_scope", False),
        ("What is the weather today?", "out_of_scope", False),
    ],
)
def test_classify_intent(query: str, expected_intent: str, should_retrieve: bool):
    intent, retrieve = classify(query)
    assert intent == expected_intent, f"Expected {expected_intent}, got {intent}"
    assert retrieve == should_retrieve


def test_classify_returns_tuple():
    result = classify("How can I apply for a loan?")
    assert isinstance(result, tuple)
    assert len(result) == 2
    intent, retrieve = result
    assert isinstance(intent, str)
    assert isinstance(retrieve, bool)


def test_empty_query_is_out_of_scope():
    intent, retrieve = classify("")
    assert intent == "out_of_scope"
    assert retrieve is False
