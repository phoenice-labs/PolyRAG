"""
Phase 5 – Query Intelligence Tests
=====================================
LLM-dependent tests require LM Studio running at localhost:1234.
They are marked @pytest.mark.lmstudio and skipped automatically when offline.

Run all:       pytest tests/phase5/ -v
Skip LLM:      pytest tests/phase5/ -v -m "not lmstudio"

Run:  pytest tests/phase5/ -v -m "not lmstudio"
"""
from __future__ import annotations

import pytest

from core.query.llm_client import LMStudioClient
from core.query.context import (
    ConversationContextTracker,
    Turn,
    QueryBundle,
    QueryIntelligencePipeline,
)
from core.query.rewriter import QueryRewriter, MultiQueryGenerator, StepBackPrompter


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "lmstudio: marks tests that require LM Studio running at localhost:1234",
    )


@pytest.fixture(scope="module")
def lm_client():
    return LMStudioClient(
        base_url="http://localhost:1234/v1",
        model="mistralai/ministral-3b",
        temperature=0.1,
        max_tokens=256,
    )


@pytest.fixture(scope="module")
def embedder():
    from core.embedding.sentence_transformer import SentenceTransformerProvider
    return SentenceTransformerProvider({"model": "all-MiniLM-L6-v2", "device": "cpu"})


# ── LM Studio availability check ─────────────────────────────────────────────

class TestLMStudioClient:
    def test_client_instantiates(self):
        client = LMStudioClient()
        assert client.base_url == "http://localhost:1234/v1"
        assert client.model == "mistralai/ministral-3b"

    def test_is_available_returns_bool(self, lm_client):
        result = lm_client.is_available()
        assert isinstance(result, bool)

    @pytest.mark.lmstudio
    def test_complete_returns_string(self, lm_client):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running at localhost:1234")
        result = lm_client.complete("Say 'hello' in one word.")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.lmstudio
    def test_system_prompt_respected(self, lm_client):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running at localhost:1234")
        result = lm_client.complete(
            prompt="What is 2+2?",
            system="Answer in exactly one word with no punctuation.",
        )
        assert len(result.split()) <= 3  # "four" or "4"


# ── Conversation context ──────────────────────────────────────────────────────

class TestConversationContextTracker:
    def test_add_turn(self):
        tracker = ConversationContextTracker(max_turns=3)
        tracker.add_turn("What is Hamlet about?", "Hamlet is a tragedy.")
        assert len(tracker.get_history()) == 1

    def test_max_turns_respected(self):
        tracker = ConversationContextTracker(max_turns=2)
        for i in range(5):
            tracker.add_turn(f"Question {i}", f"Answer {i}")
        assert len(tracker.get_history()) == 2

    def test_history_order(self):
        tracker = ConversationContextTracker(max_turns=5)
        tracker.add_turn("Q1", "A1")
        tracker.add_turn("Q2", "A2")
        history = tracker.get_history()
        assert history[0].query == "Q1"
        assert history[1].query == "Q2"

    def test_as_text_includes_queries(self):
        tracker = ConversationContextTracker()
        tracker.add_turn("What is Hamlet?", "A Danish prince.")
        text = tracker.as_text()
        assert "What is Hamlet?" in text

    def test_clear_resets_history(self):
        tracker = ConversationContextTracker()
        tracker.add_turn("Q", "A")
        tracker.clear()
        assert len(tracker.get_history()) == 0


# ── Query rewriter (requires LM Studio) ──────────────────────────────────────

class TestQueryRewriter:
    @pytest.mark.lmstudio
    def test_rewrite_returns_string(self, lm_client):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running")
        rewriter = QueryRewriter(lm_client)
        result = rewriter.rewrite("what does hamlet say bout dying")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.lmstudio
    def test_rewrite_expands_query(self, lm_client):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running")
        rewriter = QueryRewriter(lm_client)
        result = rewriter.rewrite("RAG")
        # Should expand acronym
        assert len(result) > 3

    def test_rewrite_empty_returns_empty(self, lm_client):
        """Empty query should passthrough without calling LLM."""
        rewriter = QueryRewriter(lm_client)
        result = rewriter.rewrite("")
        assert result == ""


# ── Multi-query generator ─────────────────────────────────────────────────────

class TestMultiQueryGenerator:
    @pytest.mark.lmstudio
    def test_generates_n_queries(self, lm_client):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running")
        gen = MultiQueryGenerator(lm_client, n_queries=3)
        queries = gen.generate("What does Hamlet think about death?")
        assert len(queries) >= 1         # always includes original
        assert queries[0] == "What does Hamlet think about death?"

    @pytest.mark.lmstudio
    def test_paraphrases_are_strings(self, lm_client):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running")
        gen = MultiQueryGenerator(lm_client, n_queries=2)
        queries = gen.generate("Explain Macbeth's ambition.")
        assert all(isinstance(q, str) for q in queries)


# ── Query Intelligence Pipeline ───────────────────────────────────────────────

class TestQueryIntelligencePipeline:
    @pytest.mark.lmstudio
    def test_pipeline_returns_bundle(self, lm_client, embedder):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running")
        pipeline = QueryIntelligencePipeline(
            client=lm_client,
            embedder=embedder,
            enable_stepback=False,
            enable_multi_query=True,
            enable_hyde=True,
        )
        bundle = pipeline.process("What does Hamlet say about existence?")
        assert isinstance(bundle, QueryBundle)
        assert isinstance(bundle.raw, str)
        assert isinstance(bundle.rewritten, str)
        assert isinstance(bundle.paraphrases, list)
        assert bundle.hyde_embedding is not None
        assert len(bundle.hyde_embedding) == 384

    @pytest.mark.lmstudio
    def test_pipeline_conversation_context(self, lm_client, embedder):
        if not lm_client.is_available():
            pytest.skip("LM Studio not running")
        pipeline = QueryIntelligencePipeline(
            client=lm_client, embedder=embedder,
            enable_stepback=False, enable_multi_query=False, enable_hyde=False,
        )
        pipeline.process("What play is Hamlet from?", update_history=True)
        bundle = pipeline.process("What does he say in Act 3?", update_history=False)
        # "he" should be resolved in contextualized query
        assert "hamlet" in bundle.contextualized.lower() or \
               "hamlet" in bundle.rewritten.lower() or \
               len(bundle.contextualized) > len("What does he say in Act 3?")
