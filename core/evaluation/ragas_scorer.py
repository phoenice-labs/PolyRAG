"""
RAGAS evaluation scorer — industry-standard RAG quality metrics.

Wraps the RAGAS library (https://github.com/explodinggradients/ragas) with:
  - LM Studio (OpenAI-compatible, localhost:1234) as the judge LLM
  - PolyRAG's local sentence-transformers as the embedding model
  - Graceful degradation: returns None scores when LM Studio is offline

Metrics computed:
  faithfulness       — are all answer claims grounded in the retrieved chunks?
  answer_relevancy   — does the answer address the question?
  context_precision  — are retrieved chunks ranked by relevance (precision@k)?
  context_recall     — did retrieval surface all information needed to answer?

Each metric is an LLM-judged score in [0, 1]. Higher is always better.

Usage:
    scorer = RagasScorer()
    if scorer.is_available():
        result = scorer.score(
            question="What does Hamlet say?",
            answer="Hamlet reflects on mortality...",
            contexts=["To be or not to be...", "...another chunk..."],
            ground_truth="Hamlet discusses death and suicide.",
        )
        print(result)
        # RagasResult(faithfulness=0.95, answer_relevancy=0.88,
        #             context_precision=0.75, context_recall=0.80,
        #             scorer="ragas-0.4.x", llm="lm-studio")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RagasResult:
    """RAGAS scores for a single (question, answer, contexts, ground_truth) tuple."""

    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None

    scorer: str = "ragas"
    llm: str = "unknown"
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "scorer": self.scorer,
            "llm": self.llm,
            "error": self.error,
        }

    @property
    def available(self) -> bool:
        return self.error is None and self.faithfulness is not None


# ── Scorer ────────────────────────────────────────────────────────────────────

class RagasScorer:
    """
    Computes RAGAS metrics using LM Studio as the judge LLM.

    Parameters
    ----------
    lm_studio_url   : Base URL for LM Studio OpenAI-compatible endpoint.
    lm_studio_model : Model name reported by LM Studio (any value works).
    embed_model     : Sentence-transformers model for embedding-based metrics.
    """

    def __init__(
        self,
        lm_studio_url: str = "http://localhost:1234/v1",
        lm_studio_model: str = "local-model",
        embed_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._url = lm_studio_url
        self._model = lm_studio_model
        self._embed_model = embed_model
        self._llm = None
        self._embeddings = None
        self._ready = False
        self._init_error: Optional[str] = None
        self._try_init()

    def _try_init(self) -> None:
        """Lazy-init LangChain wrappers — fails gracefully if LM Studio is offline."""
        try:
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper

            chat = ChatOpenAI(
                base_url=self._url,
                api_key="lm-studio",
                model=self._model,
                temperature=0,
                timeout=60,
            )
            self._llm = LangchainLLMWrapper(chat)
            self._ready = True
        except Exception as exc:
            self._init_error = str(exc)
            self._ready = False

    def is_available(self) -> bool:
        """Returns True if LM Studio is reachable and RAGAS can run."""
        if not self._ready:
            return False
        try:
            import requests
            resp = requests.get(
                self._url.rstrip("/v1").rstrip("/") + "/v1/models",
                timeout=3,
                headers={"Authorization": "Bearer lm-studio"},
            )
            return resp.status_code == 200
        except Exception:
            return False

    def score(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: str,
    ) -> RagasResult:
        """
        Compute all 4 RAGAS metrics for one Q&A pair.

        Parameters
        ----------
        question    : The user question.
        answer      : The generated answer from PolyRAG.
        contexts    : List of retrieved chunk texts used to generate the answer.
        ground_truth: The expected / reference answer (from your evaluation dataset).

        Returns
        -------
        RagasResult — all scores in [0, 1], or error field set if unavailable.
        """
        if not self._ready:
            return RagasResult(
                scorer="ragas-unavailable",
                llm="none",
                error=self._init_error or "RAGAS not initialised",
            )

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics.collections import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
            )

            data = Dataset.from_dict({
                "question": [question],
                "answer": [answer],
                "contexts": [contexts if contexts else [""]],
                "ground_truth": [ground_truth],
            })

            result = evaluate(
                dataset=data,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=self._llm,
                show_progress=False,
                raise_exceptions=False,
            )

            df = result.to_pandas()
            row = df.iloc[0]

            def _safe(col: str) -> Optional[float]:
                val = row.get(col)
                if val is None:
                    return None
                try:
                    f = float(val)
                    return round(f, 4) if not (f != f) else None  # NaN check
                except (TypeError, ValueError):
                    return None

            return RagasResult(
                faithfulness=_safe("faithfulness"),
                answer_relevancy=_safe("answer_relevancy"),
                context_precision=_safe("context_precision"),
                context_recall=_safe("context_recall"),
                scorer=f"ragas-{_ragas_version()}",
                llm="lm-studio",
            )

        except Exception as exc:
            return RagasResult(
                scorer="ragas-error",
                llm="lm-studio",
                error=str(exc),
            )

    def score_batch(
        self,
        questions: List[str],
        answers: List[str],
        contexts_list: List[List[str]],
        ground_truths: List[str],
    ) -> List[RagasResult]:
        """Score multiple Q&A pairs in a single RAGAS evaluate() call (more efficient)."""
        if not self._ready:
            return [
                RagasResult(scorer="ragas-unavailable", llm="none",
                            error=self._init_error or "RAGAS not initialised")
                for _ in questions
            ]

        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics.collections import (
                faithfulness, answer_relevancy, context_precision, context_recall,
            )

            data = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": [c if c else [""] for c in contexts_list],
                "ground_truth": ground_truths,
            })

            result = evaluate(
                dataset=data,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                llm=self._llm,
                show_progress=False,
                raise_exceptions=False,
            )

            df = result.to_pandas()
            results = []
            ver = _ragas_version()
            for _, row in df.iterrows():
                def _safe(col: str) -> Optional[float]:
                    val = row.get(col)
                    if val is None:
                        return None
                    try:
                        f = float(val)
                        return round(f, 4) if not (f != f) else None
                    except (TypeError, ValueError):
                        return None
                results.append(RagasResult(
                    faithfulness=_safe("faithfulness"),
                    answer_relevancy=_safe("answer_relevancy"),
                    context_precision=_safe("context_precision"),
                    context_recall=_safe("context_recall"),
                    scorer=f"ragas-{ver}",
                    llm="lm-studio",
                ))
            return results

        except Exception as exc:
            return [
                RagasResult(scorer="ragas-error", llm="lm-studio", error=str(exc))
                for _ in questions
            ]


def _ragas_version() -> str:
    try:
        import ragas
        return getattr(ragas, "__version__", "?")
    except Exception:
        return "?"


# ── Module-level singleton ────────────────────────────────────────────────────
# Shared across requests — LLM client is stateless and thread-safe.
_scorer: Optional[RagasScorer] = None


def get_ragas_scorer() -> RagasScorer:
    """Return the module-level RagasScorer singleton (lazy-initialised)."""
    global _scorer
    if _scorer is None:
        _scorer = RagasScorer()
    return _scorer
