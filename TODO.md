# Phoenice-PolyRAG — TODO: Search · Compare · Evaluate

> Final audit of the three core functional areas.  
> Items are grouped by area and priority tier: 🔴 High · 🟡 Medium · 🟢 Low.  
> Status key: `❌ Missing` · `⚠️ Partial` · `✅ Done`

---

## 🔍 SEARCH

### Gaps

| # | Priority | Status | Item | Affected Files |
|---|----------|--------|------|----------------|
| S-1 | 🔴 High | ❌ Missing | **No pagination / offset support** — `top_k` is the only result-count knob; no `offset` field means users can never page through larger result sets | `api/schemas.py` · `api/routers/search.py` · `frontend/src/pages/SearchLab.tsx` |
| S-2 | 🔴 High | ❌ Missing | **No sort options** — results are always score-descending; no sort by confidence, latency, recency, or source | `api/routers/search.py:399-414` · `SearchLab.tsx` |
| S-3 | 🔴 High | ⚠️ Partial | **Metadata filtering is post-retrieval only** — filters applied after RRF fusion, not pushed down to per-backend retrieval; reduces precision vs. pre-filter design; no `filters` field exposed in `SearchRequest` or the UI | `core/retrieval/hybrid.py:236-238` · `api/schemas.py` · `SearchLab.tsx` |
| S-4 | 🔴 High | ⚠️ Partial | **No error recovery for partial backend failures** — `asyncio.gather` without `return_exceptions=True`; Milvus gets one retry but all other backends fail silently with no circuit-breaker | `api/routers/search.py:604-605` · `api/deps.py` |
| S-5 | 🟡 Medium | ❌ Missing | **SPLADE index status not exposed** — no endpoint or UI feedback when a collection lacks a SPLADE index; users toggle SPLADE, get 0 results, receive no explanation | `api/routers/search.py:279-293` · `frontend/src/pages/IngestionStudio.tsx` |
| S-6 | 🟡 Medium | ❌ Missing | **No search analytics / diagnostics endpoint** — retrieval trail JSONL is written to disk but never queryable; no `GET /api/search/analytics` for top queries, failure rate, p50/p95/p99 per method | `data/retrieval_trails.jsonl` · `api/routers/search.py` |
| S-7 | 🟡 Medium | ⚠️ Partial | **Graph entity extraction defaults to spaCy (rule-based)** — `enable_llm_graph` flag is opt-in and requires LM Studio; users who enable Graph toggle see low quality results without understanding why | `core/graph/llm_extractor.py` · `api/schemas.py:38` · `MethodToggle.tsx` |
| S-8 | 🟢 Low | ❌ Missing | **No cross-backend result deduplication** — same chunk returned by two backends appears twice; no `backends_contributed` field in `SearchResultItem` | `api/routers/search.py` · `api/schemas.py:174` |
| S-9 | 🟢 Low | ❌ Missing | **Confidence score bug (graph results show 100%)** — graph hop scores (1.0 / 0.6 / 0.3) are used directly as confidence instead of normalised RRF score | `api/routers/search.py:406` · `core/graph/traversal.py` |

### Enhancements

| # | Priority | Item | Notes |
|---|----------|------|-------|
| SE-1 | 🔴 High | **Circuit-breaker pattern for backend failures** — per-backend failure-rate tracker (rolling 100-request window); skip backends > 50 % failure rate; exponential backoff | `api/deps.py` (new `BackendCircuitBreaker`) · `api/routers/search.py` |
| SE-2 | 🔴 High | **Input validation & quota enforcement** — `top_k` max 100, `query` max 2 000 chars, `backends` max 6; per-IP rate limit 10 req/min | `api/schemas.py` (add `Field` validators) · `api/main.py` |
| SE-3 | 🟡 Medium | **Export search results** — `GET /api/search/{search_id}/export?format=csv\|json`; cache results 1 h with auto-generated `search_id` | New `api/routers/export.py` |
| SE-4 | 🟡 Medium | **Query suggestions / auto-complete** — `GET /api/search/suggestions?q=...`; track last 1 000 unique queries per collection; top-5 by frequency + recency | `api/routers/search.py` |
| SE-5 | 🟡 Medium | **Search analytics dashboard** — `GET /api/search/analytics?days=7` returning top queries, failure rate, p50/p95/p99 latency per method, method effectiveness | `api/routers/search.py` |
| SE-6 | 🟢 Low | **A/B testing mode in SearchLab UI** — side-by-side config-A / config-B result panels; show result-overlap %, score delta, latency ratio | `frontend/src/pages/SearchLab.tsx` |
| SE-7 | 🟢 Low | **In-memory search result cache** — LRU (max 100 entries, 5-min TTL); key: `(query, backends, collection, methods_hash)` | `api/deps.py` |

---

## ⚖️ COMPARE

### Gaps

| # | Priority | Status | Item | Affected Files |
|---|----------|--------|------|----------------|
| C-1 | 🔴 High | ⚠️ Partial | **No per-backend request timeout** — `asyncio.gather` has no timeout; one hanging backend blocks the entire comparison indefinitely | `api/routers/compare.py:315` |
| C-2 | 🔴 High | ❌ Missing | **No per-stage latency breakdown** — total latency is returned but no split for query-expansion, dense search, BM25, RRF, rerank; can't identify bottlenecks | `api/routers/compare.py:118-127` · `api/schemas.py` (CompareSummary) |
| C-3 | 🔴 High | ❌ Missing | **No statistical significance testing** — score/latency differences shown with no confidence intervals or p-values; can't tell if differences are noise | `api/routers/compare.py` · `api/schemas.py` |
| C-4 | 🟡 Medium | ❌ Missing | **`ingest_time_s` always 0 for existing collections** — misleading metric; should be omitted or marked N/A when no text is ingested | `api/routers/compare.py:109` · `ComparisonMatrix.tsx` |
| C-5 | 🟡 Medium | ❌ Missing | **Cannot compare different collections across backends** — `CompareRequest.collection_name` is a single string; can't compare `collection_v1` on Qdrant vs `collection_v2` on Milvus | `api/schemas.py:80` · `api/routers/compare.py:279-304` |
| C-6 | 🟡 Medium | ⚠️ Partial | **Overlap matrix computed client-side only** — no deduplication server-side; no `overlap_matrix: Dict[str, Dict[str, float]]` in `CompareResponse` | `api/routers/compare.py` · `ComparisonMatrix.tsx` |
| C-7 | 🟡 Medium | ❌ Missing | **No comparison report generation** — results returned as JSON only; no Markdown/HTML/PDF report for stakeholder sharing | No export endpoint beyond CSV |
| C-8 | 🟢 Low | ❌ Missing | **No query complexity classification** — all queries treated equally; can't isolate backends that struggle with hard queries | `api/routers/compare.py` |

### Enhancements

| # | Priority | Item | Notes |
|---|----------|------|-------|
| CE-1 | 🔴 High | **Add per-backend `asyncio.wait_for` timeout** — add `timeout: int = 60` to `CompareRequest`; return partial results + per-backend error dict | `api/routers/compare.py` · `api/schemas.py` |
| CE-2 | 🔴 High | **Statistical significance (t-test / Mann-Whitney U)** — when `repeat_runs >= 3`, compute p-value and 95 % CI; flag non-significant differences | `api/routers/compare.py` · `api/schemas.py` |
| CE-3 | 🟡 Medium | **Stateful comparison runs with resumability** — return `comparison_id`; persist to `data/comparisons/{id}.json`; `GET /api/compare/{id}` for results; `PUT /api/compare/{id}` to retry failed backends | `api/routers/compare.py` · `api/schemas.py` |
| CE-4 | 🟡 Medium | **SSE streaming progress during comparison** — yield backend-by-backend progress events; frontend shows live progress bars | `api/deps.py` (SSE helper) · `api/routers/compare.py` |
| CE-5 | 🟡 Medium | **Warmup phase before latency measurement** — run one warmup query per backend before timing starts; keep warmup latency separate | `api/routers/compare.py` |
| CE-6 | 🟡 Medium | **Report generation endpoint** — `GET /api/compare/{run_id}/report?format=markdown\|html\|pdf`; sections: Executive Summary · Latency · Quality · Cost/Performance · Recommendation | New `api/routers/reports.py` |
| CE-7 | 🟡 Medium | **Semantic similarity analysis between backends** — cosine similarity across retrieval sets (not just exact overlap); "Semantic Overlap %" metric | `api/routers/compare.py` |
| CE-8 | 🟡 Medium | **Reproducibility metadata** — store `config_hash`, `dataset_hash`, `timestamp`, `polyrag_version` in `CompareResponse`; `GET /api/compare/baseline-runs?config_hash=...` | `api/routers/compare.py` · `api/schemas.py` |
| CE-9 | 🟢 Low | **Query difficulty estimation** — classify Easy/Medium/Hard by token count + entity count + term rarity; show heatmap: backends × difficulty | `api/routers/compare.py` |
| CE-10 | 🟢 Low | **Cost analysis per backend** — add `cost_per_1k_queries: float` to `config.yaml`; compute estimated cost per run | `config/config.yaml` · `api/schemas.py` · `api/routers/compare.py` |

---

## 📊 EVALUATE

### Gaps

| # | Priority | Status | Item | Affected Files |
|---|----------|--------|------|----------------|
| E-1 | 🔴 High | ❌ Missing | **Evaluation results lost on server restart** — `eval_store` is an in-memory dict in `api/deps.py`; no disk persistence | `api/deps.py` · `api/routers/evaluate.py:238-264` |
| E-2 | 🔴 High | ❌ Missing | **No formal `EvaluateResponse` schema** — `POST /api/evaluate` returns untyped dict; `GET /api/evaluate/{eval_id}` returns raw dict; frontend brittle | `api/schemas.py` · `api/routers/evaluate.py:264` · `frontend/src/api/evaluate.ts:116` |
| E-3 | 🔴 High | ❌ Missing | **RAGAS metrics not structured in API response** — RAGAS scores nested as opaque `{"ragas": {...}}`; no typed `RagasScores` Pydantic model in schemas | `api/routers/evaluate.py:213-218` · `api/schemas.py` |
| E-4 | 🔴 High | ❌ Missing | **No evaluation result export** — no export button in UI; no `GET /api/evaluate/{eval_id}/export?format=csv\|json\|pdf` endpoint | `frontend/src/pages/EvaluationStudio.tsx` · `api/routers/evaluate.py` |
| E-5 | 🔴 High | ❌ Missing | **No bulk Q&A import** — users must manually enter each question; no CSV/JSON file upload endpoint | No file upload endpoint; UI supports manual entry only |
| E-6 | 🟡 Medium | ⚠️ Partial | **Only 4 of 8+ RAGAS metrics implemented** — missing coherence, correctness, harmfulness, maliciousness | `core/evaluation/ragas_scorer.py:161-166` · `RagasResult` model |
| E-7 | 🟡 Medium | ❌ Missing | **No per-method evaluation** — method contributions tracked but not used to isolate quality impact per method; can't tell if Graph or BM25 is adding value | `api/routers/evaluate.py:178-197` |
| E-8 | 🟡 Medium | ❌ Missing | **No evaluation history / baseline tracking** — can't compare eval run from month 1 vs. month 2; no regression detection | No historical storage or comparison endpoint |
| E-9 | 🟡 Medium | ❌ Missing | **No statistical significance testing for RAGAS scores** — two backends may differ by noise; no confidence intervals | `core/evaluation/ragas_scorer.py` · `api/schemas.py` |
| E-10 | 🟡 Medium | ⚠️ Partial | **Word-overlap metrics labelled as RAGAS-equivalent** — "faithfulness" via word overlap is misleading; not semantic faithfulness; displayed alongside RAGAS without visual distinction | `api/routers/evaluate.py:24-74` · `EvaluationStudio.tsx` |
| E-11 | 🟡 Medium | ❌ Missing | **Cannot evaluate on ad-hoc corpus** — `EvaluateRequest` requires `collection_name`; no `corpus_text` option like Compare | `api/schemas.py:112` |
| E-12 | 🟢 Low | ❌ Missing | **No evaluation result sharing / public link** — results are session-local; no way to share via read-only URL | `api/routers/evaluate.py` · `api/deps.py` |

### Enhancements

| # | Priority | Item | Notes |
|---|----------|------|-------|
| EE-1 | 🔴 High | **Persist evaluation results to disk** — write `data/evaluations/{eval_id}.json` on completion; load on server start; auto-delete entries older than 30 days | `api/deps.py` · `api/routers/evaluate.py` |
| EE-2 | 🔴 High | **Add `RagasScores` Pydantic model + `EvaluateResponse` schema** — typed models for all evaluation outputs; remove raw dict returns | `api/schemas.py` · `api/routers/evaluate.py` |
| EE-3 | 🔴 High | **Bulk Q&A import endpoint** — `POST /api/evaluate/import-qa` accepting CSV/JSON; format: `question \| expected_answer \| expected_sources`; return `imported_count` + validation errors | `api/routers/evaluate.py` · `EvaluationStudio.tsx` |
| EE-4 | 🔴 High | **Evaluation result export** — `GET /api/evaluate/{eval_id}/export?format=csv\|json\|pdf`; CSV columns: question · backend · faithfulness · answer_relevancy · context_precision · context_recall · answer_text | `api/routers/evaluate.py` · `EvaluationStudio.tsx` |
| EE-5 | 🟡 Medium | **Extend RAGAS to 8+ metrics** — add coherence, correctness, harmfulness, maliciousness; update `RagasResult` dataclass; gate behind `extended_metrics: bool = False` for performance | `core/evaluation/ragas_scorer.py` · `api/schemas.py` |
| EE-6 | 🟡 Medium | **Quality gate thresholding with alerts** — `quality_gates: Dict[str, float]` in `RagProfile`; compare aggregate scores against gates; return `alerts` list with severity | `api/routers/evaluate.py` · `api/schemas.py` |
| EE-7 | 🟡 Medium | **Per-method evaluation variants** — add `methods_variants: List[RetrievalMethods]` to `EvaluateRequest`; run eval for each variant; return per-variant score matrix | `api/routers/evaluate.py` · `api/schemas.py` |
| EE-8 | 🟡 Medium | **Evaluation history / baseline comparison** — `GET /api/evaluate/compare?eval_id_1=...&eval_id_2=...`; return score deltas, winners/losers per question | `api/routers/evaluate.py` |
| EE-9 | 🟡 Medium | **Statistical significance (bootstrapping)** — when >= 5 questions, compute 95 % CI via bootstrap resampling; return `ci_lower`, `ci_upper` per metric | `api/routers/evaluate.py` · `api/schemas.py` |
| EE-10 | 🟡 Medium | **Batch evaluation scheduling** — `POST /api/evaluate` queues job and returns immediately; `GET /api/evaluate/{eval_id}/status` returns `running\|done\|error` + progress 0–1 | `api/jobs.py` · `api/routers/evaluate.py` |
| EE-11 | 🟡 Medium | **Evaluation report generation** — `GET /api/evaluate/{eval_id}/report?format=markdown\|html\|pdf`; sections: summary · per-question · per-backend · method analysis · recommendations | New `api/routers/reports.py` |
| EE-12 | 🟡 Medium | **Correlation analysis (metrics × method contributions)** — Spearman/Pearson correlation between method contributions and RAGAS scores; return `correlations` heatmap data | `api/routers/evaluate.py` · `api/schemas.py` |
| EE-13 | 🟢 Low | **Pre-built evaluation templates** — `GET /api/evaluate/templates`; domain sets: General · Legal · Medical · Technical; `POST /api/evaluate/load-template?domain=legal` seeds UI | `data/eval_templates/*.json` · `api/routers/evaluate.py` |
| EE-14 | 🟢 Low | **Evaluation reproducibility metadata** — store `polyrag_version`, `ragas_version`, `embedding_model`, `seed` with each result; `GET /api/evaluate/versions` | `core/evaluation/ragas_scorer.py` · `api/routers/evaluate.py` |
| EE-15 | 🟢 Low | **Question difficulty estimation** — classify Easy/Medium/Hard; show heatmap backends × difficulty; flag if all backends fail on hard questions | `api/routers/evaluate.py` |
| EE-16 | 🟢 Low | **Cost-accuracy trade-off plot** — latency vs. avg RAGAS score scatter; highlight Pareto frontier backends; return `pareto_frontier: List[str]` | `api/routers/evaluate.py` · frontend visualisation |
| EE-17 | 🟢 Low | **Confusion matrix for source-hit metrics** — add TP/FP/TN/FN counts; compute precision, recall, F1 for `source_hit` | `api/routers/evaluate.py` · `api/schemas.py` |

---

## Summary — Implementation Coverage

| Area | Component | Status |
|------|-----------|--------|
| **Search** | Hybrid retrieval (Dense + BM25 + SPLADE + Graph) | ✅ Complete |
| | Multi-backend parallel search | ✅ Complete |
| | Query expansion (5 methods) | ✅ Complete |
| | Metadata filtering | ⚠️ Post-filter only |
| | Retrieval trace + method contributions | ✅ Complete |
| | Pagination / offset | ❌ Missing |
| | Sort options | ❌ Missing |
| | Error recovery / circuit breaker | ⚠️ Milvus-only |
| | Search analytics dashboard | ❌ Missing |
| **Compare** | Multi-backend parallel comparison | ✅ Complete |
| | Latency percentiles (P50/P95) | ✅ Complete |
| | Graph A/B testing | ✅ Complete |
| | Method contributions | ✅ Complete |
| | CSV/JSON export | ✅ Complete |
| | Per-stage latency breakdown | ❌ Missing |
| | Statistical significance testing | ❌ Missing |
| | Report generation | ❌ Missing |
| | Per-backend request timeout | ⚠️ None |
| **Evaluate** | RAGAS integration (4 metrics) | ✅ Complete |
| | Word-overlap fallback scoring | ✅ Complete |
| | Q&A management UI | ✅ Complete |
| | Chunk browser + QA generation | ✅ Complete |
| | Per-backend scoring + method contributions | ✅ Complete |
| | Result persistence (disk) | ❌ Missing |
| | Formal Pydantic response schema | ❌ Missing |
| | Result export (CSV/JSON/PDF) | ❌ Missing |
| | Evaluation history / baseline tracking | ❌ Missing |
| | Per-method evaluation | ❌ Missing |
| | Statistical significance testing | ❌ Missing |
| | Bulk Q&A import | ❌ Missing |
| | Extended RAGAS metrics (8+) | ⚠️ 4/8 only |
| | Quality gate thresholding | ❌ Missing |

---

*Generated by final audit — Phoenice-PolyRAG v14.2.0*
