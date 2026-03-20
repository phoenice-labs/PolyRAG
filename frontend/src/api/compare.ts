import { api } from './client'

export interface CompareRequest {
  /** Use an existing ingested collection (no re-ingestion). */
  collection_name?: string
  /** Paste raw text to ingest on-the-fly. */
  corpus_text?: string
  backends: string[]
  queries: string[]
  full_retrieval: boolean
  repeat_runs?: number
  compare_graph_ab?: boolean   // run each query with graph ON vs OFF
}

export interface CompareChunkPreview {
  chunk_id: string
  text: string
  score: number
}

export interface CompareRow {
  backend: string
  base_top_score: number
  full_top_score: number
  base_kw_hits: number
  avg_score: number
  ingest_time_s: number
  avg_query_latency_ms: number
  latency_p50_ms: number
  latency_p95_ms: number
  total_result_count: number
  // Graph A/B summary
  avg_score_no_graph?: number
  avg_score_delta?: number
  avg_latency_no_graph_ms?: number
  avg_latency_with_graph_ms?: number
  errors: number
}

export interface ComparePerQueryRow {
  backend: string
  query: string
  top_score: number
  kw_hits: number
  avg_score: number
  result_count: number
  query_latency_ms: number
  latency_p50_ms: number
  latency_p95_ms: number
  chunk_ids: string[]
  chunks: CompareChunkPreview[]
  // Graph trail + A/B
  graph_entities?: string[]
  graph_paths?: string[]
  score_no_graph?: number
  score_delta?: number
  latency_no_graph_ms?: number
  latency_with_graph_ms?: number
  error?: string
}

export interface CompareResponse {
  summary: CompareRow[]
  per_query: ComparePerQueryRow[]
}

/** POST /api/compare — returns results synchronously (no polling). */
export const startComparison = (req: CompareRequest) =>
  api.post<CompareResponse>('/compare', req).then((r) => r.data)

/** GET /api/compare/sample-queries — ready-to-use benchmark queries. */
export const getSampleQueries = () =>
  api.get<string[]>('/compare/sample-queries').then((r) => r.data)
