import { api } from './client'
import { useStore } from '../store'

// Matches API LLMTraceEntry — one LLM call per retrieval method
export interface LLMTraceEntry {
  method: string           // e.g. "Query Rewriting", "HyDE Expansion"
  system_prompt: string    // system prompt sent to LLM
  user_message: string     // user/context message sent to LLM
  response: string         // raw LLM response
  latency_ms: number       // round-trip latency
}

// Matches API MethodContribution
export interface MethodContribution {
  method: string
  rank: number
  rrf_contribution: number
}

// Matches API SearchResultItem
export interface SearchResultItem {
  chunk_id: string
  text: string
  score: number
  metadata: Record<string, unknown>
  provenance?: string
  confidence?: number
  method_lineage?: MethodContribution[]
  post_processors?: string[]
}

// Matches API BackendSearchResult (one per backend)
export interface BackendSearchResult {
  backend: string
  answer: string
  chunks: SearchResultItem[]
  retrieval_trace: Array<{ method: string; candidates_before: number; candidates_after: number; scores: number[] }>
  llm_traces: LLMTraceEntry[]
  graph_entities: string[]
  graph_paths: string[]
  latency_ms: number
  error?: string
}

// Matches API SearchResponse
export interface ApiSearchResponse {
  query: string
  results: Record<string, BackendSearchResult>  // keyed by backend name
}

// What SearchLab.tsx iterates over
export interface SearchResponse {
  backend: string
  answer?: string
  results: SearchResultItem[]   // same as chunks
  trace?: Array<{ method: string; candidates_before: number; candidates_after: number }>
  llm_traces?: LLMTraceEntry[]
  graph_entities?: string[]
  latency_ms?: number
  error?: string
}

export interface RetrievalStep {
  method: string
  candidates_before: number
  candidates_after: number
}

export interface SearchRequest {
  query: string
  backends: string[]
  collection_name: string   // API field name
  retrieval_methods?: Record<string, boolean>  // kept for compat — mapped to methods below
  methods?: Record<string, boolean>
  top_k?: number
  embedding_model?: string
}

export const search = async (req: SearchRequest): Promise<SearchResponse[]> => {
  // Auto-read embedding model from Zustand store if not explicitly provided
  const embeddingModel = req.embedding_model ?? useStore.getState().embeddingModel
  const body = {
    query: req.query,
    backends: req.backends,
    collection_name: req.collection_name,
    top_k: req.top_k ?? 10,
    methods: req.methods ?? req.retrieval_methods ?? {},
    embedding_model: embeddingModel,
  }
  const resp = await api.post<ApiSearchResponse>('/search', body)
  const apiData = resp.data

  // Convert Dict[backend, BackendSearchResult] → SearchResponse[]
  return Object.values(apiData.results).map((r) => ({
    backend: r.backend,
    answer: r.answer,
    results: r.chunks ?? [],
    trace: r.retrieval_trace,
    llm_traces: r.llm_traces ?? [],
    graph_entities: r.graph_entities,
    latency_ms: r.latency_ms,
    error: r.error,
  }))
}

export const submitFeedback = (chunk_id: string, doc_id: string, feedback: 'up' | 'down') =>
  api.post('/feedback', { chunk_id, doc_id, feedback }).then((r) => r.data)
