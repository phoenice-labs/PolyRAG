import { api } from './client'
import type { MethodContributionStat } from './search'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface QAPair {
  id: string
  question: string
  expected_answer: string
  expected_sources: string[]
}

export interface RetrievalMethodsReq {
  enable_dense: boolean
  enable_bm25: boolean
  enable_graph: boolean
  enable_rerank: boolean
  enable_mmr: boolean
  enable_rewrite: boolean
  enable_multi_query: boolean
  enable_hyde: boolean
  enable_raptor: boolean
  enable_contextual_rerank: boolean
  enable_llm_graph: boolean
}

export interface EvalRequest {
  /** API field name is `questions` (not `qa_pairs`) */
  questions: Omit<QAPair, 'id'>[]
  backends: string[]
  collection_name: string
  methods: RetrievalMethodsReq
}

export interface EvalScore {
  faithfulness: number
  relevance: number
  source_hit: number
  graph_source_hit?: number
}

export interface EvalResult {
  question: string
  backend: string
  answer: string
  scores: EvalScore
  graph_entities?: string[]
  graph_paths?: string[]
  method_contributions?: Record<string, MethodContributionStat>
  error?: string
}

export interface EvalResponse {
  results: EvalResult[]
  summary: Record<string, EvalScore>
  eval_id: string
  collection_name: string
  backends: string[]
}

// ── Chunk browser ─────────────────────────────────────────────────────────────

export interface BrowseChunk {
  id: string
  text: string
  preview: string
  metadata: Record<string, unknown>
}

export interface BrowseChunksResponse {
  chunks: BrowseChunk[]
  total: number
  offset: number
  limit: number
}

export async function browseChunks(
  backend: string,
  collection: string,
  opts: { limit?: number; offset?: number; search?: string } = {},
): Promise<BrowseChunksResponse> {
  const params = new URLSearchParams({
    backend,
    collection,
    limit: String(opts.limit ?? 30),
    offset: String(opts.offset ?? 0),
    search: opts.search ?? '',
  })
  const res = await api.get<BrowseChunksResponse>(`/evaluate/browse-chunks?${params}`)
  return res.data
}

export interface GenerateQAResponse {
  question: string
  answer: string
  chunk_id: string
  source: 'llm' | 'heuristic'
  note?: string
}

export async function generateQA(
  chunkText: string,
  chunkId: string,
): Promise<GenerateQAResponse> {
  const res = await api.post<GenerateQAResponse>('/evaluate/generate-qa', {
    chunk_text: chunkText,
    chunk_id: chunkId,
  })
  return res.data
}

/**
 * Submit evaluation job. The backend runs all questions × backends and returns
 * an eval_id; we then fetch the full results via GET /evaluate/{eval_id}.
 */
export async function runEvaluation(req: EvalRequest): Promise<EvalResponse> {
  // Step 1: kick off evaluation
  const kickoff = await api.post<{ eval_id: string; question_count: number }>('/evaluate', req)
  const { eval_id } = kickoff.data

  // Step 2: fetch full results
  const full = await api.get<{
    eval_id: string
    results: Array<{
      question: string
      expected_answer: string
      expected_sources: string[]
      per_backend: Record<string, {
        answer?: string
        scores?: EvalScore
        graph_entities?: string[]
        graph_paths?: string[]
        error?: string
      }>
    }>
  }>(`/evaluate/${eval_id}`)

  // Step 3: flatten per_backend → EvalResult[]
  const results: EvalResult[] = []
  const summaryAcc: Record<string, { f: number; r: number; s: number; n: number }> = {}

  for (const row of full.data.results) {
    for (const [backend, data] of Object.entries(row.per_backend)) {
      const scores: EvalScore = data.scores
        ? {
            faithfulness: data.scores.faithfulness ?? 0,
            relevance: data.scores.relevance ?? 0,
            source_hit: data.scores.source_hit ?? 0,
            graph_source_hit: data.scores.graph_source_hit,
          }
        : { faithfulness: 0, relevance: 0, source_hit: 0 }
      results.push({
        question: row.question,
        backend,
        answer: data.answer ?? '',
        scores,
        graph_entities: data.graph_entities,
        graph_paths: data.graph_paths,
        method_contributions: (data as Record<string, unknown>).method_contributions as Record<string, MethodContributionStat> | undefined,
        error: data.error,
      })
      if (!summaryAcc[backend]) summaryAcc[backend] = { f: 0, r: 0, s: 0, n: 0 }
      summaryAcc[backend].f += scores.faithfulness
      summaryAcc[backend].r += scores.relevance
      summaryAcc[backend].s += scores.source_hit
      summaryAcc[backend].n += 1
    }
  }

  const summary: Record<string, EvalScore> = {}
  for (const [b, acc] of Object.entries(summaryAcc)) {
    summary[b] = {
      faithfulness: acc.n ? acc.f / acc.n : 0,
      relevance:    acc.n ? acc.r / acc.n : 0,
      source_hit:   acc.n ? acc.s / acc.n : 0,
    }
  }

  return { eval_id, results, summary, collection_name: req.collection_name, backends: req.backends }
}
