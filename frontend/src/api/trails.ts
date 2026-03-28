import { api } from './client'

export interface RetrievalTrailStep {
  method: string
  candidates_before: number
  candidates_after: number
}

export interface MethodContributionStats {
  candidates_before?: number
  candidates_after?: number
  delta?: number
  chunks_contributed?: number
  contribution_pct?: number
}

export interface RetrievalTrailRecord {
  timestamp: string
  query: string
  backend: string
  methods_used: Record<string, boolean>
  retrieval_trace: RetrievalTrailStep[]
  result_count: number
  latency_ms: number
  /** Query expansion variants captured during Phase 1 (present when LLM methods were on). */
  query_variants?: {
    rewritten?: string
    paraphrases?: string[]
    hyde_text?: string
    stepback?: string
  }
  /** Per-method marginal contribution stats computed after each search. */
  method_contributions?: Record<string, MethodContributionStats>
}

export const fetchRetrievalTrails = async (
  limit = 50,
  backend?: string,
): Promise<RetrievalTrailRecord[]> => {
  const params = new URLSearchParams({ limit: String(limit) })
  if (backend) params.set('backend', backend)
  const resp = await api.get<RetrievalTrailRecord[]>(`/retrieval-trails?${params}`)
  return resp.data
}

export const clearRetrievalTrails = async (): Promise<void> => {
  await api.delete('/retrieval-trails')
}
