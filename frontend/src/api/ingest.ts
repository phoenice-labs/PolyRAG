import { api } from './client'
import { useStore } from '../store'

export interface IngestJobsResponse {
  job_ids: Record<string, string>  // { backend: job_id }
}

export interface IngestJob {
  job_id: string
  status: 'pending' | 'running' | 'done' | 'error'
  backend: string
  created_at: string
  updated_at?: string
  log_lines?: string[]
  result?: { upserted: number; total_chunks: number }
  error?: string
}

export interface IngestRequest {
  text?: string
  corpus_path?: string
  backends: string[]
  collection_name: string
  chunk_strategy: string
  chunk_size: number
  overlap: number
  enable_er: boolean
  enable_splade: boolean   // pre-build SPLADE index during ingestion (similar to ER)
  embedding_model?: string
}

export interface ChunkPreview {
  chunks: Array<{ text: string; start: number; end: number; index: number }>
  total: number
}

export const ingestText = (req: IngestRequest) => {
  // Auto-read embedding model from Zustand store if not explicitly provided
  const embeddingModel = req.embedding_model ?? useStore.getState().embeddingModel
  return api.post<IngestJobsResponse>('/ingest', { ...req, embedding_model: embeddingModel }).then((r) => r.data)
}

export const previewChunks = (text: string, strategy: string, chunk_size: number, chunk_overlap: number) =>
  api.post<ChunkPreview>('/chunks/preview', { text, strategy, chunk_size, chunk_overlap }).then((r) => r.data)

export const getJobStatus = (jobId: string) =>
  api.get<IngestJob>(`/ingest/${jobId}/status`).then((r) => r.data)

export const listJobs = () =>
  api.get<IngestJob[]>('/jobs').then((r) => r.data)
