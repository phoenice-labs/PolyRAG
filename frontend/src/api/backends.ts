import { api } from './client'

export interface BackendStatus {
  name: string
  status: string
  ping_ms?: number
  requires_docker?: boolean
  error?: string
}

export interface Collection {
  name: string
  chunk_count: number
  index_type?: string
}

export const getBackends = () =>
  api.get<BackendStatus[]>('/backends').then((r) => r.data)

export const getCollections = (backend: string) =>
  api.get<Collection[]>(`/collections/${backend}`).then((r) => r.data)

export const deleteCollection = (backend: string, collection: string) =>
  api.delete(`/collections/${backend}/${encodeURIComponent(collection)}`).then((r) => r.data)

export const clearAllCollections = (backend: string) =>
  api.delete(`/collections/${backend}`).then((r) => r.data)

// ── Knowledge Graph API ──────────────────────────────────────────────────────

export const deleteGraph = (collection: string) =>
  api.delete(`/graph/${encodeURIComponent(collection)}`).then((r) => r.data as { deleted: boolean; collection: string; kuzu_cleared: boolean; pipelines_evicted: number })

export const deleteAllGraphs = () =>
  api.delete('/graph').then((r) => r.data as { deleted: string[]; count: number; kuzu_cleared: boolean; pipelines_evicted: number; errors?: string[] })

// ── Purge API ────────────────────────────────────────────────────────────────

export interface PurgeSummary {
  backends_deleted: string[]
  backends_skipped: string[]
  backends_error: Record<string, string>
  graph_snapshot_deleted: boolean
  kuzu_cleared: boolean
  splade_files_removed: number
  bm25_pkls_removed: number
  pipelines_evicted: number
}

export interface PurgeResult {
  collection: string
  summary: PurgeSummary
  details: Record<string, unknown>
}

export interface PurgeAllResult {
  collections_purged: string[]
  count: number
  kuzu_cleared: boolean
  bm25_pkls_removed: number
  pipelines_evicted: number
  per_collection: Record<string, PurgeSummary>
}

export const purgeCollection = (collection: string) =>
  api.delete(`/purge/${encodeURIComponent(collection)}`).then((r) => r.data as PurgeResult)

export const purgeAllCollections = () =>
  api.delete('/purge').then((r) => r.data as PurgeAllResult)
