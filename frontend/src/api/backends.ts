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
