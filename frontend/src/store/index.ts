import { create } from 'zustand'
import { persist } from 'zustand/middleware'

const STORAGE_KEY = 'polyrag-settings'

function loadSetting<T>(key: string, fallback: T): T {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
    return (saved[key] as T) ?? fallback
  } catch {
    return fallback
  }
}

export const EMBEDDING_MODELS = [
  { value: 'all-MiniLM-L6-v2',      label: 'MiniLM-L6-v2 (384-dim · ~80 MB · fast)',         dim: 384 },
  { value: 'BAAI/bge-base-en-v1.5', label: 'BGE-base-en-v1.5 (768-dim · ~440 MB · balanced)', dim: 768 },
  { value: 'BAAI/bge-large-en-v1.5',label: 'BGE-large-en-v1.5 (1024-dim · ~1.3 GB · best)',   dim: 1024 },
] as const

export type EmbeddingModelId = typeof EMBEDDING_MODELS[number]['value']

export interface IngestConfig {
  chunkSize: number
  overlap: number
  strategy: string
  collection: string
  extractEntities: boolean
  enableSplade: boolean
  clearFirst: boolean
}

const DEFAULT_INGEST_CONFIG: IngestConfig = {
  chunkSize: 400,
  overlap: 64,
  strategy: 'sentence',
  collection: 'polyrag_docs',
  extractEntities: false,
  enableSplade: false,
  clearFirst: false,
}

interface AppState {
  selectedBackends: string[]
  setSelectedBackends: (b: string[]) => void
  activeCollection: string
  setActiveCollection: (c: string) => void
  embeddingModel: EmbeddingModelId
  setEmbeddingModel: (m: EmbeddingModelId) => void
  retrievalMethods: Record<string, boolean>
  /** Keys that were auto-enabled by a parent-child dependency rule (not by the user directly). */
  autoEnabledMethods: Record<string, boolean>
  setRetrievalMethod: (key: string, val: boolean) => void
  backendStatuses: Record<string, 'unknown' | 'ok' | 'error'>
  setBackendStatus: (name: string, status: 'unknown' | 'ok' | 'error') => void
  /** Active ingest job IDs keyed by backend — survives page navigation. */
  activeIngestJobs: Record<string, string>
  setActiveIngestJob: (backend: string, jobId: string) => void
  clearActiveIngestJobs: () => void
  /** Persisted ingestion studio form config — survives navigation + browser refresh. */
  ingestConfig: IngestConfig
  setIngestConfig: (patch: Partial<IngestConfig>) => void
}

const defaultMethods = {
  enable_dense: true,
  enable_bm25: true,
  enable_splade: false,   // off by default — requires SPLADE index built at ingest time (enable in Ingestion Studio)
  enable_graph: true,
  enable_rerank: true,
  enable_mmr: true,
  enable_rewrite: false,
  enable_multi_query: false,
  enable_hyde: false,
  enable_raptor: false,
  enable_contextual_rerank: false,
  enable_llm_graph: false,
}

// Parent → children: disabling parent auto-disables children.
const METHOD_PARENTS: Record<string, string[]> = {
  enable_rewrite: ['enable_multi_query'],
  enable_graph: ['enable_llm_graph'],
}

// Child → parent: enabling child auto-enables parent.
const METHOD_CHILD_TO_PARENT: Record<string, string> = {
  enable_multi_query: 'enable_rewrite',
  enable_llm_graph: 'enable_graph',
}

function applyDependencies(
  methods: Record<string, boolean>,
  autoEnabled: Record<string, boolean>,
  key: string,
  val: boolean,
): { methods: Record<string, boolean>; autoEnabled: Record<string, boolean> } {
  const next = { ...methods, [key]: val }
  const nextAuto = { ...autoEnabled }

  // User explicitly toggled this key → no longer considered auto-enabled.
  delete nextAuto[key]

  if (val) {
    // Enabling a child → auto-enable its parent if it wasn't already on.
    const parent = METHOD_CHILD_TO_PARENT[key]
    if (parent && !next[parent]) {
      next[parent] = true
      nextAuto[parent] = true   // mark parent as auto-enabled
    }
  } else {
    // Disabling a parent → cascade-disable its children.
    const children = METHOD_PARENTS[key] ?? []
    for (const child of children) {
      next[child] = false
      delete nextAuto[child]
    }
    // If this key was auto-enabled, clear its marker too.
    delete nextAuto[key]
  }

  return { methods: next, autoEnabled: nextAuto }
}

export const useStore = create<AppState>()(
  persist(
    (set) => ({
      selectedBackends: ['faiss', 'chromadb'],
      setSelectedBackends: (b) => set({ selectedBackends: b }),
      activeCollection: 'polyrag_docs',
      setActiveCollection: (c) => set({ activeCollection: c }),
      embeddingModel: loadSetting<EmbeddingModelId>('embeddingModel', 'all-MiniLM-L6-v2'),
      setEmbeddingModel: (m) => {
        try {
          const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
          localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...saved, embeddingModel: m }))
        } catch { /* ignore */ }
        set({ embeddingModel: m })
      },
      retrievalMethods: defaultMethods,
      autoEnabledMethods: {},
      setRetrievalMethod: (key, val) =>
        set((s) => {
          const result = applyDependencies(s.retrievalMethods, s.autoEnabledMethods, key, val)
          return { retrievalMethods: result.methods, autoEnabledMethods: result.autoEnabled }
        }),
      backendStatuses: {},
      setBackendStatus: (name, status) =>
        set((s) => ({ backendStatuses: { ...s.backendStatuses, [name]: status } })),
      activeIngestJobs: {},
      setActiveIngestJob: (backend, jobId) =>
        set((s) => ({ activeIngestJobs: { ...s.activeIngestJobs, [backend]: jobId } })),
      clearActiveIngestJobs: () => set({ activeIngestJobs: {} }),
      ingestConfig: DEFAULT_INGEST_CONFIG,
      setIngestConfig: (patch) =>
        set((s) => ({ ingestConfig: { ...s.ingestConfig, ...patch } })),
    }),
    {
      name: 'polyrag-store-v1',
      // Only persist the fields that should survive tab navigation.
      // backendStatuses is transient — refreshed by health checks.
      // retrievalMethods and autoEnabledMethods are intentionally NOT persisted
      // so each session starts with a clean, well-known method set.
      partialize: (s) => ({
        ingestConfig: s.ingestConfig,
        selectedBackends: s.selectedBackends,
        activeCollection: s.activeCollection,
        embeddingModel: s.embeddingModel,
        activeIngestJobs: s.activeIngestJobs,
      }),
      storage: {
        // sessionStorage: survives in-page navigation, cleared when tab closes.
        // This prevents stale job IDs bleeding into new sessions.
        getItem: (name) => {
          try { return sessionStorage.getItem(name) } catch { return null }
        },
        setItem: (name, value) => {
          try { sessionStorage.setItem(name, value) } catch { /* ignore */ }
        },
        removeItem: (name) => {
          try { sessionStorage.removeItem(name) } catch { /* ignore */ }
        },
      },
    }
  )
)
