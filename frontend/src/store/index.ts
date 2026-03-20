import { create } from 'zustand'

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
}

const defaultMethods = {
  enable_dense: true,
  enable_bm25: true,
  enable_splade: false,   // off by default — downloads ~440 MB model on first use; toggle on after model is cached
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

export const useStore = create<AppState>((set) => ({
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
}))
