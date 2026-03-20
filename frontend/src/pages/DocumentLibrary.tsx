import { useState, useEffect, useCallback } from 'react'
import { getBackends, getCollections, deleteCollection, clearAllCollections, type Collection } from '../api/backends'

const BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']

const BACKEND_COLORS: Record<string, string> = {
  faiss: 'text-blue-400', chromadb: 'text-purple-400', qdrant: 'text-red-400',
  weaviate: 'text-green-400', milvus: 'text-yellow-400', pgvector: 'text-cyan-400',
}

interface EnhanceStatus {
  collection: string
  graph_exists: boolean
  node_count: number
  edge_count: number
  llm_enhanced: boolean
  llm_enhanced_at: string | null
}

async function getEnhanceStatus(collection: string): Promise<EnhanceStatus> {
  const r = await fetch(`/api/graph/${encodeURIComponent(collection)}/enhance-status`)
  if (!r.ok) throw new Error(`HTTP ${r.status}`)
  return r.json()
}

export default function DocumentLibrary() {
  const [activeTab, setActiveTab] = useState('faiss')
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(false)
  const [backendStatus, setBackendStatus] = useState<Record<string, string>>({})
  const [deleting, setDeleting] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)
  const [message, setMessage] = useState<{ type: 'ok' | 'err'; text: string } | null>(null)

  // Enhance Graph state: keyed by collection name
  const [enhanceStatus, setEnhanceStatus] = useState<Record<string, EnhanceStatus>>({})
  const [enhancing, setEnhancing] = useState<string | null>(null)  // collection being enhanced
  const [enhanceLogs, setEnhanceLogs] = useState<string[]>([])
  const [enhanceTarget, setEnhanceTarget] = useState<string | null>(null)  // expanded log panel

  // Load backend health once
  useEffect(() => {
    getBackends().then((bs) => {
      const s: Record<string, string> = {}
      bs.forEach((b) => { s[b.name] = b.status })
      setBackendStatus(s)
    }).catch(() => {})
  }, [])

  const loadCollections = useCallback(async (backend: string) => {
    setLoading(true)
    setMessage(null)
    try {
      const data = await getCollections(backend)
      setCollections(data)
      // Load enhancement status for all collections in parallel
      data.forEach((col) => {
        getEnhanceStatus(col.name)
          .then((s) => setEnhanceStatus((prev) => ({ ...prev, [col.name]: s })))
          .catch(() => {})
      })
    } catch (e) {
      setCollections([])
      setMessage({ type: 'err', text: `Failed to load collections: ${e}` })
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { loadCollections(activeTab) }, [activeTab, loadCollections])

  const handleDelete = async (collection: string) => {
    if (!confirm(`Delete collection "${collection}" from ${activeTab}?\n\nThis cannot be undone.`)) return
    setDeleting(collection)
    try {
      await deleteCollection(activeTab, collection)
      setMessage({ type: 'ok', text: `Deleted "${collection}" from ${activeTab}` })
      loadCollections(activeTab)
    } catch (e) {
      setMessage({ type: 'err', text: `Delete failed: ${e}` })
    } finally {
      setDeleting(null)
    }
  }

  const handleClearAll = async () => {
    if (!confirm(`⚠ Clear ALL collections from "${activeTab}"?\n\nThis will permanently delete ALL vectors and chunks in this backend.`)) return
    setClearing(true)
    try {
      const result = await clearAllCollections(activeTab)
      const count = result?.count ?? 0
      setMessage({ type: 'ok', text: `Cleared ${count} collection(s) from ${activeTab}` })
      loadCollections(activeTab)
    } catch (e) {
      setMessage({ type: 'err', text: `Clear failed: ${e}` })
    } finally {
      setClearing(false)
    }
  }

  const handleEnhanceGraph = async (collection: string) => {
    if (enhancing) return
    setEnhancing(collection)
    setEnhanceTarget(collection)
    setEnhanceLogs([`[enhance] Starting LLM graph enhancement for "${collection}" on ${activeTab}…`])

    try {
      const res = await fetch(`/api/graph/${encodeURIComponent(collection)}/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend: activeTab, max_chunks: 500 }),
      })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const { job_id } = await res.json()

      const es = new EventSource(`/api/graph/${encodeURIComponent(collection)}/enhance/${job_id}/stream`)
      es.onmessage = (evt) => {
        const data: string = evt.data
        if (data.startsWith('STATUS:')) {
          es.close()
          setEnhancing(null)
          const succeeded = data === 'STATUS:done'
          setMessage({
            type: succeeded ? 'ok' : 'err',
            text: succeeded
              ? `✅ LLM graph enhancement complete for "${collection}"`
              : `❌ Enhancement failed for "${collection}" — check logs below`,
          })
          // Refresh status badge
          getEnhanceStatus(collection)
            .then((s) => setEnhanceStatus((prev) => ({ ...prev, [collection]: s })))
            .catch(() => {})
        } else {
          setEnhanceLogs((prev) => [...prev, data])
        }
      }
      es.onerror = () => {
        es.close()
        setEnhancing(null)
        setMessage({ type: 'err', text: 'Enhancement stream disconnected unexpectedly.' })
      }
    } catch (err) {
      setEnhanceLogs((prev) => [...prev, `ERROR: ${err}`])
      setEnhancing(null)
      setMessage({ type: 'err', text: `Enhancement failed: ${err}` })
    }
  }

  const dotColor = (status?: string) => {
    if (!status) return 'bg-gray-600'
    if (status === 'available' || status === 'connected') return 'bg-green-400'
    if (status === 'error') return 'bg-red-400'
    return 'bg-gray-500'
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Document Library</h1>
        <button
          onClick={handleClearAll}
          disabled={clearing || loading}
          className="px-3 py-1.5 text-xs bg-red-900/50 hover:bg-red-800/70 border border-red-700 text-red-300 rounded disabled:opacity-50 transition-colors"
        >
          {clearing ? 'Clearing...' : `⚠ Clear All in ${activeTab}`}
        </button>
      </div>

      {/* Backend tabs */}
      <div className="flex border-b border-gray-700 overflow-x-auto">
        {BACKENDS.map((b) => (
          <button
            key={b}
            onClick={() => setActiveTab(b)}
            className={`flex items-center gap-1.5 px-4 py-2 text-sm border-b-2 whitespace-nowrap transition-colors ${
              activeTab === b ? 'border-sky-500 text-white' : 'border-transparent text-gray-400 hover:text-gray-200'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${dotColor(backendStatus[b])}`} />
            <span className={activeTab === b ? BACKEND_COLORS[b] : ''}>{b}</span>
          </button>
        ))}
      </div>

      {/* Status message */}
      {message && (
        <div className={`text-sm px-3 py-2 rounded border ${
          message.type === 'ok'
            ? 'bg-green-900/30 border-green-700 text-green-300'
            : 'bg-red-900/30 border-red-700 text-red-300'
        }`}>
          {message.text}
          <button onClick={() => setMessage(null)} className="ml-3 text-gray-400 hover:text-white">✕</button>
        </div>
      )}

      {/* Collections table */}
      {loading ? (
        <div className="text-gray-500 py-8 text-center">Loading collections...</div>
      ) : collections.length === 0 ? (
        <div className="text-gray-600 py-12 text-center">
          <p className="text-4xl mb-3">📭</p>
          <p>No collections found in <span className={BACKEND_COLORS[activeTab]}>{activeTab}</span></p>
          <p className="text-sm mt-1 text-gray-700">Ingest a document first from the Ingestion Studio</p>
        </div>
      ) : (
        <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700 bg-gray-800/50">
                <th className="px-4 py-2.5 text-left text-gray-400 font-medium">Collection</th>
                <th className="px-4 py-2.5 text-left text-gray-400 font-medium">Chunks</th>
                <th className="px-4 py-2.5 text-left text-gray-400 font-medium">Knowledge Graph</th>
                <th className="px-4 py-2.5 text-right text-gray-400 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {collections.map((col) => {
                const es = enhanceStatus[col.name]
                const isEnhancing = enhancing === col.name
                return (
                  <tr key={col.name} className="border-b border-gray-800 hover:bg-gray-800/30 transition-colors">
                    <td className="px-4 py-3">
                      <span className="font-mono text-gray-200">{col.name}</span>
                    </td>
                    <td className="px-4 py-3 text-gray-400">
                      {col.chunk_count > 0
                        ? <span className="text-green-400 font-medium">{col.chunk_count.toLocaleString()}</span>
                        : <span className="text-gray-600">—</span>
                      }
                    </td>
                    <td className="px-4 py-3">
                      {/* Graph status + Enhance button */}
                      <div className="flex items-center gap-2 flex-wrap">
                        {es ? (
                          es.graph_exists ? (
                            <span className="text-xs text-gray-400">
                              {es.node_count}n / {es.edge_count}e
                              {es.llm_enhanced
                                ? <span className="ml-1 text-indigo-400" title={`LLM enhanced at ${es.llm_enhanced_at}`}>✦ LLM</span>
                                : <span className="ml-1 text-gray-600"> (spaCy only)</span>
                              }
                            </span>
                          ) : (
                            <span className="text-xs text-gray-600 italic">no graph yet</span>
                          )
                        ) : (
                          <span className="text-xs text-gray-700">…</span>
                        )}
                        <button
                          onClick={() => handleEnhanceGraph(col.name)}
                          disabled={isEnhancing || !!enhancing}
                          className={`px-2 py-0.5 text-xs rounded border transition-colors flex items-center gap-1 ${
                            es?.llm_enhanced
                              ? 'bg-indigo-900/30 border-indigo-700 text-indigo-300 hover:bg-indigo-800/50'
                              : 'bg-indigo-800/50 border-indigo-600 text-indigo-200 hover:bg-indigo-700/70'
                          } disabled:opacity-40`}
                          title={es?.llm_enhanced
                            ? `Re-run LLM enhancement (last run: ${es.llm_enhanced_at})`
                            : 'Run LLM entity/relation extraction to enrich the knowledge graph'
                          }
                        >
                          {isEnhancing ? <><span className="animate-spin">⟳</span> Enhancing…</> : <>🧠 {es?.llm_enhanced ? 'Re-enhance' : 'Enhance Graph'}</>}
                        </button>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={() => handleDelete(col.name)}
                        disabled={deleting === col.name}
                        className="px-2.5 py-1 text-xs bg-red-900/40 hover:bg-red-800/60 border border-red-800 text-red-400 hover:text-red-300 rounded disabled:opacity-50 transition-colors"
                      >
                        {deleting === col.name ? 'Deleting...' : 'Delete'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>

          <div className="px-4 py-2 border-t border-gray-800 flex items-center justify-between">
            <span className="text-xs text-gray-600">{collections.length} collection(s)</span>
            <button
              onClick={() => loadCollections(activeTab)}
              className="text-xs text-gray-500 hover:text-gray-300"
            >
              ↻ Refresh
            </button>
          </div>
        </div>
      )}

      {/* Enhance log panel — shown when a job is running or just finished */}
      {enhanceTarget && enhanceLogs.length > 0 && (
        <div className="bg-gray-900 border border-indigo-800/50 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-indigo-900/20 border-b border-indigo-800/30">
            <span className="text-sm font-medium text-indigo-300">
              🧠 LLM Enhancement — <span className="font-mono">{enhanceTarget}</span>
              {enhancing ? <span className="ml-2 text-xs text-gray-400 animate-pulse">running…</span> : <span className="ml-2 text-xs text-green-400">done</span>}
            </span>
            <button
              onClick={() => { setEnhanceTarget(null); setEnhanceLogs([]) }}
              className="text-gray-500 hover:text-white text-xs"
            >✕ close</button>
          </div>
          <div className="p-3 max-h-48 overflow-y-auto space-y-0.5">
            {enhanceLogs.map((line, i) => (
              <div key={i} className="text-xs font-mono text-gray-300 whitespace-pre-wrap">{line}</div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
