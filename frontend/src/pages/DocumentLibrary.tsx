import { useState, useEffect, useCallback } from 'react'
import { getBackends, getCollections, deleteCollection, clearAllCollections, deleteGraph, deleteAllGraphs, purgeCollection, purgeAllCollections, type Collection, type PurgeResult, type PurgeAllResult } from '../api/backends'

const BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']

const BACKEND_COLORS: Record<string, string> = {
  faiss: 'text-blue-400', chromadb: 'text-purple-400', qdrant: 'text-red-400',
  weaviate: 'text-green-400', milvus: 'text-yellow-400', pgvector: 'text-cyan-400',
}

/** Returns the job ID of a running ingest for the collection, or null if unlocked. */
async function getCollectionLockStatus(collection: string): Promise<string | null> {
  try {
    const res = await fetch(`/api/ingest/jobs`)
    if (!res.ok) return null
    const jobs: { id: string; status: string; collection_name: string | null }[] = await res.json()
    const running = jobs.find(
      (j) =>
        (j.status === 'running' || j.status === 'pending') &&
        j.collection_name != null &&
        (j.collection_name === collection ||
          j.collection_name.startsWith(collection + '_') ||
          collection.startsWith(j.collection_name + '_'))
    )
    return running?.id ?? null
  } catch {
    return null
  }
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

  // Knowledge Graph state — keyed by collection name (backend-agnostic)
  const [enhanceStatus, setEnhanceStatus] = useState<Record<string, EnhanceStatus>>({})
  const [enhancing, setEnhancing] = useState<string | null>(null)
  const [enhanceLogs, setEnhanceLogs] = useState<string[]>([])
  const [enhanceTarget, setEnhanceTarget] = useState<string | null>(null)
  const [clearingGraph, setClearingGraph] = useState<string | null>(null)   // collection name being cleared
  const [clearingAllGraphs, setClearingAllGraphs] = useState(false)
  // All unique collection names seen across all backends (drives the KG section)
  const [allCollections, setAllCollections] = useState<string[]>([])

  // Purge state
  const [purging, setPurging] = useState<string | null>(null)          // collection name being purged
  const [purgingAll, setPurgingAll] = useState(false)
  const [purgeResult, setPurgeResult] = useState<PurgeResult | PurgeAllResult | null>(null)

  // Load backend health + probe all backends for collection names on mount
  useEffect(() => {
    getBackends().then((bs) => {
      const s: Record<string, string> = {}
      bs.forEach((b) => { s[b.name] = b.status })
      setBackendStatus(s)
    }).catch(() => {})

    // Fetch collections from all backends in parallel to build the full KG collection list
    Promise.allSettled(
      BACKENDS.map((b) => getCollections(b).catch(() => [] as Collection[]))
    ).then((results) => {
      const seen = new Set<string>()
      results.forEach((r) => {
        const cols: Collection[] = r.status === 'fulfilled' ? r.value : []
        cols.forEach((c) => seen.add(c.name))
      })
      const names = Array.from(seen).sort()
      setAllCollections(names)
      // Load enhance status for every unique collection name
      names.forEach((name) => {
        getEnhanceStatus(name)
          .then((s) => setEnhanceStatus((prev) => ({ ...prev, [name]: s })))
          .catch(() => {})
      })
    })
  }, [])

  const loadCollections = useCallback(async (backend: string) => {
    setLoading(true)
    setMessage(null)
    try {
      const data = await getCollections(backend)
      setCollections(data)
      // Merge any new collection names into allCollections
      setAllCollections((prev) => {
        const merged = new Set([...prev, ...data.map((c) => c.name)])
        return Array.from(merged).sort()
      })
      // Load enhance status for newly seen collections
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
    const lockJobId = await getCollectionLockStatus(collection)
    if (lockJobId) {
      setMessage({ type: 'err', text: `Cannot delete: "${collection}" is currently being ingested (job ${lockJobId.slice(0, 8)}…). Wait for ingest to complete.` })
      return
    }
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

  const handleClearGraph = async (collection: string) => {
    if (!confirm(`Clear Knowledge Graph for "${collection}"?\n\nThis deletes the graph snapshot (nodes + edges). The vectors in the vector store are NOT affected. You can rebuild by re-ingesting with ER enabled.`)) return
    setClearingGraph(collection)
    try {
      await deleteGraph(collection)
      setMessage({ type: 'ok', text: `Knowledge Graph cleared for "${collection}"` })
      // Mark graph_exists=false in enhanceStatus — keep the row visible so user can see the cleared status
      setEnhanceStatus((prev) => ({
        ...prev,
        [collection]: {
          collection,
          graph_exists: false,
          node_count: 0,
          edge_count: 0,
          llm_enhanced: false,
          llm_enhanced_at: null,
        },
      }))
    } catch (e) {
      setMessage({ type: 'err', text: `Failed to clear graph: ${e}` })
    } finally {
      setClearingGraph(null)
    }
  }

  const handleClearAllGraphs = async () => {
    if (!confirm(`⚠ Clear ALL Knowledge Graphs?\n\nThis deletes ALL graph snapshots for every collection. Vectors in vector stores are NOT affected.\nYou can rebuild by re-ingesting with ER enabled.`)) return
    setClearingAllGraphs(true)
    try {
      const result = await deleteAllGraphs()
      setMessage({ type: 'ok', text: `Cleared ${result.count} Knowledge Graph(s). Visit the Graph page and refresh to see the updated state.` })
      // Mark all collections as having no graph — keep rows visible so user can see cleared status
      setEnhanceStatus((prev) => {
        const next: Record<string, EnhanceStatus> = {}
        Object.keys(prev).forEach((col) => {
          next[col] = { collection: col, graph_exists: false, node_count: 0, edge_count: 0, llm_enhanced: false, llm_enhanced_at: null }
        })
        // Also mark any allCollections that might not be in prev yet
        allCollections.forEach((col) => {
          if (!next[col]) {
            next[col] = { collection: col, graph_exists: false, node_count: 0, edge_count: 0, llm_enhanced: false, llm_enhanced_at: null }
          }
        })
        return next
      })
    } catch (e) {
      setMessage({ type: 'err', text: `Failed to clear all graphs: ${e}` })
    } finally {
      setClearingAllGraphs(false)
    }
  }

  const handlePurge = async (collection: string) => {
    if (!confirm(
      `⚡ PURGE "${collection}" from ALL storage layers?\n\n` +
      `This will permanently delete:\n` +
      `  • All 6 vector store backends\n` +
      `  • Knowledge Graph snapshot + Kuzu DB\n` +
      `  • SPLADE index files\n` +
      `  • BM25 cache files\n` +
      `  • Pipeline cache entries\n\n` +
      `This cannot be undone.`
    )) return
    const lockJobId = await getCollectionLockStatus(collection)
    if (lockJobId) {
      setMessage({ type: 'err', text: `Cannot purge: "${collection}" is currently being ingested (job ${lockJobId.slice(0, 8)}…). Wait for ingest to complete.` })
      return
    }
    setPurging(collection)
    setPurgeResult(null)
    try {
      const result = await purgeCollection(collection)
      setPurgeResult(result)
      const deleted = result.summary.backends_deleted.length
      setMessage({ type: 'ok', text: `Purged "${collection}" from ${deleted} backend(s). See details below.` })
      loadCollections(activeTab)
      // Remove from allCollections if no backend has it anymore
      setAllCollections((prev) => prev.filter((c) => c !== collection || deleted === 0))
      setEnhanceStatus((prev) => {
        const next = { ...prev }
        delete next[collection]
        return next
      })
    } catch (e) {
      setMessage({ type: 'err', text: `Purge failed: ${e}` })
    } finally {
      setPurging(null)
    }
  }

  const handlePurgeAll = async () => {
    if (!confirm(
      `⚡ PURGE ALL collections from ALL storage layers?\n\n` +
      `This will permanently delete everything across all 6 vector backends, ` +
      `all Knowledge Graphs, all SPLADE indexes, all BM25 caches, and the Kuzu DB.\n\n` +
      `This cannot be undone.`
    )) return
    setPurgingAll(true)
    setPurgeResult(null)
    try {
      const result = await purgeAllCollections()
      setPurgeResult(result)
      setMessage({ type: 'ok', text: `Purged ${result.count} collection(s) from all storage layers.` })
      setCollections([])
      setAllCollections([])
      setEnhanceStatus({})
    } catch (e) {
      setMessage({ type: 'err', text: `Purge all failed: ${e}` })
    } finally {
      setPurgingAll(false)
    }
  }

  const handleEnhanceGraph = async (collection: string) => {
    if (enhancing) return
    // Use the first healthy backend (graph enhance only needs a backend to read chunks from)
    const backend = Object.entries(backendStatus).find(([, s]) => s === 'available' || s === 'connected')?.[0] ?? activeTab
    setEnhancing(collection)
    setEnhanceTarget(collection)
    setEnhanceLogs([`[enhance] Starting LLM graph enhancement for "${collection}" on ${backend}…`])

    try {
      const res = await fetch(`/api/graph/${encodeURIComponent(collection)}/enhance`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ backend, max_chunks: 500 }),
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
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <h1 className="text-xl font-semibold">Document Library</h1>
        <div className="flex items-center gap-2">
          <button
            onClick={handleClearAll}
            disabled={clearing || loading}
            className="px-3 py-1.5 text-xs bg-red-900/50 hover:bg-red-800/70 border border-red-700 text-red-300 rounded disabled:opacity-50 transition-colors"
          >
            {clearing ? 'Clearing...' : `⚠ Clear All in ${activeTab}`}
          </button>
          <button
            onClick={handlePurgeAll}
            disabled={purgingAll || loading}
            title="Holistic purge: removes every collection from ALL 6 backends + graph + SPLADE + BM25 + pipeline cache"
            className="px-3 py-1.5 text-xs bg-orange-900/50 hover:bg-orange-800/70 border border-orange-700 text-orange-300 rounded disabled:opacity-50 transition-colors"
          >
            {purgingAll ? '⟳ Purging All...' : '⚡ Purge Everything'}
          </button>
        </div>
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
                <th className="px-4 py-2.5 text-right text-gray-400 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {collections.map((col) => {
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
                    <td className="px-4 py-3 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          onClick={() => handleDelete(col.name)}
                          disabled={deleting === col.name || purging === col.name}
                          className="px-2.5 py-1 text-xs bg-red-900/40 hover:bg-red-800/60 border border-red-800 text-red-400 hover:text-red-300 rounded disabled:opacity-50 transition-colors"
                        >
                          {deleting === col.name ? 'Deleting...' : 'Delete'}
                        </button>
                        <button
                          onClick={() => handlePurge(col.name)}
                          disabled={purging === col.name || deleting === col.name}
                          title="Purge from ALL storage layers (vectors + graph + SPLADE + BM25 + pipeline cache)"
                          className="px-2.5 py-1 text-xs bg-orange-900/40 hover:bg-orange-800/60 border border-orange-700 text-orange-400 hover:text-orange-300 rounded disabled:opacity-50 transition-colors"
                        >
                          {purging === col.name ? '⟳ Purging...' : '⚡ Purge All'}
                        </button>
                      </div>
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

      {/* Purge result detail panel */}
      {purgeResult && (
        <div className="bg-gray-900 border border-orange-800/50 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-orange-950/30 border-b border-orange-800/30">
            <span className="text-sm font-medium text-orange-300">
              ⚡ Purge Result
              {'collection' in purgeResult
                ? <> — <span className="font-mono">{purgeResult.collection}</span></>
                : <> — {(purgeResult as { count: number }).count} collection(s) purged</>
              }
            </span>
            <button onClick={() => setPurgeResult(null)} className="text-gray-500 hover:text-white text-xs">✕ close</button>
          </div>
          <div className="p-4 space-y-3">
            {'summary' in purgeResult ? (
              // Single collection purge
              <div className="grid grid-cols-2 gap-x-8 gap-y-1 text-xs">
                <div className="text-gray-400">Vector backends deleted</div>
                <div className="text-green-400 font-mono">{purgeResult.summary.backends_deleted.join(', ') || '—'}</div>
                <div className="text-gray-400">Backends skipped (not found)</div>
                <div className="text-gray-500 font-mono">{purgeResult.summary.backends_skipped.join(', ') || '—'}</div>
                {Object.keys(purgeResult.summary.backends_error).length > 0 && <>
                  <div className="text-gray-400">Backend errors</div>
                  <div className="text-red-400 font-mono">{Object.entries(purgeResult.summary.backends_error).map(([b, e]) => `${b}: ${e}`).join('; ')}</div>
                </>}
                <div className="text-gray-400">Graph snapshot deleted</div>
                <div className={purgeResult.summary.graph_snapshot_deleted ? 'text-green-400' : 'text-gray-500'}>{purgeResult.summary.graph_snapshot_deleted ? 'Yes' : 'Not found'}</div>
                <div className="text-gray-400">Kuzu DB cleared</div>
                <div className={purgeResult.summary.kuzu_cleared ? 'text-green-400' : 'text-gray-500'}>{purgeResult.summary.kuzu_cleared ? 'Yes (global)' : 'Not cleared'}</div>
                <div className="text-gray-400">SPLADE files removed</div>
                <div className="text-gray-300 font-mono">{purgeResult.summary.splade_files_removed}</div>
                <div className="text-gray-400">BM25 pkl files removed</div>
                <div className="text-gray-300 font-mono">{purgeResult.summary.bm25_pkls_removed}</div>
                <div className="text-gray-400">Pipeline cache evicted</div>
                <div className="text-gray-300 font-mono">{purgeResult.summary.pipelines_evicted} entries</div>
              </div>
            ) : (
              // Purge-all result
              <div className="space-y-2 text-xs">
                <div className="grid grid-cols-2 gap-x-8 gap-y-1">
                  <div className="text-gray-400">Collections purged</div>
                  <div className="text-green-400 font-mono">{(purgeResult as PurgeAllResult).count}</div>
                  <div className="text-gray-400">Kuzu DB cleared</div>
                  <div className={((purgeResult as PurgeAllResult).kuzu_cleared) ? 'text-green-400' : 'text-gray-500'}>{(purgeResult as PurgeAllResult).kuzu_cleared ? 'Yes' : 'No'}</div>
                  <div className="text-gray-400">BM25 pkl files removed</div>
                  <div className="text-gray-300 font-mono">{(purgeResult as PurgeAllResult).bm25_pkls_removed}</div>
                  <div className="text-gray-400">Pipeline cache evicted</div>
                  <div className="text-gray-300 font-mono">{(purgeResult as PurgeAllResult).pipelines_evicted} entries</div>
                </div>
                {(purgeResult as PurgeAllResult).collections_purged.length > 0 && (
                  <div className="pt-1">
                    <div className="text-gray-500 mb-1">Collections:</div>
                    <div className="font-mono text-gray-400">{(purgeResult as PurgeAllResult).collections_purged.join(', ')}</div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Knowledge Graph section (backend-agnostic, one entry per collection) ── */}
      {allCollections.length > 0 && (
        <div className="bg-gray-900 rounded-lg border border-indigo-900/50 overflow-hidden">
          <div className="px-4 py-2.5 bg-indigo-950/40 border-b border-indigo-900/40 flex items-center justify-between">
            <span className="text-sm font-medium text-indigo-300">🕸 Knowledge Graph</span>
            <div className="flex items-center gap-3">
              <span className="text-xs text-gray-500">Shared across all backends — build once per collection</span>
              <button
                onClick={handleClearAllGraphs}
                disabled={clearingAllGraphs || !!enhancing}
                className="px-2 py-0.5 text-xs rounded border border-red-800 bg-red-900/30 text-red-400 hover:bg-red-800/50 hover:text-red-300 disabled:opacity-40 transition-colors"
                title="Delete ALL knowledge graph snapshots. Vector data is NOT affected."
              >
                {clearingAllGraphs ? '⟳ Clearing...' : '🗑 Clear All Graphs'}
              </button>
            </div>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-800/30">
                <th className="px-4 py-2 text-left text-gray-400 font-medium">Collection</th>
                <th className="px-4 py-2 text-left text-gray-400 font-medium">Nodes</th>
                <th className="px-4 py-2 text-left text-gray-400 font-medium">Edges</th>
                <th className="px-4 py-2 text-left text-gray-400 font-medium">Status</th>
                <th className="px-4 py-2 text-right text-gray-400 font-medium">Action</th>
              </tr>
            </thead>
            <tbody>
              {allCollections.map((name) => {
                const es = enhanceStatus[name]
                const isEnhancing = enhancing === name
                const isClearing = clearingGraph === name
                return (
                  <tr key={name} className="border-b border-gray-800 hover:bg-gray-800/20 transition-colors">
                    <td className="px-4 py-2.5 font-mono text-gray-200 text-xs">{name}</td>
                    <td className="px-4 py-2.5 text-gray-400 text-xs">
                      {es ? (es.graph_exists ? es.node_count : <span className="text-gray-600">—</span>) : <span className="text-gray-700">…</span>}
                    </td>
                    <td className="px-4 py-2.5 text-gray-400 text-xs">
                      {es ? (es.graph_exists ? es.edge_count : <span className="text-gray-600">—</span>) : <span className="text-gray-700">…</span>}
                    </td>
                    <td className="px-4 py-2.5 text-xs">
                      {es ? (
                        es.llm_enhanced
                          ? <span className="text-indigo-400" title={`LLM enhanced at ${es.llm_enhanced_at}`}>✦ LLM enhanced</span>
                          : es.graph_exists
                            ? <span className="text-gray-500">spaCy only</span>
                            : <span className="text-gray-600 italic">no graph yet</span>
                      ) : <span className="text-gray-700">…</span>}
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <div className="flex items-center justify-end gap-2">
                        {/* Clear Graph — only shown when a snapshot exists */}
                        {es?.graph_exists && (
                          <button
                            onClick={() => handleClearGraph(name)}
                            disabled={!!enhancing || !!clearingGraph}
                            className="px-2 py-0.5 text-xs rounded border border-red-800 bg-red-900/20 text-red-400 hover:bg-red-800/40 disabled:opacity-40 transition-colors"
                            title="Delete this collection's Knowledge Graph snapshot"
                          >
                            {isClearing ? '⟳' : '🗑 Clear'}
                          </button>
                        )}
                        <button
                          onClick={() => handleEnhanceGraph(name)}
                          disabled={!!enhancing || !!clearingGraph}
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
                          {isEnhancing
                            ? <><span className="animate-spin">⟳</span> Enhancing…</>
                            : <>🧠 {es?.llm_enhanced ? 'Re-enhance' : 'Enhance Graph'}</>
                          }
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
