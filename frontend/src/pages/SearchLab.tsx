import { useState, useEffect } from 'react'
import BackendSelector from '../components/BackendSelector/BackendSelector'
import MethodToggle from '../components/MethodToggle/MethodToggle'
import ResultCard from '../components/ResultCard/ResultCard'
import RetrievalTrace from '../components/RetrievalTrace/RetrievalTrace'
import { useStore } from '../store'
import { search, type SearchResponse, type LLMTraceEntry } from '../api/search'
import { fetchRetrievalTrails, clearRetrievalTrails, type RetrievalTrailRecord } from '../api/trails'
import { getCollections, type Collection } from '../api/backends'

// ── Collection Picker ─────────────────────────────────────────────────────────

function CollectionPicker() {
  const { selectedBackends, activeCollection, setActiveCollection } = useStore()
  const [collections, setCollections] = useState<Collection[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!selectedBackends.length) return
    setLoading(true)
    // Fetch collections from all selected backends, deduplicate by name
    Promise.allSettled(selectedBackends.map(b => getCollections(b)))
      .then(results => {
        const seen = new Set<string>()
        const merged: Collection[] = []
        for (const r of results) {
          if (r.status === 'fulfilled') {
            for (const col of r.value) {
              if (!seen.has(col.name)) {
                seen.add(col.name)
                merged.push(col)
              }
            }
          }
        }
        merged.sort((a, b) => a.name.localeCompare(b.name))
        setCollections(merged)
        // Auto-select first if current activeCollection not present
        if (merged.length > 0 && !seen.has(activeCollection)) {
          setActiveCollection(merged[0].name)
        }
      })
      .finally(() => setLoading(false))
  }, [selectedBackends.join(',')])  // re-fetch when backends change

  return (
    <div className="bg-gray-900 rounded-lg p-4">
      <div className="text-xs text-gray-400 uppercase tracking-wider mb-2 flex items-center justify-between">
        <span>Collection</span>
        {loading && <span className="text-gray-600 animate-pulse">loading…</span>}
      </div>
      {collections.length === 0 && !loading ? (
        <p className="text-xs text-gray-600 italic">No collections found. Ingest data first.</p>
      ) : (
        <select
          value={activeCollection}
          onChange={e => setActiveCollection(e.target.value)}
          className="w-full bg-gray-800 text-gray-200 border border-gray-700 rounded px-2 py-1.5 text-xs focus:outline-none focus:border-brand-500"
        >
          {collections.map(col => (
            <option key={col.name} value={col.name}>
              {col.name}
              {col.chunk_count != null ? ` (${col.chunk_count.toLocaleString()} chunks)` : ''}
            </option>
          ))}
        </select>
      )}
      {activeCollection && (
        <p className="text-[10px] text-gray-600 mt-1 truncate" title={activeCollection}>
          Active: {activeCollection}
        </p>
      )}
    </div>
  )
}

function LLMTracePanel({ traces }: { traces: LLMTraceEntry[] }) {
  const [openIdx, setOpenIdx] = useState<number | null>(null)

  if (!traces.length) return null

  return (
    <div className="mt-4 bg-gray-900 rounded-lg border border-gray-700">
      <div className="px-4 py-2 border-b border-gray-700 flex items-center gap-2">
        <span className="text-sky-400 font-semibold text-sm">🔬 LLM Trace</span>
        <span className="text-xs text-gray-500">({traces.length} LLM call{traces.length !== 1 ? 's' : ''} made during this search)</span>
      </div>
      <div className="divide-y divide-gray-800">
        {traces.map((t, i) => (
          <div key={i} className="px-4 py-2">
            <button
              onClick={() => setOpenIdx(openIdx === i ? null : i)}
              className="w-full flex items-center gap-3 text-left hover:bg-gray-800/50 rounded px-1 py-1 -mx-1 transition-colors"
            >
              <span className="bg-sky-900 text-sky-300 text-xs font-medium px-2 py-0.5 rounded shrink-0">
                {t.method}
              </span>
              <span className="text-xs text-gray-400 truncate flex-1">
                {t.user_message.slice(0, 100)}…
              </span>
              <span className="text-xs text-gray-600 shrink-0">{t.latency_ms.toFixed(0)}ms</span>
              <span className="text-gray-600 text-xs shrink-0">{openIdx === i ? '▲' : '▼'}</span>
            </button>
            {openIdx === i && (
              <div className="mt-2 space-y-2 text-xs font-mono max-h-[60vh] overflow-y-auto pr-1">
                <div className="bg-gray-800 rounded p-2">
                  <div className="text-amber-400 font-semibold mb-1 font-sans">System Prompt</div>
                  <pre className="text-gray-300 whitespace-pre-wrap break-words max-h-40 overflow-y-auto">{t.system_prompt}</pre>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <div className="text-blue-400 font-semibold mb-1 font-sans">User Message / Context</div>
                  <pre className="text-gray-300 whitespace-pre-wrap break-words max-h-64 overflow-y-auto">{t.user_message}</pre>
                </div>
                <div className="bg-gray-800 rounded p-2">
                  <div className="text-green-400 font-semibold mb-1 font-sans">LLM Response <span className="text-gray-500 font-normal">({t.latency_ms.toFixed(0)}ms)</span></div>
                  <pre className="text-gray-200 whitespace-pre-wrap break-words max-h-64 overflow-y-auto">{t.response}</pre>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Retrieval Trails Panel ────────────────────────────────────────────────────

function TrailsPanel({ searchCount, activeBackends }: { searchCount: number; activeBackends: string[] }) {
  const [trails, setTrails] = useState<RetrievalTrailRecord[]>([])
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState(false)
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)
  // Default filter to the first active backend so user only sees relevant trails immediately
  const [backendFilter, setBackendFilter] = useState<string>('__active__')

  const resolvedFilter = backendFilter === '__active__'
    ? (activeBackends.length === 1 ? activeBackends[0] : undefined)
    : (backendFilter === '__all__' ? undefined : backendFilter)

  const load = async () => {
    setLoading(true)
    try {
      setTrails(await fetchRetrievalTrails(50, resolvedFilter))
    } catch {
      // silently ignore if API not available
    } finally {
      setLoading(false)
    }
  }

  const handleClear = async () => {
    await clearRetrievalTrails()
    setTrails([])
  }

  // Reload whenever the panel is open AND a new search finishes (searchCount or filter changes).
  useEffect(() => {
    if (open) load()
  }, [open, searchCount, backendFilter, activeBackends.join(',')])

  // Known backends for the filter dropdown
  const ALL_BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']

  // Display label for current filter
  const filterLabel = backendFilter === '__all__'
    ? 'All backends'
    : backendFilter === '__active__'
      ? activeBackends.length === 1
        ? activeBackends[0]
        : `Active (${activeBackends.join(', ')})`
      : backendFilter

  return (
    <div className="border border-gray-700 rounded-lg bg-gray-900">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-3 text-sm text-gray-300 hover:text-white"
      >
        <span>📋 Retrieval Trails</span>
        <span>{open ? '▼' : '▶'}</span>
      </button>
      {open && (
        <div className="border-t border-gray-700">
          <div className="flex items-center justify-between px-4 py-2 border-b border-gray-800 flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <span className="text-xs text-gray-500">{trails.length} trail{trails.length !== 1 ? 's' : ''}</span>
              {/* Backend filter */}
              <span className="text-xs text-gray-600">|</span>
              <span className="text-xs text-gray-500">Backend:</span>
              <select
                value={backendFilter}
                onChange={(e) => setBackendFilter(e.target.value)}
                className="text-xs bg-gray-800 border border-gray-700 text-gray-300 rounded px-1.5 py-0.5 focus:outline-none focus:border-sky-500"
              >
                <option value="__active__">Active ({activeBackends.length === 1 ? activeBackends[0] : activeBackends.join(', ') || '—'})</option>
                <option value="__all__">All backends</option>
                {ALL_BACKENDS.map((b) => (
                  <option key={b} value={b}>{b}</option>
                ))}
              </select>
            </div>
            <div className="flex gap-2">
              <button onClick={load} className="text-xs text-sky-400 hover:text-sky-300">↻ Refresh</button>
              <button onClick={handleClear} className="text-xs text-red-400 hover:text-red-300">✕ Clear</button>
            </div>
          </div>
          {loading && <div className="p-4 text-xs text-gray-500">Loading...</div>}
          {!loading && trails.length === 0 && (
            <div className="p-4 text-xs text-gray-600 text-center">
              No trails for <span className="text-gray-400 font-medium">{filterLabel}</span>. Run a search to record a trail.
            </div>
          )}
          <div className="divide-y divide-gray-800 max-h-96 overflow-y-auto">
            {trails.map((trail, i) => (
              <div key={i} className="px-4 py-2">
                <button
                  onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                  className="w-full flex items-center gap-2 text-left hover:bg-gray-800/40 rounded px-1 -mx-1 py-1 transition-colors"
                >
                  <span className="text-[10px] text-gray-600 w-32 shrink-0">
                    {new Date(trail.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="text-xs text-sky-400 shrink-0">[{trail.backend}]</span>
                  <span className="text-xs text-gray-300 flex-1 truncate">{trail.query}</span>
                  <span className="text-xs text-gray-600 shrink-0">{trail.result_count} results</span>
                  <span className="text-xs text-gray-600 shrink-0">{trail.latency_ms.toFixed(0)}ms</span>
                  <span className="text-gray-600 text-xs shrink-0">{expandedIdx === i ? '▲' : '▼'}</span>
                </button>
                {expandedIdx === i && (
                  <div className="mt-2 space-y-2 max-h-72 overflow-y-auto pr-1">
                    {/* Methods used */}
                    <div className="flex flex-wrap gap-1">
                      {Object.entries(trail.methods_used)
                        .filter(([, v]) => v)
                        .map(([k]) => (
                          <span key={k} className="text-[10px] bg-gray-800 text-gray-400 px-1.5 py-0.5 rounded">
                            {k.replace('enable_', '')}
                          </span>
                        ))}
                    </div>
                    {/* Method Contribution Bars */}
                    {trail.method_contributions && Object.keys(trail.method_contributions).length > 0 && (
                      <div className="mt-2">
                        <div className="text-xs font-semibold text-gray-500 mb-1">Method Contributions</div>
                        <div className="flex flex-wrap gap-1">
                          {Object.entries(trail.method_contributions)
                            .filter(([, s]) => (s.contribution_pct ?? 0) > 0)
                            .sort(([, a], [, b]) => (b.contribution_pct ?? 0) - (a.contribution_pct ?? 0))
                            .map(([method, stats]) => (
                              <div key={method} className="flex items-center gap-1 text-xs bg-blue-50 rounded px-2 py-0.5">
                                <span className="font-medium text-blue-700">{method.replace('enable_', '')}</span>
                                <div className="w-16 bg-blue-100 rounded-full h-1.5">
                                  <div
                                    className="bg-blue-500 h-1.5 rounded-full"
                                    style={{ width: `${Math.min(stats.contribution_pct ?? 0, 100)}%` }}
                                  />
                                </div>
                                <span className="text-blue-600">{stats.contribution_pct?.toFixed(1)}%</span>
                              </div>
                            ))
                          }
                        </div>
                      </div>
                    )}
                    {/* Query expansion variants */}
                    {trail.query_variants && Object.keys(trail.query_variants).length > 0 && (
                      <div className="space-y-1 border-l-2 border-gray-700 pl-2">
                        {trail.query_variants.rewritten && (
                          <div className="text-[10px] text-gray-400">
                            <span className="text-amber-400 font-medium">Rewritten: </span>
                            {trail.query_variants.rewritten}
                          </div>
                        )}
                        {trail.query_variants.paraphrases?.map((p, j) => (
                          <div key={j} className="text-[10px] text-gray-400">
                            <span className="text-sky-400 font-medium">Paraphrase {j + 1}: </span>
                            {p}
                          </div>
                        ))}
                        {trail.query_variants.hyde_text && (
                          <div className="text-[10px] text-gray-400">
                            <span className="text-purple-400 font-medium">HyDE: </span>
                            <span className="italic">{trail.query_variants.hyde_text.slice(0, 150)}{trail.query_variants.hyde_text.length > 150 ? '…' : ''}</span>
                          </div>
                        )}
                        {trail.query_variants.stepback && (
                          <div className="text-[10px] text-gray-400">
                            <span className="text-green-400 font-medium">Step-back: </span>
                            {trail.query_variants.stepback}
                          </div>
                        )}
                      </div>
                    )}
                    {/* Per-phase retrieval trace */}
                    <div className="space-y-1">
                      {trail.retrieval_trace.map((step, j) => {
                        const reduction = step.candidates_before > 0
                          ? Math.round((1 - step.candidates_after / step.candidates_before) * 100)
                          : 0
                        return (
                          <div key={j} className="flex items-center gap-2 text-xs text-gray-400">
                            <span className="w-40 truncate text-gray-500">{step.method}</span>
                            <span className="tabular-nums">{step.candidates_before} → {step.candidates_after}</span>
                            {step.candidates_before > 0 && step.candidates_after < step.candidates_before && (
                              <span className="text-amber-600">−{reduction}%</span>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Search Lab Guide Modal ────────────────────────────────────────────────────

function SearchGuideModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[620px] max-h-[88vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg">🔍</span>
            <h2 className="text-base font-semibold text-white">Search Lab — Guide</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* Purpose */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What is Search Lab?</h3>
            <p className="text-gray-300 leading-relaxed">
              Search Lab is your <span className="text-white font-medium">interactive retrieval workbench</span>. Run queries
              against one or more vector database backends simultaneously, mix and match retrieval methods, and
              inspect every retrieved chunk — including scores, sources, and full LLM reasoning traces.
            </p>
          </section>

          {/* Retrieval methods */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Retrieval Methods</h3>
            <div className="space-y-1.5 text-xs">
              {[
                { group: 'Always Available', color: 'text-sky-400', methods: [
                  { name: 'Dense Vector', desc: 'Semantic similarity search using your chosen embedding model (MiniLM / BGE). Best for conceptual, paraphrased, or natural-language queries.' },
                  { name: 'BM25 Keyword', desc: 'Classic keyword frequency search. Best for exact terminology, IDs, names, and short precise queries. Complements dense retrieval.' },
                  { name: 'Knowledge Graph', desc: 'Entity-relation graph search via spaCy NER + Kuzu. Surfaces chunks connected through named entities (people, places, events).' },
                  { name: 'Cross-Encoder Rerank', desc: 'Re-scores top candidates using a cross-encoder model for higher precision. Adds ~200–500 ms but significantly improves ranking.' },
                  { name: 'MMR Diversity', desc: 'Maximal Marginal Relevance — reduces redundancy by penalising duplicate-content chunks. Use when results feel repetitive.' },
                  { name: 'SPLADE', desc: 'Sparse neural retrieval (naver/splade-cocondenser-selfdistil, ~110 MB, Apache 2.0). Learned term expansion — better than BM25 for domain-specific queries. Requires SPLADE index built at ingest time (toggle "Enable SPLADE Index" in Ingestion Studio).' },
                ]},
                { group: 'LLM-Required (needs LM Studio)', color: 'text-amber-400', methods: [
                  { name: 'Query Rewrite', desc: 'LLM rephrases your query before retrieval — fixes typos, expands abbreviations, clarifies ambiguous questions.' },
                  { name: 'Multi-Query', desc: 'LLM generates 3 alternative phrasings of your query and merges results — improves recall for complex questions.' },
                  { name: 'HyDE', desc: 'Hypothetical Document Embeddings — LLM drafts a synthetic answer, then retrieves chunks similar to that answer rather than the question.' },
                  { name: 'RAPTOR', desc: 'Retrieval-Augmented Paths Through Organised Reasoning — hierarchical summarisation tree built at ingest, retrieves at multiple abstraction levels.' },
                  { name: 'Contextual Rerank', desc: 'LLM re-reads top chunks in context of your query and re-ranks them by relevance. Highest quality, highest latency.' },
                  { name: 'LLM Graph Extract', desc: 'LLM extracts entities and relations directly from the query for graph-aware retrieval — more accurate than spaCy for complex relationships.' },
                ]},
              ].map(({ group, color, methods }) => (
                <div key={group} className="mb-3">
                  <div className={`font-semibold mb-1.5 ${color}`}>{group}</div>
                  <div className="space-y-1">
                    {methods.map(({ name, desc }) => (
                      <div key={name} className="flex gap-2">
                        <span className="text-gray-200 font-mono shrink-0 w-44">{name}</span>
                        <span className="text-gray-400 leading-relaxed">{desc}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* How to use */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">How to Run a Search</h3>
            <ol className="space-y-2 text-gray-300 leading-relaxed text-xs">
              {[
                'Select one or more backends in the left sidebar. FAISS and ChromaDB work without Docker.',
                'Toggle retrieval methods in the Method Toggle panel. Start with Dense + BM25 + Rerank for best baseline quality.',
                'Set Results per backend (top_k) — 10 is a good default. Increase to 20+ for broad exploratory queries.',
                'Type your query and press Enter or Search. Results appear per backend, each with chunk text, score, source, and latency.',
                'Expand a chunk card to read the full text. Hover over the score to see which methods contributed to RRF fusion.',
                'Use the Retrieval Trail panel at the bottom to see a persistent log of past queries and which chunks were returned.',
              ].map((text, i) => (
                <li key={i} className="flex gap-3">
                  <span className="w-5 h-5 rounded-full bg-brand-500 text-white text-xs flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
                  <span>{text}</span>
                </li>
              ))}
            </ol>
          </section>

          {/* A/B mode callout */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">A/B Mode</h3>
            <div className="bg-gray-800 rounded p-3 border border-gray-700 text-xs text-gray-300">
              Enable <span className="text-white font-medium">A/B Mode</span> (sidebar checkbox) to lock results into a strict 2-column layout —
              Backend A on the left, Backend B on the right. Ideal for head-to-head comparisons.
              Select exactly 2 backends first, then enable A/B Mode. Click the <span className="text-white font-medium">?</span> next to the A/B checkbox for a detailed guide.
            </div>
          </section>

          {/* Tips */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Tips</h3>
            <ul className="space-y-1.5 text-xs text-gray-400 list-disc list-inside">
              <li>RRF (Reciprocal Rank Fusion) combines all active methods — <span className="text-gray-200">more methods = better recall</span>, not just the latest one.</li>
              <li>LLM methods add 1–5 seconds latency. Disable them for fast exploratory searches, enable for final quality checks.</li>
              <li>The search history dropdown (below the search bar) lets you replay recent queries — great for comparing results after changing settings.</li>
              <li>Results are <span className="text-gray-200">not cached</span> — each search hits the live backend, so settings changes take effect immediately.</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  )
}

// ── A/B Mode Guide Modal ──────────────────────────────────────────────────────

function ABModeGuideModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[580px] max-h-[85vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg">⚖️</span>
            <h2 className="text-base font-semibold text-white">A/B Mode — Guide</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* What it is */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What is A/B Mode?</h3>
            <p className="text-gray-300 leading-relaxed">
              A/B Mode renders exactly <span className="text-white font-medium">2 backends side-by-side</span> in a
              fixed two-column layout, making it easy to visually compare the answers and retrieved chunks from two
              different vector stores against the same query.
            </p>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                <div className="text-sky-400 font-semibold mb-1">Normal Mode</div>
                <div className="text-gray-400">All selected backends shown in a responsive grid — useful for comparing 3+ backends at once.</div>
              </div>
              <div className="bg-gray-800 rounded p-2.5 border border-brand-500">
                <div className="text-brand-400 font-semibold mb-1">A/B Mode ✅</div>
                <div className="text-gray-400">Locks to 2 columns — always exactly backend A on the left, backend B on the right. Clean head-to-head.</div>
              </div>
            </div>
          </section>

          {/* How to use */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">How to Use</h3>
            <ol className="space-y-2.5 text-gray-300 leading-relaxed">
              {[
                { n: '1', text: 'Select exactly 2 backends in the Backend Selector (left sidebar). Example: faiss and chromadb.' },
                { n: '2', text: 'Enable A/B Mode by toggling the checkbox in the sidebar.' },
                { n: '3', text: 'Type your query and press Search or Enter.' },
                { n: '4', text: 'The two backends are shown side-by-side. Backend A is left (first selected), Backend B is right (second selected).' },
                { n: '5', text: 'Compare answers, chunk content, scores, and latency directly across the two columns.' },
              ].map(({ n, text }) => (
                <li key={n} className="flex gap-3">
                  <span className="w-5 h-5 rounded-full bg-brand-500 text-white text-xs flex items-center justify-center shrink-0 mt-0.5">{n}</span>
                  <span>{text}</span>
                </li>
              ))}
            </ol>
          </section>

          {/* What to look for */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">What to Compare</h3>
            <div className="space-y-2">
              {[
                { icon: '💬', label: 'Answer quality', desc: 'Does each backend produce a different answer for the same query? Which is more accurate or complete?' },
                { icon: '📄', label: 'Retrieved chunks', desc: 'Are the same chunks appearing? Different chunks suggest the vector store is indexing or scoring differently.' },
                { icon: '🎯', label: 'Relevance scores', desc: 'Higher scores indicate the chunk is a stronger match for your query. Compare top scores across backends.' },
                { icon: '⚡', label: 'Latency', desc: 'Each backend shows its response time in ms. In-process backends (FAISS, ChromaDB) are typically faster than server backends (Qdrant, Milvus).' },
                { icon: '⚠', label: 'Errors', desc: 'If a backend shows an error, it may not be running — check Docker with: docker compose -f docker-compose.polyrag.yml ps' },
              ].map(({ icon, label, desc }) => (
                <div key={label} className="flex gap-3 bg-gray-800/60 rounded-lg p-3">
                  <span className="text-base shrink-0 mt-0.5">{icon}</span>
                  <div>
                    <div className="font-medium text-gray-200">{label}</div>
                    <div className="text-gray-400 text-xs mt-0.5 leading-relaxed">{desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* How to verify */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">How to Verify Results</h3>
            <div className="space-y-2 text-gray-300 leading-relaxed">
              <p>To confirm A/B results are meaningful and not a fluke:</p>
              <ul className="space-y-1.5 text-xs list-none">
                <li className="flex gap-2"><span className="text-sky-400 shrink-0">→</span><span><span className="text-white font-medium">Check the Retrieval Trace</span> (bottom of page) — shows how many candidates each retrieval method produced and how they were filtered. If one backend has 0 candidates from Dense, it may not be ingested.</span></li>
                <li className="flex gap-2"><span className="text-sky-400 shrink-0">→</span><span><span className="text-white font-medium">Check Retrieval Trails</span> — the collapsible panel below shows the full per-method breakdown for each search, including BM25, Dense, and SPLADE candidate counts.</span></li>
                <li className="flex gap-2"><span className="text-sky-400 shrink-0">→</span><span><span className="text-white font-medium">Ensure both backends are ingested</span> with the same data and embedding model. Go to Ingest Studio → select both backends → run ingestion. Mixing MiniLM vs BGE collections gives incomparable results.</span></li>
                <li className="flex gap-2"><span className="text-sky-400 shrink-0">→</span><span><span className="text-white font-medium">Use Compare page</span> for deeper analysis — runs multiple queries, calculates P50/P95 latency, shows Jaccard overlap matrix, and graph A/B Δ scores.</span></li>
              </ul>
            </div>
          </section>

          {/* Tip */}
          <section className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">💡 Tips</h3>
            <ul className="space-y-1 text-xs text-gray-400 list-none">
              <li>• A/B Mode only shows the <span className="text-gray-200">first 2 selected backends</span> — if you have 3 selected, the 3rd is hidden in A/B Mode.</li>
              <li>• Combine with <span className="text-gray-200">retrieval method toggles</span> (Dense only vs Dense+BM25+SPLADE) to isolate which signal improves results on a specific backend.</li>
              <li>• For production-level benchmarking use the <span className="text-sky-400">Compare page</span> — it runs queries in parallel, stores results, and shows run history diffs.</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  )
}


export default function SearchLab() {
  const { selectedBackends, activeCollection, retrievalMethods } = useStore()
  const [query, setQuery] = useState('')
  const [topK, setTopK] = useState<number>(() => {
    try { return parseInt(localStorage.getItem('polyrag_top_k') ?? '10', 10) } catch { return 10 }
  })
  const [history, setHistory] = useState<string[]>(() => {
    try { return JSON.parse(localStorage.getItem('polyrag_search_history') ?? '[]') } catch { return [] }
  })
  const [results, setResults] = useState<SearchResponse[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [abMode, setAbMode] = useState(false)
  const [showAbGuide, setShowAbGuide] = useState(false)
  const [showSearchGuide, setShowSearchGuide] = useState(false)
  // Incremented after each successful search so TrailsPanel auto-reloads.
  const [trailSearchCount, setTrailSearchCount] = useState(0)

  useEffect(() => {
    localStorage.setItem('polyrag_search_history', JSON.stringify(history))
  }, [history])

  useEffect(() => {
    localStorage.setItem('polyrag_top_k', String(topK))
  }, [topK])

  const handleSearch = async () => {
    if (!query.trim()) return
    setLoading(true)
    setError(null)
    try {
      const data = await search({
        query,
        backends: selectedBackends,
        collection_name: activeCollection,
        methods: retrievalMethods,
        top_k: topK,
      })
      setResults(data)
      setHistory((prev) => [query, ...prev.filter((q) => q !== query)].slice(0, 20))
      setTrailSearchCount((c) => c + 1)
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex gap-4 h-full">
      {/* Sidebar */}
      <div className="w-56 shrink-0 space-y-4">
        <div className="bg-gray-900 rounded-lg p-4">
          <BackendSelector />
        </div>
        <CollectionPicker />
        <div className="bg-gray-900 rounded-lg p-4">
          <MethodToggle />
        </div>
        {/* Top-K control */}
        <div className="bg-gray-900 rounded-lg p-4">
          <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Results per backend</div>
          <div className="flex items-center gap-2">
            <input
              type="range"
              min={3} max={30} step={1}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="flex-1 accent-sky-500"
            />
            <span className="text-sm text-gray-200 tabular-nums w-6 text-right">{topK}</span>
          </div>
          <div className="text-[10px] text-gray-500 mt-1">Chunks returned (top_k). Each retrieval method fetches up to {topK * 5} candidates internally before RRF fusion.</div>
        </div>
        <div className="flex items-center gap-2 px-4">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer">
            <input
              type="checkbox"
              checked={abMode}
              onChange={(e) => setAbMode(e.target.checked)}
              className="rounded border-gray-600 bg-gray-800"
            />
            A/B Mode
          </label>
          <button
            onClick={() => setShowAbGuide(true)}
            title="What is A/B Mode?"
            className="w-5 h-5 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-300 hover:text-white text-xs font-bold flex items-center justify-center transition-colors shrink-0"
          >
            ?
          </button>
        </div>
      </div>

      {/* Main */}
      <div className="flex-1 flex flex-col gap-4 overflow-hidden">
        {/* Search bar */}
        <div className="flex gap-2">
          <div className="flex-1 flex gap-2">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Enter your query..."
              className="flex-1 bg-gray-800 text-gray-200 rounded border border-gray-700 px-3 py-2 text-sm focus:outline-none focus:border-brand-500"
            />
            <select
              value=""
              onChange={(e) => { if (e.target.value) setQuery(e.target.value) }}
              className="bg-gray-800 text-gray-400 rounded border border-gray-700 px-2 text-sm focus:outline-none"
            >
              <option value="">History</option>
              {history.map((q, i) => (
                <option key={i} value={q}>{q.slice(0, 40)}</option>
              ))}
            </select>
          </div>
          <button
            onClick={handleSearch}
            disabled={loading || !query.trim()}
            className="px-4 py-2 bg-brand-500 hover:bg-sky-600 disabled:opacity-50 text-white rounded text-sm font-medium"
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
          <button
            onClick={() => setShowSearchGuide(true)}
            title="How does Search Lab work?"
            className="w-9 h-9 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-300 hover:text-white text-sm font-bold flex items-center justify-center transition-colors shrink-0"
          >
            ?
          </button>
        </div>

        {error && (
          <div className="bg-red-900/30 border border-red-700 rounded p-3 text-sm text-red-300">{error}</div>
        )}

        {/* Results */}
        <div className="flex-1 overflow-y-auto">
          {results.length === 0 && !loading && (
            <div className="text-gray-600 text-center mt-16">Run a search to see results</div>
          )}
          {abMode ? (
            <div className="grid grid-cols-2 gap-4">
              {results.slice(0, 2).map((res) => (
                <div key={res.backend} className="space-y-3">
                  <div className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-gray-300">{res.backend}</span>
                    {res.latency_ms && <span className="text-xs text-gray-600">{res.latency_ms.toFixed(0)}ms</span>}
                    {res.error && <span className="text-xs text-red-400">Error: {res.error}</span>}
                  </div>
                  {res.answer && (
                    <div className="bg-gray-800 rounded p-3 text-sm text-gray-200 border-l-2 border-sky-500">
                      {res.answer}
                    </div>
                  )}
                  {res.results.map((r, i) => <ResultCard key={r.chunk_id || i} result={r} backend={res.backend} />)}
                  {res.results.length === 0 && !res.error && (
                    <div className="text-xs text-gray-600 py-4 text-center">No results from {res.backend}</div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${Math.max(1, results.length)}, minmax(300px, 1fr))` }}>
              {results.map((res) => (
                <div key={res.backend} className="space-y-3 min-w-0">
                  <div className="flex items-center gap-2 sticky top-0 bg-gray-950/80 py-1 z-10">
                    <span className="text-sm font-semibold text-gray-300">{res.backend}</span>
                    {res.latency_ms && <span className="text-xs text-gray-600">{res.latency_ms.toFixed(0)}ms</span>}
                    {res.error && <span className="text-xs text-red-400 truncate">⚠ {res.error}</span>}
                    <span className="ml-auto text-xs text-gray-600">{res.results.length} result(s)</span>
                  </div>
                  {res.answer && (
                    <div className="bg-gray-800 rounded p-3 text-sm text-gray-200 border-l-2 border-sky-500">
                      <span className="text-xs text-sky-500 font-medium block mb-1">Answer</span>
                      {res.answer}
                    </div>
                  )}
                  {res.results.map((r, i) => <ResultCard key={r.chunk_id || i} result={r} backend={res.backend} />)}
                  {res.results.length === 0 && !res.error && (
                    <div className="text-xs text-gray-600 py-8 text-center">No results from {res.backend}</div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {results.map((res) => (
          <RetrievalTrace
            key={`trace-${res.backend}`}
            steps={res.trace ?? []}
            methodContributions={res.method_contributions}
            chunks={res.results?.map(r => ({ chunk_id: r.chunk_id, text: r.text, score: r.score, method_lineage: r.method_lineage, metadata: r.metadata }))}
            answer={res.answer}
            graphEntities={res.graph_entities}
            graphPaths={res.graph_paths}
            queryVariants={res.query_variants}
            label={`Method Traceability — ${res.backend}`}
          />
        ))}

        {/* Persistent Retrieval Trails (always visible, auto-reloads after each search) */}
        <TrailsPanel searchCount={trailSearchCount} activeBackends={selectedBackends} />

        {/* LLM Trace Panels — one per backend result that has LLM calls */}
        {results.map((res) =>
          res.llm_traces && res.llm_traces.length > 0 ? (
            <div key={`trace-${res.backend}`}>
              <div className="text-xs text-gray-500 mb-1 mt-2">{res.backend} — LLM Calls</div>
              <LLMTracePanel traces={res.llm_traces} />
            </div>
          ) : null
        )}
      </div>
      {showAbGuide && <ABModeGuideModal onClose={() => setShowAbGuide(false)} />}
      {showSearchGuide && <SearchGuideModal onClose={() => setShowSearchGuide(false)} />}
    </div>
  )
}
