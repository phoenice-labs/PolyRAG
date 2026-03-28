import { useState, useEffect, useCallback, useRef } from 'react'
import { runEvaluation, browseChunks, generateQA, type QAPair, type EvalResult, type EvalScore, type RetrievalMethodsReq, type BrowseChunk } from '../api/evaluate'
import { getCollections, type Collection } from '../api/backends'
import { useStore } from '../store'
import RetrievalTrace from '../components/RetrievalTrace/RetrievalTrace'

// ── Constants ─────────────────────────────────────────────────────────────────

const ALL_BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']

const BACKEND_COLORS: Record<string, string> = {
  faiss: 'bg-blue-500', chromadb: 'bg-purple-500', qdrant: 'bg-red-500',
  weaviate: 'bg-green-500', milvus: 'bg-yellow-500', pgvector: 'bg-cyan-500',
}

const DEFAULT_METHODS: RetrievalMethodsReq = {
  enable_dense: true, enable_bm25: true, enable_graph: true,
  enable_rerank: true, enable_mmr: true,
  enable_rewrite: false, enable_multi_query: false, enable_hyde: false,
  enable_raptor: false, enable_contextual_rerank: false, enable_llm_graph: false,
}

const METHOD_LABELS: Record<keyof RetrievalMethodsReq, { label: string; group: 'base' | 'llm' }> = {
  enable_dense:             { label: 'Dense Vector',         group: 'base' },
  enable_bm25:              { label: 'BM25 Keyword',         group: 'base' },
  enable_graph:             { label: 'Knowledge Graph',       group: 'base' },
  enable_rerank:            { label: 'Cross-Encoder Rerank', group: 'base' },
  enable_mmr:               { label: 'MMR Diversity',         group: 'base' },
  enable_rewrite:           { label: 'Query Rewrite',         group: 'llm'  },
  enable_multi_query:       { label: 'Multi-Query',           group: 'llm'  },
  enable_hyde:              { label: 'HyDE',                 group: 'llm'  },
  enable_raptor:            { label: 'RAPTOR',               group: 'llm'  },
  enable_contextual_rerank: { label: 'Contextual Rerank',    group: 'llm'  },
  enable_llm_graph:         { label: 'LLM Graph Extract',    group: 'llm'  },
}

const SAMPLE_PAIRS: Omit<QAPair, 'id'>[] = [
  { question: "What themes does Shakespeare explore in Hamlet?", expected_answer: 'revenge, mortality, corruption, madness', expected_sources: [] },
  { question: "Who is Ophelia in Hamlet?", expected_answer: "Ophelia is Polonius's daughter and Hamlet's love interest", expected_sources: [] },
  { question: "What does the ghost of Hamlet's father reveal?", expected_answer: 'He was murdered by Claudius who poured poison in his ear', expected_sources: [] },
]

let _idCtr = 0
const genId = () => `qa-${++_idCtr}-${Date.now()}`

// ── ChunkBrowser modal ────────────────────────────────────────────────────────

interface ChunkBrowserProps {
  collectionName: string
  availableBackends: string[]
  onAdd: (pair: Omit<QAPair, 'id'>) => void
  onClose: () => void
}

function ChunkBrowser({ collectionName, availableBackends, onAdd, onClose }: ChunkBrowserProps) {
  const [browser, setBrowser] = useState(availableBackends[0] ?? '')
  const [search, setSearch] = useState('')
  const [offset, setOffset] = useState(0)
  const [chunks, setChunks] = useState<BrowseChunk[]>([])
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [selected, setSelected] = useState<BrowseChunk | null>(null)
  const [generating, setGenerating] = useState(false)
  const [generated, setGenerated] = useState<{ question: string; answer: string; source: string; note?: string } | null>(null)
  const [addedIds, setAddedIds] = useState<Set<string>>(new Set())
  const searchTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const PAGE = 30

  const load = useCallback(async (bk: string, q: string, off: number) => {
    if (!bk || !collectionName) return
    setLoading(true); setErr(null)
    try {
      const res = await browseChunks(bk, collectionName, { limit: PAGE, offset: off, search: q })
      setChunks(res.chunks); setTotal(res.total)
    } catch (e) {
      setErr(String(e)); setChunks([])
    } finally {
      setLoading(false)
    }
  }, [collectionName])

  useEffect(() => { load(browser, '', 0) }, [])   // eslint-disable-line react-hooks/exhaustive-deps

  const handleSearch = (q: string) => {
    setSearch(q); setOffset(0)
    if (searchTimer.current) clearTimeout(searchTimer.current)
    searchTimer.current = setTimeout(() => load(browser, q, 0), 400)
  }

  const handleBackend = (bk: string) => {
    setBrowser(bk); setOffset(0); setChunks([]); setSelected(null); setGenerated(null)
    load(bk, search, 0)
  }

  const handleSelect = (chunk: BrowseChunk) => {
    if (selected?.id === chunk.id) { setSelected(null); setGenerated(null); return }
    setSelected(chunk); setGenerated(null)
  }

  const handleGenerate = async () => {
    if (!selected) return
    setGenerating(true)
    try {
      const res = await generateQA(selected.text, selected.id)
      setGenerated({ question: res.question, answer: res.answer, source: res.source, note: res.note })
    } catch (e) {
      setGenerated({ question: '', answer: '', source: 'error', note: String(e) })
    } finally {
      setGenerating(false)
    }
  }

  const handleAdd = () => {
    if (!selected || !generated?.question) return
    onAdd({
      question: generated.question,
      expected_answer: generated.answer,
      expected_sources: [selected.id],
    })
    setAddedIds((prev) => new Set(prev).add(selected.id))
    // advance to next chunk
    const idx = chunks.findIndex((c) => c.id === selected.id)
    const next = chunks[idx + 1]
    setSelected(next ?? null); setGenerated(null)
  }

  const pages = Math.ceil(total / PAGE)
  const page = Math.floor(offset / PAGE)

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" onClick={onClose} />

      <div className="relative w-full max-w-5xl max-h-[90vh] bg-gray-900 rounded-xl border border-gray-700 shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gray-700 flex-shrink-0">
          <div>
            <h2 className="text-sm font-semibold text-white">Browse Collection: <span className="text-sky-400 font-mono">{collectionName}</span></h2>
            <p className="text-xs text-gray-400 mt-0.5">Click a chunk to read it → Generate Q&amp;A → Add to Evaluation.</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 text-xl leading-none ml-4">✕</button>
        </div>

        {/* Toolbar */}
        <div className="flex items-center gap-3 px-5 py-2.5 border-b border-gray-800 flex-shrink-0">
          <div className="flex items-center gap-1.5 flex-shrink-0">
            <span className="text-xs text-gray-500">Source:</span>
            <select value={browser} onChange={(e) => handleBackend(e.target.value)}
              className="text-xs bg-gray-800 text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-sky-500">
              {availableBackends.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </div>
          <div className="flex-1 relative">
            <span className="absolute left-2.5 top-1/2 -translate-y-1/2 text-gray-500 text-xs pointer-events-none">🔍</span>
            <input value={search} onChange={(e) => handleSearch(e.target.value)}
              placeholder="Filter chunks by keyword…"
              className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 pl-7 pr-8 py-1.5 focus:outline-none focus:border-sky-500" />
            {search && <button onClick={() => handleSearch('')} className="absolute right-2.5 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 text-xs">✕</button>}
          </div>
          <span className="text-xs text-gray-500 flex-shrink-0">
            {loading ? '⟳ Loading…' : `${total.toLocaleString()} chunk${total !== 1 ? 's' : ''}`}
          </span>
        </div>

        {/* Two-panel body */}
        <div className="flex flex-1 overflow-hidden min-h-0">

          {/* Left: chunk list */}
          <div className="w-80 flex-shrink-0 border-r border-gray-800 overflow-y-auto flex flex-col">
            {err && <div className="m-3 text-xs text-red-400 bg-red-900/20 border border-red-800 rounded p-2">{err}</div>}
            {!loading && chunks.length === 0 && !err && (
              <div className="text-center py-12 text-gray-500 text-sm px-4">
                No chunks found{search ? ` for "${search}"` : ''}.
              </div>
            )}
            {loading && chunks.length === 0 && (
              <div className="text-center py-12 text-gray-600 text-xs animate-pulse">Fetching chunks…</div>
            )}

            <div className="flex-1">
              {chunks.map((chunk, i) => {
                const isSelected = selected?.id === chunk.id
                const isAdded = addedIds.has(chunk.id)
                return (
                  <button
                    key={chunk.id}
                    onClick={() => handleSelect(chunk)}
                    className={`w-full text-left px-4 py-3 border-b border-gray-800 transition-colors ${
                      isSelected ? 'bg-sky-900/30 border-l-2 border-l-sky-500' : 'hover:bg-gray-800/50'
                    }`}
                  >
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="text-xs text-gray-600 font-mono w-6 flex-shrink-0">{offset + i + 1}.</span>
                      {isAdded && <span className="text-xs text-green-500 flex-shrink-0">✓</span>}
                      <span className="font-mono text-xs text-gray-500 truncate">{String(chunk.id).slice(0, 20)}…</span>
                    </div>
                    {(chunk.metadata?.section_title || chunk.metadata?.doc_id) && (
                      <div className="flex gap-1.5 mb-1">
                        {chunk.metadata?.section_title && <span className="text-xs text-purple-400 truncate">§ {String(chunk.metadata.section_title)}</span>}
                        {!chunk.metadata?.section_title && chunk.metadata?.doc_id && <span className="text-xs text-sky-400 truncate">📄 {String(chunk.metadata.doc_id)}</span>}
                      </div>
                    )}
                    <p className={`text-xs leading-relaxed line-clamp-2 ${isSelected ? 'text-gray-200' : 'text-gray-400'}`}>{chunk.preview}</p>
                  </button>
                )
              })}
            </div>

            {/* Pagination */}
            {total > PAGE && (
              <div className="flex items-center justify-between px-4 py-2.5 border-t border-gray-800 flex-shrink-0 bg-gray-900">
                <button onClick={() => { const o = Math.max(0, offset - PAGE); setOffset(o); load(browser, search, o) }}
                  disabled={page === 0}
                  className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-gray-300 rounded border border-gray-700">
                  ← Prev
                </button>
                <span className="text-xs text-gray-500">{page + 1} / {pages}</span>
                <button onClick={() => { const o = offset + PAGE; setOffset(o); load(browser, search, o) }}
                  disabled={page >= pages - 1}
                  className="text-xs px-2.5 py-1 bg-gray-800 hover:bg-gray-700 disabled:opacity-40 text-gray-300 rounded border border-gray-700">
                  Next →
                </button>
              </div>
            )}
          </div>

          {/* Right: selected chunk + Q&A panel */}
          <div className="flex-1 overflow-y-auto flex flex-col min-w-0">
            {!selected ? (
              <div className="flex items-center justify-center h-full text-gray-600 text-sm">
                ← Click a chunk to view its full text
              </div>
            ) : (
              <div className="p-5 space-y-4 flex flex-col h-full">
                {/* Chunk metadata */}
                <div className="flex flex-wrap gap-2 text-xs">
                  <span className="bg-gray-800 text-gray-400 rounded px-2 py-0.5 font-mono">{selected.id}</span>
                  {selected.metadata?.doc_id && <span className="bg-sky-900/40 text-sky-400 rounded px-2 py-0.5">📄 {String(selected.metadata.doc_id)}</span>}
                  {selected.metadata?.section_title && <span className="bg-purple-900/40 text-purple-400 rounded px-2 py-0.5">§ {String(selected.metadata.section_title)}</span>}
                  {selected.metadata?.chunk_type && <span className="bg-gray-800 text-gray-500 rounded px-2 py-0.5">{String(selected.metadata.chunk_type)}</span>}
                  {selected.metadata?.token_count && <span className="bg-gray-800 text-gray-500 rounded px-2 py-0.5">{String(selected.metadata.token_count)} tokens</span>}
                </div>

                {/* Full chunk text */}
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 flex-1">
                  <div className="text-xs font-medium text-gray-500 mb-2 uppercase tracking-wider">Chunk Text</div>
                  <p className="text-sm text-gray-300 leading-relaxed whitespace-pre-wrap">{selected.text}</p>
                </div>

                {/* Generate Q&A section */}
                <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-4 space-y-3 flex-shrink-0">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-medium text-gray-400">Generate Evaluation Q&amp;A from this chunk</div>
                    <button onClick={handleGenerate} disabled={generating}
                      className="text-xs px-3 py-1.5 bg-purple-700 hover:bg-purple-600 disabled:opacity-50 text-white rounded font-medium transition-colors whitespace-nowrap">
                      {generating ? '⟳ Generating…' : generated ? '↺ Regenerate' : '🧠 Generate Q&A'}
                    </button>
                  </div>

                  {generated && (
                    <div className={`space-y-2 ${generated.source === 'error' ? '' : ''}`}>
                      {generated.source === 'error' ? (
                        <p className="text-xs text-red-400 bg-red-900/20 rounded p-2">{generated.note}</p>
                      ) : (
                        <>
                          {generated.source === 'heuristic' && (
                            <p className="text-xs text-yellow-500">⚠ LM Studio unavailable — heuristic Q&amp;A generated (editable after adding)</p>
                          )}
                          <div className="bg-gray-900 rounded border border-gray-700 p-3 space-y-2">
                            <div>
                              <span className="text-xs font-medium text-sky-400 block mb-0.5">Question</span>
                              <p className="text-sm text-gray-200">{generated.question}</p>
                            </div>
                            <div>
                              <span className="text-xs font-medium text-green-400 block mb-0.5">Expected Answer</span>
                              <p className="text-sm text-gray-300">{generated.answer}</p>
                            </div>
                            <div>
                              <span className="text-xs font-medium text-gray-500 block mb-0.5">Expected Source</span>
                              <p className="text-xs text-gray-500 font-mono">{selected.id}</p>
                            </div>
                          </div>
                          <button onClick={handleAdd}
                            className="w-full text-xs py-2 bg-sky-700 hover:bg-sky-600 text-white rounded font-medium transition-colors">
                            ✓ Add to Evaluation &amp; continue →
                          </button>
                        </>
                      )}
                    </div>
                  )}

                  {!generated && !generating && (
                    <p className="text-xs text-gray-600">Click <strong>Generate Q&amp;A</strong> to create a question and answer from this chunk using the LLM.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


function ScoreBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100)
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-500 w-6">{label}</span>
      <div className="flex-1 bg-gray-700 rounded-full h-2">
        <div className={`h-2 rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className={`text-xs w-8 text-right ${pct >= 70 ? 'text-green-400' : pct >= 40 ? 'text-yellow-400' : 'text-red-400'}`}>{pct}%</span>
    </div>
  )
}

function ScoreCell({ score }: { score: EvalScore }) {
  const hasGraph = score.graph_source_hit != null
  const dims = hasGraph ? 4 : 3
  const avg = (score.faithfulness + score.relevance + score.source_hit + (score.graph_source_hit ?? 0)) / dims
  const pct = Math.round(avg * 100)
  const color = pct >= 70 ? 'text-green-400' : pct >= 40 ? 'text-yellow-400' : 'text-red-400'
  return (
    <div className="space-y-0.5 min-w-[80px]">
      <div className={`text-xs font-semibold ${color}`}>{pct}% avg</div>
      <div className="text-xs text-gray-500">F:{Math.round(score.faithfulness * 100)}%</div>
      <div className="text-xs text-gray-500">R:{Math.round(score.relevance * 100)}%</div>
      <div className="text-xs text-gray-500">S:{Math.round(score.source_hit * 100)}%</div>
      {hasGraph && (
        <div className={`text-xs ${(score.graph_source_hit! * 100) >= 50 ? 'text-indigo-400' : 'text-gray-500'}`}
          title="Graph source hit: Knowledge Graph surfaced the expected source">
          G:{Math.round(score.graph_source_hit! * 100)}%
        </div>
      )}
    </div>
  )
}

// ── Evaluate Guide Modal ──────────────────────────────────────────────────────

function EvaluateGuideModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[620px] max-h-[88vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg">🧪</span>
            <h2 className="text-base font-semibold text-white">Evaluation Studio — Guide</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* Purpose */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What is Evaluation Studio?</h3>
            <p className="text-gray-300 leading-relaxed">
              Evaluation Studio lets you <span className="text-white font-medium">measure retrieval quality objectively</span> using
              ground-truth Question–Answer pairs. You define what the correct answer should be, run it through one or more
              backends, and the system scores each result for faithfulness, relevance, and source accuracy.
            </p>
          </section>

          {/* Why it matters */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Why Run Evaluations?</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                { icon: '✅', title: 'Validate Retrieval', desc: 'Confirm that your RAG pipeline actually retrieves the right chunks for known questions.' },
                { icon: '🔄', title: 'Regression Testing', desc: 'Re-run after changing chunk size, strategy, or retrieval methods to ensure quality doesn\'t drop.' },
                { icon: '⚖️', title: 'Backend Comparison', desc: 'Compare faithfulness scores across ChromaDB, FAISS, Qdrant, etc. with identical Q&A pairs.' },
                { icon: '📐', title: 'Tune Parameters', desc: 'Find the optimal top_k, chunking strategy, and retrieval method combination for your corpus.' },
              ].map(({ icon, title, desc }) => (
                <div key={title} className="bg-gray-800 rounded p-2.5 border border-gray-700">
                  <div className="text-white font-medium mb-1">{icon} {title}</div>
                  <div className="text-gray-400 leading-relaxed">{desc}</div>
                </div>
              ))}
            </div>
          </section>

          {/* 3-step workflow */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">3-Step Workflow</h3>
            <ol className="space-y-3 text-gray-300 leading-relaxed">
              {[
                {
                  n: '1', title: 'Select Collection & Backends',
                  text: 'Choose the ingested collection to evaluate against. Select one or more backends — only backends that have the collection loaded (shown with ✓) will return results.',
                },
                {
                  n: '2', title: 'Choose Retrieval Methods',
                  text: 'Toggle which retrieval methods to use (Dense, BM25, Graph, Rerank, MMR, and optional LLM-powered methods). More methods = higher quality but slower.',
                },
                {
                  n: '3', title: 'Build Ground-Truth Q&A Pairs',
                  text: 'Add question–answer pairs manually, paste a JSON dataset, or use the Chunk Browser to auto-generate Q&A from your actual corpus chunks using LLM.',
                },
              ].map(({ n, title, text }) => (
                <li key={n} className="flex gap-3">
                  <span className="w-6 h-6 rounded-full bg-brand-500 text-white text-xs flex items-center justify-center shrink-0 mt-0.5 font-bold">{n}</span>
                  <div>
                    <div className="text-white font-medium mb-0.5">{title}</div>
                    <div className="text-gray-400 text-xs leading-relaxed">{text}</div>
                  </div>
                </li>
              ))}
            </ol>
          </section>

          {/* Score metrics */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Understanding the Scores</h3>
            <div className="space-y-1.5 text-xs">
              {[
                { metric: 'Faithfulness', desc: 'Does the retrieved answer faithfully reflect the expected answer? Measured by keyword overlap. 1.0 = perfect match.' },
                { metric: 'Relevance', desc: 'Does the retrieved content relate to the question? Scored on semantic similarity between question and chunks.' },
                { metric: 'Source Accuracy', desc: 'Did the correct source documents appear in the retrieved chunks? Only scored when expected_sources are specified.' },
                { metric: 'Overall / Verdict', desc: 'Aggregated score: PASS (≥0.7), PARTIAL (≥0.4), FAIL (<0.4). Colour-coded green / yellow / red.' },
              ].map(({ metric, desc }) => (
                <div key={metric} className="flex gap-2">
                  <span className="text-sky-400 font-mono shrink-0 w-32">{metric}</span>
                  <span className="text-gray-400">{desc}</span>
                </div>
              ))}
            </div>
          </section>

          {/* Chunk Browser */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Chunk Browser — Auto-Generate Q&A</h3>
            <div className="bg-gray-800 rounded p-3 border border-gray-700 text-xs text-gray-300 space-y-1.5">
              <p>The <span className="text-white font-medium">Chunk Browser</span> lets you search through your actual ingested chunks and auto-generate Q&A pairs using an LLM:</p>
              <ol className="space-y-1 list-decimal list-inside text-gray-400">
                <li>Open the browser — browse or search chunks by keyword.</li>
                <li>Click a chunk to select and read it in full.</li>
                <li>Click <span className="text-white">"Generate Q&A"</span> — the LLM (LM Studio) produces a question and expected answer grounded in that chunk's content.</li>
                <li>Click <span className="text-white">"Add to Evaluation"</span> — the pair is added to your Q&A list ready to run.</li>
              </ol>
              <p className="text-yellow-400 mt-2">⚠ Requires LM Studio running locally at localhost:1234. Without it, generation is skipped.</p>
            </div>
          </section>

          {/* Tips */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Tips for Better Evaluations</h3>
            <ul className="space-y-1.5 text-xs text-gray-400 list-disc list-inside">
              <li>Use <span className="text-gray-200">5–20 Q&A pairs</span> — too few gives unreliable averages; too many is slow.</li>
              <li>Make <span className="text-gray-200">expected_answer</span> concise and keyword-rich (e.g., <em>"revenge, mortality, Claudius"</em>) for accurate faithfulness scoring.</li>
              <li>Add <span className="text-gray-200">expected_sources</span> (document IDs or filenames) to test source recall — leave empty to skip source scoring.</li>
              <li>Compare the same Q&A set across different <span className="text-gray-200">chunk sizes</span> (ingest twice with different settings) to find the sweet spot.</li>
              <li>Run evaluations after every significant change to your ingestion or retrieval pipeline to catch regressions early.</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function EvaluationStudio() {
  const { activeCollection } = useStore()

  // Guide modal
  const [showGuide, setShowGuide] = useState(false)

  // Step 1 — seed from Zustand activeCollection (already scoped, e.g. "polyrag_docs_minilm")
  const [collectionName, setCollectionName] = useState(activeCollection || 'polyrag_docs')
  const [collectionInput, setCollectionInput] = useState(activeCollection || 'polyrag_docs')
  const [selectedBackends, setSelectedBackends] = useState<string[]>(['milvus'])
  const [backendCollections, setBackendCollections] = useState<Record<string, Collection[]>>({})

  // Step 2
  const [methods, setMethods] = useState<RetrievalMethodsReq>({ ...DEFAULT_METHODS })

  // Step 3
  const [pairs, setPairs] = useState<QAPair[]>([])
  const [newQ, setNewQ] = useState('')
  const [newA, setNewA] = useState('')
  const [newSrc, setNewSrc] = useState('')
  const [showBrowser, setShowBrowser] = useState(false)

  // Results
  const [results, setResults] = useState<EvalResult[]>([])
  const [summary, setSummary] = useState<Record<string, EvalScore>>({})
  const [expandedRow, setExpandedRow] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [evalInfo, setEvalInfo] = useState<{ collection: string; backends: string[] } | null>(null)

  useEffect(() => {
    ALL_BACKENDS.forEach((b) => {
      getCollections(b)
        .then((cols) => setBackendCollections((prev) => ({ ...prev, [b]: cols })))
        .catch(() => {})
    })
  }, [])

  // Auto-select: if current collectionName doesn't exist in any backend, pick the first known one.
  useEffect(() => {
    if (Object.keys(backendCollections).length === 0) return
    const known = [...new Set(Object.values(backendCollections).flat().map((c) => c.name))].sort()
    if (known.length === 0) return
    if (!known.includes(collectionName) && known[0]) {
      setCollectionName(known[0])
      setCollectionInput(known[0])
    }
  }, [backendCollections]) // eslint-disable-line react-hooks/exhaustive-deps

  const allCollectionNames = [...new Set(
    Object.values(backendCollections).flat().map((c) => c.name)
  )].sort()

  const toggleBackend = (b: string) =>
    setSelectedBackends((prev) =>
      prev.includes(b) ? prev.filter((x) => x !== b) : [...prev, b]
    )

  const toggleMethod = useCallback((key: keyof RetrievalMethodsReq) => {
    setMethods((prev) => {
      const next = { ...prev, [key]: !prev[key] }
      if (key === 'enable_rewrite' && !next.enable_rewrite) next.enable_multi_query = false
      if (key === 'enable_multi_query' && next.enable_multi_query) next.enable_rewrite = true
      if (key === 'enable_graph' && !next.enable_graph) next.enable_llm_graph = false
      if (key === 'enable_llm_graph' && next.enable_llm_graph) next.enable_graph = true
      return next
    })
  }, [])

  const addPair = () => {
    if (!newQ.trim() || !newA.trim()) return
    setPairs((prev) => [...prev, {
      id: genId(),
      question: newQ.trim(),
      expected_answer: newA.trim(),
      expected_sources: newSrc.split(',').map((s) => s.trim()).filter(Boolean),
    }])
    setNewQ(''); setNewA(''); setNewSrc('')
  }

  const loadSamples = () => setPairs(SAMPLE_PAIRS.map((p) => ({ ...p, id: genId() })))
  const removePair = (id: string) => setPairs((prev) => prev.filter((p) => p.id !== id))

  const addPairFromBrowser = useCallback((pair: Omit<QAPair, 'id'>) => {
    setPairs((prev) => [...prev, { ...pair, id: genId() }])
  }, [])

  // backends that currently have this collection loaded
  const backendsWithCollection = ALL_BACKENDS.filter((b) => {
    const cols = backendCollections[b] ?? []
    return cols.some((c) => c.name === collectionName)
  })

  const canRun = pairs.length > 0 && selectedBackends.length > 0 && !!collectionName.trim()

  const handleRun = async () => {
    if (!canRun) return
    setLoading(true); setError(null); setResults([]); setSummary({}); setExpandedRow(null)
    try {
      const res = await runEvaluation({
        questions: pairs.map(({ question, expected_answer, expected_sources }) => ({ question, expected_answer, expected_sources })),
        backends: selectedBackends,
        collection_name: collectionName,
        methods,
      })
      setResults(res.results); setSummary(res.summary)
      setEvalInfo({ collection: res.collection_name, backends: res.backends })
    } catch (err) {
      setError(String(err))
    } finally {
      setLoading(false)
    }
  }

  const uniqueBackends = [...new Set(results.map((r) => r.backend))]
  const uniqueQuestions = [...new Set(results.map((r) => r.question))]

  return (
    <div className="space-y-5 pb-10">
      <div className="flex items-start gap-3">
        <div>
          <h1 className="text-xl font-semibold">Evaluation Studio</h1>
          <p className="text-sm text-gray-400">
            Test retrieval quality by running ground-truth Q&amp;A pairs against one or more backends and comparing faithfulness, relevance, and source accuracy.
          </p>
        </div>
        <button
          onClick={() => setShowGuide(true)}
          title="How does Evaluation Studio work?"
          className="w-6 h-6 mt-0.5 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-300 hover:text-white text-xs font-bold flex items-center justify-center transition-colors shrink-0"
        >
          ?
        </button>
      </div>

      {/* Step 1 */}
      <section className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 bg-gray-800/60 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-white">Step 1 — Select Collection &amp; Backends</h2>
          <p className="text-xs text-gray-400 mt-0.5">Choose which data to evaluate against. Backends showing ✓ have the collection loaded.</p>
        </div>
        <div className="p-4 space-y-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1.5 font-medium">Collection name</label>
            {allCollectionNames.length > 0 ? (
              <select
                value={collectionName}
                onChange={(e) => { setCollectionName(e.target.value); setCollectionInput(e.target.value) }}
                className="w-full max-w-xs bg-gray-800 text-sm text-gray-200 rounded border border-gray-600 px-2 py-1.5 focus:outline-none focus:border-sky-500"
              >
                {allCollectionNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            ) : (
              <input
                value={collectionInput}
                onChange={(e) => setCollectionInput(e.target.value)}
                onBlur={() => setCollectionName(collectionInput)}
                placeholder="e.g. polyrag_docs"
                className="w-full max-w-xs bg-gray-800 text-sm text-gray-200 rounded border border-gray-600 px-2 py-1.5 focus:outline-none focus:border-sky-500"
              />
            )}
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1.5 font-medium">Backends to evaluate</label>
            <div className="grid grid-cols-3 gap-2">
              {ALL_BACKENDS.map((b) => {
                const cols = backendCollections[b] ?? []
                const match = cols.find((c) => c.name === collectionName)
                const isSelected = selectedBackends.includes(b)
                return (
                  <label key={b} className={`flex items-center gap-2 px-3 py-2 rounded border cursor-pointer transition-colors ${isSelected ? 'border-sky-600 bg-sky-900/20' : 'border-gray-700 bg-gray-800/30 hover:border-gray-500'}`}>
                    <input type="checkbox" checked={isSelected} onChange={() => toggleBackend(b)} className="rounded border-gray-600" />
                    <span className={`w-2 h-2 rounded-sm flex-shrink-0 ${BACKEND_COLORS[b]}`} />
                    <span className="text-sm text-gray-300 flex-1 capitalize">{b}</span>
                    {match
                      ? <span className="text-xs text-green-400">✓ {match.chunk_count.toLocaleString()}</span>
                      : <span className="text-xs text-gray-600">no data</span>}
                  </label>
                )
              })}
            </div>
            {selectedBackends.length === 0 && <p className="text-xs text-red-400 mt-1">Select at least one backend.</p>}
          </div>
        </div>
      </section>

      {/* Step 2 */}
      <section className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 bg-gray-800/60 border-b border-gray-700">
          <h2 className="text-sm font-semibold text-white">Step 2 — Retrieval Methods</h2>
          <p className="text-xs text-gray-400 mt-0.5">LLM methods require LM Studio running on localhost:1234.</p>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-2 gap-x-8 gap-y-1">
            <div className="space-y-1">
              <p className="text-xs font-medium text-gray-500 mb-2">Base retrieval</p>
              {(Object.keys(METHOD_LABELS) as Array<keyof RetrievalMethodsReq>)
                .filter((k) => METHOD_LABELS[k].group === 'base')
                .map((key) => (
                  <label key={key} className="flex items-center gap-2 cursor-pointer group py-0.5">
                    <input type="checkbox" checked={methods[key]} onChange={() => toggleMethod(key)} className="rounded border-gray-600 bg-gray-800" />
                    <span className="text-sm text-gray-300 group-hover:text-white">{METHOD_LABELS[key].label}</span>
                  </label>
                ))}
            </div>
            <div className="space-y-1">
              <p className="text-xs font-medium text-gray-500 mb-2">LLM-assisted (requires LM Studio)</p>
              {(Object.keys(METHOD_LABELS) as Array<keyof RetrievalMethodsReq>)
                .filter((k) => METHOD_LABELS[k].group === 'llm')
                .map((key) => (
                  <label key={key} className="flex items-center gap-2 cursor-pointer group py-0.5">
                    <input type="checkbox" checked={methods[key]} onChange={() => toggleMethod(key)} className="rounded border-gray-600 bg-gray-800" />
                    <span className="text-sm text-gray-300 group-hover:text-white">{METHOD_LABELS[key].label}</span>
                  </label>
                ))}
            </div>
          </div>
        </div>
      </section>

      {/* Step 3 */}
      <section className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 bg-gray-800/60 border-b border-gray-700 flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-white">Step 3 — Ground-Truth Q&amp;A Pairs</h2>
            <p className="text-xs text-gray-400 mt-0.5">Each question is asked to the RAG system; the answer is scored against your expected answer.</p>
          </div>
          <div className="flex gap-2">
            {backendsWithCollection.length > 0 && (
              <button
                onClick={() => setShowBrowser(true)}
                className="text-xs px-2.5 py-1 bg-purple-800 hover:bg-purple-700 text-purple-200 rounded border border-purple-700 transition-colors"
              >
                📚 Browse Collection
              </button>
            )}
            <button onClick={loadSamples} className="text-xs px-2.5 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded border border-gray-600 transition-colors">
              Load sample Qs
            </button>
          </div>
        </div>
        <div className="p-4 space-y-3">
          <div className="grid grid-cols-[1fr_1fr_1fr_auto] gap-2 items-end">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Question *</label>
              <input value={newQ} onChange={(e) => setNewQ(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addPair()}
                placeholder="e.g. What does Hamlet say?"
                className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-sky-500" />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Expected answer *</label>
              <input value={newA} onChange={(e) => setNewA(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addPair()}
                placeholder="Key facts or phrases expected"
                className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-sky-500" />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Expected sources (optional, comma-sep)</label>
              <input value={newSrc} onChange={(e) => setNewSrc(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && addPair()}
                placeholder="e.g. doc_id_123, hamlet.txt"
                className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-sky-500" />
            </div>
            <button onClick={addPair} disabled={!newQ.trim() || !newA.trim()} className="px-3 py-1.5 bg-sky-700 hover:bg-sky-600 disabled:opacity-40 text-white rounded text-sm font-medium">
              + Add
            </button>
          </div>

          {pairs.length > 0 && (
            <div className="space-y-1.5">
              {pairs.map((p, i) => (
                <div key={p.id} className="flex items-start gap-2 bg-gray-800 rounded p-2.5 text-xs">
                  <span className="text-gray-600 font-mono w-4 flex-shrink-0 mt-0.5">{i + 1}.</span>
                  <div className="flex-1 min-w-0">
                    <div className="text-gray-200 font-medium truncate">{p.question}</div>
                    <div className="text-gray-400 mt-0.5 truncate">&#8594; {p.expected_answer}</div>
                    {p.expected_sources.length > 0 && <div className="text-gray-600 mt-0.5">sources: {p.expected_sources.join(', ')}</div>}
                  </div>
                  <button onClick={() => removePair(p.id)} className="text-red-500 hover:text-red-400 flex-shrink-0">&#10005;</button>
                </div>
              ))}
              <p className="text-xs text-gray-600">{pairs.length} question{pairs.length > 1 ? 's' : ''} ready</p>
            </div>
          )}
        </div>
      </section>

      {/* Run */}
      <div className="flex items-center gap-4">
        <button onClick={handleRun} disabled={loading || !canRun}
          className="px-5 py-2.5 bg-sky-700 hover:bg-sky-600 disabled:opacity-50 text-white rounded text-sm font-semibold transition-colors">
          {loading ? '⟳ Running Evaluation…' : '▶ Run Evaluation'}
        </button>
        {!canRun && !loading && (
          <p className="text-xs text-gray-500">
            {pairs.length === 0 ? 'Add at least one Q&A pair.' : !collectionName ? 'Enter a collection name.' : 'Select at least one backend.'}
          </p>
        )}
        {loading && <p className="text-xs text-gray-400 animate-pulse">Querying {selectedBackends.length} backend{selectedBackends.length > 1 ? 's' : ''} × {pairs.length} question{pairs.length > 1 ? 's' : ''}…</p>}
      </div>

      {error && <div className="bg-red-900/30 border border-red-700 rounded p-3 text-sm text-red-300">{error}</div>}

      {/* Results */}
      {results.length > 0 && evalInfo && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-gray-300">Results</h2>
            <span className="text-xs text-gray-500">
              Collection: <span className="font-mono text-sky-400">{evalInfo.collection}</span>
              {' · '}{uniqueBackends.length} backend{uniqueBackends.length > 1 ? 's' : ''}
              {' · '}{uniqueQuestions.length} question{uniqueQuestions.length > 1 ? 's' : ''}
            </span>
          </div>

          <div className="flex gap-4 text-xs text-gray-500 bg-gray-900/50 px-3 py-2 rounded border border-gray-700/50">
            <span><span className="text-gray-300 font-medium">F</span> = Faithfulness — expected answer words found in the generated answer</span>
            <span><span className="text-gray-300 font-medium">R</span> = Relevance — how much the answer addresses the question</span>
            <span><span className="text-gray-300 font-medium">S</span> = Source Hit — expected source doc found in retrieved chunks</span>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-700 bg-gray-800/50">
                    <th className="px-4 py-2.5 text-left text-gray-400 font-medium w-64">Question</th>
                    {uniqueBackends.map((b) => <th key={b} className="px-4 py-2.5 text-left text-gray-400 font-medium capitalize">{b}</th>)}
                    <th className="px-4 py-2.5 w-8" />
                  </tr>
                </thead>
                <tbody>
                  {uniqueQuestions.map((q, qi) => {
                    const rowKey = `q-${qi}`
                    const expanded = expandedRow === rowKey
                    return (
                      <>
                        <tr key={rowKey} className="border-b border-gray-800 hover:bg-gray-800/30 cursor-pointer transition-colors" onClick={() => setExpandedRow(expanded ? null : rowKey)}>
                          <td className="px-4 py-3 text-xs text-gray-300 max-w-xs"><div className="truncate">{q}</div></td>
                          {uniqueBackends.map((b) => {
                            const res = results.find((r) => r.question === q && r.backend === b)
                            return (
                              <td key={b} className="px-4 py-3">
                                {res?.error ? <span className="text-xs text-red-400">error</span>
                                  : res ? <ScoreCell score={res.scores} /> : <span className="text-gray-600">—</span>}
                              </td>
                            )
                          })}
                          <td className="px-4 py-3 text-gray-600 text-xs">{expanded ? '▲' : '▼'}</td>
                        </tr>
                        {expanded && (
                          <tr key={`${rowKey}-exp`} className="bg-gray-800/20">
                            <td colSpan={uniqueBackends.length + 2} className="px-4 py-3">
                              <div className="text-xs text-gray-400 mb-2 font-medium">Question: <span className="text-gray-200">{q}</span></div>
                              <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${uniqueBackends.length}, minmax(200px, 1fr))` }}>
                                {uniqueBackends.map((b) => {
                                  const res = results.find((r) => r.question === q && r.backend === b)
                                  if (!res) return null
                                  return (
                                    <div key={b} className="bg-gray-900 rounded p-3 border border-gray-700">
                                      <div className="flex items-center gap-1.5 mb-2">
                                        <span className={`w-2 h-2 rounded-sm ${BACKEND_COLORS[b]}`} />
                                        <span className="text-xs font-medium text-gray-300 capitalize">{b}</span>
                                      </div>
                                      {res.error ? <p className="text-xs text-red-400">{res.error}</p> : (
                                        <>
                                          <p className="text-xs text-gray-400 mb-2 leading-relaxed">{res.answer || '(no answer)'}</p>
                                          <div className="space-y-1">
                                            <ScoreBar label="F" value={res.scores.faithfulness} color="bg-sky-500" />
                                            <ScoreBar label="R" value={res.scores.relevance}    color="bg-purple-500" />
                                            <ScoreBar label="S" value={res.scores.source_hit}   color="bg-green-500" />
                                            {res.scores.graph_source_hit != null && (
                                              <ScoreBar label="G" value={res.scores.graph_source_hit} color="bg-indigo-500" />
                                            )}
                                          </div>
                                          {res.method_contributions && Object.keys(res.method_contributions).length > 0 && (
                                            <div className="mt-2">
                                              <RetrievalTrace
                                                methodContributions={res.method_contributions}
                                                answer={res.answer}
                                                label="Method Traceability"
                                              />
                                            </div>
                                          )}
                                          {((res.graph_entities?.length ?? 0) > 0 || (res.graph_paths?.length ?? 0) > 0) && (
                                            <div className="mt-3 space-y-2">
                                              {(res.graph_entities?.length ?? 0) > 0 && (
                                                <div>
                                                  <p className="text-xs text-gray-500 mb-1">🕸️ Entities</p>
                                                  <div className="flex flex-wrap gap-1">
                                                    {res.graph_entities!.map((e, i) => {
                                                      const [type, ...rest] = e.split(':')
                                                      const name = rest.join(':') || type
                                                      return (
                                                        <span key={i} className="text-[10px] bg-indigo-900/40 border border-indigo-800 text-indigo-300 px-1.5 py-0.5 rounded-full">
                                                          <span className="text-indigo-500 font-mono">{type}</span>:{name}
                                                        </span>
                                                      )
                                                    })}
                                                  </div>
                                                </div>
                                              )}
                                              {(res.graph_paths?.length ?? 0) > 0 && (
                                                <div>
                                                  <p className="text-xs text-gray-500 mb-1">🔗 Paths</p>
                                                  <div className="space-y-1">
                                                    {res.graph_paths!.map((p, i) => (
                                                      <div key={i} className="text-[10px] font-mono text-emerald-400 bg-gray-800 px-2 py-1 rounded truncate" title={p}>{p}</div>
                                                    ))}
                                                  </div>
                                                </div>
                                              )}
                                            </div>
                                          )}
                                        </>
                                      )}
                                    </div>
                                  )
                                })}
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    )
                  })}
                  <tr className="border-t-2 border-gray-600 bg-gray-800/20">
                    <td className="px-4 py-3 text-xs text-gray-300 font-semibold">Overall average</td>
                    {uniqueBackends.map((b) => (
                      <td key={b} className="px-4 py-3">
                        {summary[b] ? <ScoreCell score={summary[b]} /> : <span className="text-gray-600">—</span>}
                      </td>
                    ))}
                    <td />
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="bg-gray-900 rounded-lg border border-gray-700 p-4">
            <h3 className="text-xs font-medium text-gray-400 mb-4">Overall Score by Backend</h3>
            <div className="space-y-3">
              {uniqueBackends.map((b) => {
                const s = summary[b]; if (!s) return null
                const avg = (s.faithfulness + s.relevance + s.source_hit + (s.graph_source_hit ?? 0)) / (s.graph_source_hit != null ? 4 : 3)
                return (
                  <div key={b} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <span className={`w-2 h-2 rounded-sm ${BACKEND_COLORS[b]}`} />
                        <span className="text-gray-300 capitalize">{b}</span>
                      </div>
                      <span className={avg >= 0.7 ? 'text-green-400' : avg >= 0.4 ? 'text-yellow-400' : 'text-red-400'}>{Math.round(avg * 100)}%</span>
                    </div>
                    <div className="flex gap-1 h-3">
                      <div className="bg-sky-600 rounded-sm h-full" style={{ width: `${s.faithfulness * 100}%`, minWidth: 2 }} title={`Faithfulness: ${Math.round(s.faithfulness * 100)}%`} />
                      <div className="bg-purple-600 rounded-sm h-full" style={{ width: `${s.relevance * 100}%`, minWidth: 2 }} title={`Relevance: ${Math.round(s.relevance * 100)}%`} />
                      <div className="bg-green-600 rounded-sm h-full" style={{ width: `${s.source_hit * 100}%`, minWidth: 2 }} title={`Source Hit: ${Math.round(s.source_hit * 100)}%`} />
                      {s.graph_source_hit != null && (
                        <div className="bg-indigo-600 rounded-sm h-full" style={{ width: `${s.graph_source_hit * 100}%`, minWidth: 2 }} title={`Graph Source Hit: ${Math.round(s.graph_source_hit * 100)}%`} />
                      )}
                    </div>
                  </div>
                )
              })}
            </div>
            <div className="flex gap-4 mt-3 text-xs text-gray-600">
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-sky-600 rounded-sm" /> Faithfulness (F)</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-purple-600 rounded-sm" /> Relevance (R)</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-green-600 rounded-sm" /> Source Hit (S)</span>
              <span className="flex items-center gap-1"><span className="w-2 h-2 bg-indigo-600 rounded-sm" /> Graph Hit (G)</span>
            </div>
          </div>
        </div>
      )}

      {showBrowser && (
        <ChunkBrowser
          collectionName={collectionName}
          availableBackends={backendsWithCollection}
          onAdd={(pair) => addPairFromBrowser(pair)}
          onClose={() => setShowBrowser(false)}
        />
      )}
      {showGuide && <EvaluateGuideModal onClose={() => setShowGuide(false)} />}
    </div>
  )
}