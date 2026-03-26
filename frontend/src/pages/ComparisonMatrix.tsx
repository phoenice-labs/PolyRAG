/**
 * ComparisonMatrix — compare retrieval quality across vector database backends.
 *
 * Features:
 *  1. Chunk Preview Panel  — click a per-query row to see retrieved chunks
 *  2. Bar Chart toggle     — visualise latency & score as horizontal bars
 *  3. Overlap Matrix       — % of shared chunk IDs between backends per query
 *  4. Winner Badge         — auto-detected fastest / highest-score / most-results
 *  5. Run History / Diff   — localStorage run history, diff two runs side-by-side
 *  6. P95 Latency          — repeat_runs toggle for P50/P95 percentile latency
 */
import { useState, useEffect, useCallback, useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { useStore } from '../store'
import {
  startComparison, getSampleQueries,
  type CompareRow, type ComparePerQueryRow, type CompareChunkPreview,
} from '../api/compare'
import { getCollections, type Collection } from '../api/backends'
import RetrievalTrace from '../components/RetrievalTrace/RetrievalTrace'

// ── Backend metadata ────────────────────────────────────────────────────────
const ALL_BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']
const BACKEND_COLORS: Record<string, string> = {
  faiss: 'bg-blue-500', chromadb: 'bg-purple-500', qdrant: 'bg-red-500',
  weaviate: 'bg-green-500', milvus: 'bg-yellow-500', pgvector: 'bg-indigo-500',
}
const BACKEND_TEXT: Record<string, string> = {
  faiss: 'text-blue-400', chromadb: 'text-purple-400', qdrant: 'text-red-400',
  weaviate: 'text-green-400', milvus: 'text-yellow-400', pgvector: 'text-indigo-400',
}

const HISTORY_KEY = 'polyrag_compare_history'
const MAX_HISTORY  = 10

// ── Table column definitions ────────────────────────────────────────────────
const SUMMARY_COLS = [
  { key: 'backend',               label: 'Backend',            tip: 'Vector database backend' },
  { key: 'avg_query_latency_ms',  label: 'Avg Latency',        tip: 'Average query P50 latency — lower is faster (primary differentiator)' },
  { key: 'latency_p95_ms',        label: 'P95 Latency',        tip: 'P95 query latency (visible only when Repeat Runs > 1)' },
  { key: 'base_top_score',        label: 'Top Score',          tip: 'Highest RRF score. Scores are similar across backends (same embedding model); latency is the key differentiator.' },
  { key: 'full_top_score',        label: 'Full Score',         tip: 'Highest score with LLM methods ON (Full Retrieval only)' },
  { key: 'base_kw_hits',          label: 'KW Hits',            tip: 'Chunks containing query keywords' },
  { key: 'avg_score',             label: 'Avg Score',          tip: 'Average RRF score across queries' },
  { key: 'total_result_count',    label: 'Results',            tip: 'Total chunks returned across all queries' },
  { key: 'ingest_time_s',         label: 'Ingest (s)',         tip: 'Ingestion time (0 = existing collection)' },
  { key: 'errors',                label: 'Errors',             tip: 'Failed queries' },
] as const

type SortKey = typeof SUMMARY_COLS[number]['key']

// ── Saved run (localStorage) ────────────────────────────────────────────────
interface SavedRun {
  id: string
  label: string
  ts: number
  summary: CompareRow[]
  per_query: ComparePerQueryRow[]
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function scoreColor(key: string, value: number | string | undefined): string {
  if (typeof value !== 'number') return ''
  if (key === 'errors') return value > 0 ? 'text-red-400' : 'text-green-400'
  if (key === 'ingest_time_s') return value < 5 ? 'text-green-400' : value < 15 ? 'text-yellow-400' : 'text-red-400'
  if (key.includes('latency')) return value < 500 ? 'text-green-400' : value < 2000 ? 'text-yellow-400' : 'text-red-400'
  if (key.includes('kw_hits') || key === 'base_kw_hits') return value > 0 ? 'text-green-400' : 'text-gray-500'
  if (key.includes('result_count')) return value > 0 ? 'text-green-400' : 'text-red-400'
  if (key !== 'backend') return value >= 0.7 ? 'text-green-400' : value >= 0.3 ? 'text-yellow-400' : 'text-red-400'
  return ''
}

function fmtLatency(ms: number) { return ms > 0 ? `${ms.toFixed(0)} ms` : '—' }
function fmtScore(v: number)    { return v > 0 ? v.toFixed(4) : '—' }

function download(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href = url; a.download = filename; a.click()
  URL.revokeObjectURL(url)
}

function loadHistory(): SavedRun[] {
  try { return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]') } catch { return [] }
}
function saveHistory(runs: SavedRun[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(runs.slice(0, MAX_HISTORY)))
}

// ── Sub-components ──────────────────────────────────────────────────────────

function Section({ title, children, action }: { title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-gray-200">{title}</h2>
        {action}
      </div>
      {children}
    </div>
  )
}

function ModeButton({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick}
      className={`px-4 py-2 rounded text-sm font-medium transition-colors ${active ? 'bg-sky-600 text-white' : 'bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700'}`}>
      {children}
    </button>
  )
}

function Tooltip({ text }: { text: string }) {
  return <span className="ml-1 text-gray-600 cursor-help" title={text}>ⓘ</span>
}

function BackendAvailability({ backend, selected, collections, collectionName, onToggle }: {
  backend: string; selected: boolean; collections: Record<string, Collection[]>
  collectionName: string; onToggle: (b: string) => void
}) {
  const colList = collections[backend] ?? []
  const match = collectionName ? colList.find(c => c.name === collectionName) : undefined
  const status = collectionName
    ? (match ? `✓ ${match.chunk_count} chunks` : '✗ not found')
    : `${colList.length} collection(s)`
  const statusColor = collectionName ? (match ? 'text-green-400' : 'text-red-400') : 'text-gray-400'
  return (
    <label className="flex items-center gap-2 cursor-pointer group py-1">
      <input type="checkbox" checked={selected} onChange={() => onToggle(backend)}
        className="rounded border-gray-600 bg-gray-800 text-brand-500" />
      <span className={`w-2 h-2 rounded-sm flex-shrink-0 ${BACKEND_COLORS[backend]}`} />
      <span className="text-sm text-gray-300 group-hover:text-white flex-1 capitalize">{backend}</span>
      <span className={`text-xs ${statusColor}`}>{status}</span>
    </label>
  )
}

// Feature 4 — Winner Badge Card
function WinnerBadges({ rows, graphAb }: { rows: CompareRow[]; graphAb: boolean }) {
  if (rows.length === 0) return null
  const validLatency = rows.filter(r => r.avg_query_latency_ms > 0)
  const fastest  = validLatency.length ? validLatency.reduce((a, b) => a.avg_query_latency_ms < b.avg_query_latency_ms ? a : b) : null
  const topScore = rows.reduce((a, b) => a.base_top_score > b.base_top_score ? a : b)
  const mostRes  = rows.reduce((a, b) => a.total_result_count > b.total_result_count ? a : b)
  const noErrors = rows.filter(r => r.errors === 0)

  // Graph A/B: who benefited most from graph?
  const graphWinner = graphAb && rows.some(r => (r.avg_score_delta ?? 0) !== 0)
    ? rows.reduce((a, b) => (a.avg_score_delta ?? 0) > (b.avg_score_delta ?? 0) ? a : b)
    : null

  const badges: { icon: string; label: string; backend: string; detail: string }[] = []
  if (fastest)  badges.push({ icon: '⚡', label: 'Fastest', backend: fastest.backend,  detail: `${fastest.avg_query_latency_ms.toFixed(0)} ms avg` })
  if (topScore.base_top_score > 0) badges.push({ icon: '🎯', label: 'Top Score', backend: topScore.backend, detail: topScore.base_top_score.toFixed(4) })
  if (mostRes.total_result_count > 0) badges.push({ icon: '📦', label: 'Most Results', backend: mostRes.backend, detail: `${mostRes.total_result_count} total` })
  if (noErrors.length > 0 && noErrors.length < rows.length) badges.push({ icon: '✅', label: 'Zero Errors', backend: noErrors.map(r => r.backend).join(', '), detail: '' })
  if (graphWinner && (graphWinner.avg_score_delta ?? 0) > 0)
    badges.push({ icon: '🕸️', label: 'Graph Boost', backend: graphWinner.backend, detail: `+${(graphWinner.avg_score_delta! * 100).toFixed(1)}% Δ` })

  return (
    <div className="flex flex-wrap gap-3 mb-3">
      {badges.map(b => (
        <div key={b.label} className="flex items-center gap-2 bg-gray-800 border border-gray-700 rounded-lg px-3 py-2">
          <span className="text-lg">{b.icon}</span>
          <div>
            <div className="text-xs text-gray-400">{b.label}</div>
            <div className={`text-sm font-semibold capitalize ${BACKEND_TEXT[b.backend] ?? 'text-white'}`}>
              {b.backend} {b.detail && <span className="text-gray-400 font-normal text-xs">({b.detail})</span>}
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

// Feature 2 — Bar Chart
function BarChart({ rows, metric, label }: { rows: CompareRow[]; metric: keyof CompareRow; label: string }) {
  const values = rows.map(r => Number(r[metric]) || 0)
  const maxVal = Math.max(...values, 0.001)
  return (
    <div className="space-y-2">
      <p className="text-xs text-gray-400">{label}</p>
      {rows.map((row, i) => (
        <div key={row.backend} className="flex items-center gap-2">
          <span className={`w-20 text-xs text-right capitalize ${BACKEND_TEXT[row.backend] ?? 'text-gray-300'}`}>{row.backend}</span>
          <div className="flex-1 bg-gray-800 rounded h-5 relative overflow-hidden">
            <div
              className={`h-full rounded transition-all duration-500 ${BACKEND_COLORS[row.backend] ?? 'bg-gray-500'} opacity-70`}
              style={{ width: `${(values[i] / maxVal) * 100}%` }}
            />
          </div>
          <span className="w-20 text-xs text-gray-300">{metric.toString().includes('latency') ? fmtLatency(values[i]) : fmtScore(values[i])}</span>
        </div>
      ))}
    </div>
  )
}

// Feature 3 — Overlap Matrix
function OverlapMatrix({ perQuery, backends }: { perQuery: ComparePerQueryRow[]; backends: string[] }) {
  const queries = [...new Set(perQuery.map(r => r.query))]
  if (queries.length === 0 || backends.length < 2) return null

  // For each query, build {backend -> Set<chunk_id>}
  const overlapData: { q: string; pairs: { a: string; b: string; pct: number }[] }[] = queries.map(q => {
    const rows = perQuery.filter(r => r.query === q)
    const sets: Record<string, Set<string>> = {}
    rows.forEach(r => { sets[r.backend] = new Set(r.chunk_ids || []) })

    const pairs: { a: string; b: string; pct: number }[] = []
    const bs = backends.filter(b => sets[b])
    for (let i = 0; i < bs.length; i++) {
      for (let j = i + 1; j < bs.length; j++) {
        const a = sets[bs[i]], b = sets[bs[j]]
        const union = new Set([...a, ...b])
        const inter = [...a].filter(id => b.has(id))
        pairs.push({ a: bs[i], b: bs[j], pct: union.size > 0 ? Math.round((inter.length / union.size) * 100) : 0 })
      }
    }
    return { q, pairs }
  })

  return (
    <div className="space-y-3">
      <p className="text-xs text-gray-400">
        Percentage of retrieved chunk IDs shared between each pair of backends (Jaccard overlap).
        100% = identical results; 0% = completely different results.
      </p>
      {overlapData.map(({ q, pairs }) => (
        <div key={q} className="space-y-1">
          <p className="text-xs text-gray-500 truncate" title={q}>Query: {q}</p>
          <div className="flex flex-wrap gap-2">
            {pairs.map(p => (
              <span key={`${p.a}-${p.b}`}
                className={`text-xs px-2 py-0.5 rounded border ${p.pct >= 70 ? 'border-green-700 text-green-300 bg-green-900/20' : p.pct >= 30 ? 'border-yellow-700 text-yellow-300 bg-yellow-900/20' : 'border-gray-700 text-gray-400 bg-gray-800'}`}>
                <span className={`capitalize ${BACKEND_TEXT[p.a] ?? ''}`}>{p.a}</span>
                {' ↔ '}
                <span className={`capitalize ${BACKEND_TEXT[p.b] ?? ''}`}>{p.b}</span>
                {': '}{p.pct}%
              </span>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

// Feature 1 — Chunk Preview Panel (with optional graph trail)
function ChunkPreviewPanel({ chunks, backend, query, graphEntities, graphPaths, methodContributions, onClose }: {
  chunks: CompareChunkPreview[]; backend: string; query: string
  graphEntities?: string[]; graphPaths?: string[]
  methodContributions?: Record<string, { chunks_contributed?: number; contribution_pct?: number }>
  onClose: () => void
}) {
  const [tab, setTab] = useState<'chunks' | 'graph'>('chunks')
  const hasGraph = (graphEntities?.length ?? 0) + (graphPaths?.length ?? 0) > 0

  return (
    <div className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/60" onClick={onClose}>
      <div className="bg-gray-900 border border-gray-700 rounded-t-xl sm:rounded-xl w-full max-w-2xl max-h-[80vh] flex flex-col"
        onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
          <div>
            <span className={`text-sm font-semibold capitalize ${BACKEND_TEXT[backend] ?? 'text-white'}`}>{backend}</span>
            <span className="text-xs text-gray-400 ml-2">— retrieved chunks</span>
            <p className="text-xs text-gray-500 mt-0.5 truncate max-w-sm" title={query}>{query}</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-white text-lg leading-none">✕</button>
        </div>
        {hasGraph && (
          <div className="flex border-b border-gray-800">
            {(['chunks', 'graph'] as const).map(t => (
              <button key={t} onClick={() => setTab(t)}
                className={`px-4 py-2 text-xs font-medium border-b-2 transition-colors ${tab === t ? 'border-sky-500 text-sky-400' : 'border-transparent text-gray-500 hover:text-gray-300'}`}>
                {t === 'chunks' ? `📄 Chunks (${chunks.length})` : `🕸️ Graph trail (${(graphEntities?.length ?? 0) + (graphPaths?.length ?? 0)})`}
              </button>
            ))}
          </div>
        )}
        <div className="overflow-y-auto p-4 space-y-3">
          {tab === 'chunks' ? (
            <>
              {chunks.length === 0 && <p className="text-xs text-gray-500">No chunks returned.</p>}
              {chunks.map((c, i) => (
                <div key={c.chunk_id} className="bg-gray-800 rounded p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500 font-mono">#{i + 1} · {c.chunk_id.slice(0, 24)}…</span>
                    <span className={`text-xs font-mono ${scoreColor('base_top_score', c.score)}`}>{c.score.toFixed(4)}</span>
                  </div>
                  <p className="text-xs text-gray-300 leading-relaxed">{c.text}{c.text.length >= 300 ? '…' : ''}</p>
                </div>
              ))}
              {(methodContributions || chunks.some(c => c.method_lineage?.length)) && (
                <RetrievalTrace
                  methodContributions={methodContributions}
                  chunks={chunks.map(c => ({ chunk_id: c.chunk_id, text: c.text, score: c.score, method_lineage: c.method_lineage }))}
                  label={`Method Traceability — ${backend}`}
                />
              )}
            </>
          ) : (
            <div className="space-y-4">
              {(graphEntities?.length ?? 0) > 0 && (
                <div>
                  <p className="text-xs text-gray-400 mb-2 font-medium">🏷️ Entities found by Knowledge Graph</p>
                  <div className="flex flex-wrap gap-1.5">
                    {graphEntities!.map((e, i) => {
                      const [type, ...rest] = e.split(':')
                      const name = rest.join(':') || type
                      return (
                        <span key={i} className="inline-flex items-center gap-1 text-xs bg-indigo-900/40 border border-indigo-700 text-indigo-300 px-2 py-0.5 rounded-full">
                          <span className="text-indigo-500 font-mono text-[10px]">{type.toUpperCase()}</span>
                          <span>{name}</span>
                        </span>
                      )
                    })}
                  </div>
                </div>
              )}
              {(graphPaths?.length ?? 0) > 0 && (
                <div>
                  <p className="text-xs text-gray-400 mb-2 font-medium">🔗 Graph paths traversed</p>
                  <div className="space-y-2">
                    {graphPaths!.map((p, i) => (
                      <div key={i} className="bg-gray-800 rounded px-3 py-2 text-xs text-emerald-300 font-mono">
                        {p}
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {!hasGraph && <p className="text-xs text-gray-500">No graph data returned for this query.</p>}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// Feature 5 — Run History Diff
function HistoryPanel({ history, onLoad, onClear, onDiff }: {
  history: SavedRun[]; onLoad: (r: SavedRun) => void
  onClear: () => void; onDiff: (a: SavedRun, b: SavedRun) => void
}) {
  const [sel, setSel] = useState<string[]>([])
  if (history.length === 0) return <p className="text-xs text-gray-500">No saved runs yet. Runs are auto-saved after each comparison.</p>

  const toggleSel = (id: string) =>
    setSel(prev => prev.includes(id) ? prev.filter(x => x !== id) : prev.length < 2 ? [...prev, id] : prev)

  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3">
        <p className="text-xs text-gray-400 flex-1">{sel.length === 2 ? '✓ Select "Diff" to compare these two runs' : 'Select 2 runs to diff, or click a run to reload it.'}</p>
        {sel.length === 2 && (
          <button onClick={() => { const [a, b] = sel.map(id => history.find(r => r.id === id)!); onDiff(a, b) }}
            className="text-xs px-2 py-1 bg-sky-700 hover:bg-sky-600 text-white rounded">Diff</button>
        )}
        <button onClick={onClear} className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded">Clear history</button>
      </div>
      <div className="space-y-1 max-h-48 overflow-y-auto">
        {history.map(run => (
          <div key={run.id}
            className={`flex items-center gap-2 px-2 py-1.5 rounded cursor-pointer text-xs border ${sel.includes(run.id) ? 'border-sky-600 bg-sky-900/20' : 'border-gray-800 hover:bg-gray-800'}`}
            onClick={() => toggleSel(run.id)}>
            <input type="checkbox" readOnly checked={sel.includes(run.id)} className="rounded border-gray-600 bg-gray-800" />
            <span className="flex-1 text-gray-300">{run.label}</span>
            <span className="text-gray-500">{new Date(run.ts).toLocaleTimeString()}</span>
            <button onClick={e => { e.stopPropagation(); onLoad(run) }}
              className="text-sky-400 hover:text-sky-300 underline ml-1">Load</button>
          </div>
        ))}
      </div>
    </div>
  )
}

// Feature 5 — Diff view
function DiffTable({ runA, runB }: { runA: SavedRun; runB: SavedRun }) {
  const metrics: (keyof CompareRow)[] = ['avg_query_latency_ms', 'base_top_score', 'avg_score', 'total_result_count', 'errors']
  const backends = [...new Set([...runA.summary.map(r => r.backend), ...runB.summary.map(r => r.backend)])]

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="px-3 py-2 text-left text-gray-400">Backend</th>
            {metrics.map(m => (
              <th key={m} className="px-3 py-2 text-left text-gray-400" colSpan={2}>
                {m.replace(/_/g, ' ')}
                <div className="flex gap-2 text-gray-600 font-normal">
                  <span>{runA.label}</span><span>→</span><span>{runB.label}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {backends.map(b => {
            const a = runA.summary.find(r => r.backend === b)
            const bRow = runB.summary.find(r => r.backend === b)
            return (
              <tr key={b} className="border-b border-gray-800">
                <td className={`px-3 py-2 capitalize font-medium ${BACKEND_TEXT[b] ?? 'text-gray-300'}`}>{b}</td>
                {metrics.map(m => {
                  const va = a ? Number(a[m]) : null
                  const vb = bRow ? Number(bRow[m]) : null
                  const isLatencyOrError = m.includes('latency') || m === 'errors'
                  const improved = va != null && vb != null && (isLatencyOrError ? vb < va : vb > va)
                  const regressed = va != null && vb != null && (isLatencyOrError ? vb > va : vb < va)
                  return (
                    <td key={m} className="px-3 py-2" colSpan={2}>
                      <span className="text-gray-500">{va != null ? (m.includes('latency') ? fmtLatency(va) : va.toFixed(3)) : '—'}</span>
                      <span className="mx-1 text-gray-700">→</span>
                      <span className={improved ? 'text-green-400' : regressed ? 'text-red-400' : 'text-gray-300'}>
                        {vb != null ? (m.includes('latency') ? fmtLatency(vb) : vb.toFixed(3)) : '—'}
                        {improved ? ' ▲' : regressed ? ' ▼' : ''}
                      </span>
                    </td>
                  )
                })}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ── Compare Guide Modal ─────────────────────────────────────────────────────

function CompareGuideModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[620px] max-h-[88vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg">📊</span>
            <h2 className="text-base font-semibold text-white">Backend Comparison — Guide</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* Purpose */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What is Backend Comparison?</h3>
            <p className="text-gray-300 leading-relaxed">
              Backend Comparison runs the <span className="text-white font-medium">same queries simultaneously</span> across
              multiple vector database backends (FAISS, ChromaDB, Qdrant, Weaviate, Milvus, PGVector) and gives you a
              side-by-side performance report — so you can pick the right backend for your use case.
            </p>
          </section>

          {/* Objective */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Objective</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                { icon: '⚡', title: 'Latency Benchmarking', desc: 'Find the fastest backend for your hardware, collection size, and query patterns.' },
                { icon: '🎯', title: 'Score Parity Check', desc: 'Verify that RRF scores are consistent across backends (same embedding model = same scores).' },
                { icon: '🔁', title: 'P95 Stress Testing', desc: 'Use Repeat Runs > 1 to measure P50/P95 latency under repeated queries — catches cold-start spikes.' },
                { icon: '🔍', title: 'Chunk Overlap Analysis', desc: 'Check if two backends return the same chunks — high overlap means both are equally correct.' },
              ].map(({ icon, title, desc }) => (
                <div key={title} className="bg-gray-800 rounded p-2.5 border border-gray-700">
                  <div className="text-white font-medium mb-1">{icon} {title}</div>
                  <div className="text-gray-400 leading-relaxed">{desc}</div>
                </div>
              ))}
            </div>
          </section>

          {/* How to use */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">How to Run a Comparison</h3>
            <ol className="space-y-2.5 text-gray-300 leading-relaxed">
              {[
                { n: '1', text: 'Choose Data Source — use an existing ingested collection (Existing Collection) or paste raw text (Paste Text) which will be ingested on-the-fly for each backend.' },
                { n: '2', text: 'Select Backends — tick the backends you want to compare. At least 2 required. Backends not yet set up will return empty results without crashing the others.' },
                { n: '3', text: 'Enter Queries — type one query per line, or click "Load Sample Queries" for pre-built Shakespeare questions. More queries = more reliable averages.' },
                { n: '4', text: 'Configure Options — set Repeat Runs > 1 for P95 latency; enable Full Retrieval to include LLM-powered methods (requires LM Studio running locally).' },
                { n: '5', text: 'Click Run Comparison — results appear in a sortable table. Click any per-query row to see the actual chunks retrieved by each backend.' },
              ].map(({ n, text }) => (
                <li key={n} className="flex gap-3">
                  <span className="w-5 h-5 rounded-full bg-brand-500 text-white text-xs flex items-center justify-center shrink-0 mt-0.5">{n}</span>
                  <span>{text}</span>
                </li>
              ))}
            </ol>
          </section>

          {/* What the columns mean */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Reading the Results Table</h3>
            <div className="space-y-1.5 text-xs">
              {[
                { col: 'Avg Latency', desc: 'Mean query round-trip time in ms. Primary differentiator — lower is better.' },
                { col: 'P95 Latency', desc: 'Worst-case latency (95th percentile). Only shown when Repeat Runs > 1.' },
                { col: 'Top Score', desc: 'Highest RRF fusion score across all queries. Near-identical across backends when using the same embedding model.' },
                { col: 'KW Hits', desc: 'Number of returned chunks that contain query keywords. Indicates BM25 recall quality.' },
                { col: 'Results', desc: 'Total chunks returned. Low counts usually mean the collection is missing on that backend.' },
                { col: 'Ingest (s)', desc: 'Time to ingest the pasted corpus. Zero means an existing collection was used.' },
                { col: '🏆 Winner Badges', desc: 'Auto-detected: fastest backend, highest-scoring backend, most-results backend.' },
              ].map(({ col, desc }) => (
                <div key={col} className="flex gap-2">
                  <span className="text-sky-400 font-mono shrink-0 w-28">{col}</span>
                  <span className="text-gray-400">{desc}</span>
                </div>
              ))}
            </div>
          </section>

          {/* Advanced features */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Advanced Features</h3>
            <div className="space-y-2 text-xs text-gray-300">
              <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                <span className="text-white font-medium">📈 Bar Chart View</span> — Toggle to visualise any metric as horizontal bars, colour-coded per backend. Great for quick visual comparison.
              </div>
              <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                <span className="text-white font-medium">🔗 Overlap Matrix</span> — Shows % of shared chunk IDs between each pair of backends per query. High overlap (≥80%) = backends are equivalent; low overlap = retrieval quality differs.
              </div>
              <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                <span className="text-white font-medium">🕑 Run History & Diff</span> — Runs are auto-saved locally. Select 2 runs and click "Diff" to compare metric changes across different configurations or corpus sizes.
              </div>
              <div className="bg-gray-800 rounded p-2.5 border border-gray-700">
                <span className="text-white font-medium">⬇ Export</span> — Download results as CSV or JSON for external analysis, reporting, or charting in Python/Excel.
              </div>
            </div>
          </section>

          {/* Tips */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Tips</h3>
            <ul className="space-y-1.5 text-xs text-gray-400 list-disc list-inside">
              <li>Use <span className="text-gray-200">5–10 representative queries</span> for reliable averages — 1-query runs are noisy.</li>
              <li>Latency differences are only meaningful on a <span className="text-gray-200">warm collection</span> — exclude the first run or use Repeat Runs ≥ 3.</li>
              <li>If a backend shows 0 results, check it's running via Docker (<code className="bg-gray-800 px-1 rounded">docker compose up -d</code>) and the collection is ingested.</li>
              <li>FAISS and ChromaDB run in-process — always the fastest. Qdrant, Weaviate, Milvus, PGVector require Docker services.</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  )
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function ComparisonMatrix() {
  const navigate = useNavigate()
  const { activeCollection } = useStore()

  // ── Guide modal ──
  const [showGuide, setShowGuide] = useState(false)

  // ── Data source ──
  type Mode = 'existing' | 'paste'
  const [mode, setMode]                     = useState<Mode>('existing')
  const [collectionName, setCollectionName] = useState(activeCollection || 'polyrag_docs')
  const [corpusText, setCorpusText]         = useState('')

  // ── Backend selection ──
  const [selectedBackends, setSelectedBackends] = useState<string[]>(['faiss', 'chromadb'])
  const toggleBackend = (b: string) =>
    setSelectedBackends(prev => prev.includes(b) ? prev.filter(x => x !== b) : [...prev, b])

  // ── Collections per backend ──
  const [backendCollections, setBackendCollections] = useState<Record<string, Collection[]>>({})
  useEffect(() => {
    ALL_BACKENDS.forEach(backend => {
      getCollections(backend).then(cols => setBackendCollections(prev => ({ ...prev, [backend]: cols }))).catch(() => {})
    })
  }, [])
  const allCollectionNames = useMemo(() =>
    [...new Set(Object.values(backendCollections).flat().map(c => c.name))].sort(),
    [backendCollections])

  // ── Queries & options ──
  const [queriesText, setQueriesText]     = useState('')
  const [sampleQueries, setSampleQueries] = useState<string[]>([])
  const [fullRetrieval, setFullRetrieval] = useState(false)
  const [repeatRuns, setRepeatRuns]       = useState(1)
  const [compareGraphAb, setCompareGraphAb] = useState(false)
  useEffect(() => { getSampleQueries().then(setSampleQueries).catch(() => {}) }, [])
  const loadSamples = () => setQueriesText(sampleQueries.join('\n'))

  // ── Results ──
  const [summaryRows, setSummaryRows]   = useState<CompareRow[]>([])
  const [perQueryRows, setPerQueryRows] = useState<ComparePerQueryRow[]>([])
  const [loading, setLoading]           = useState(false)
  const [runError, setRunError]         = useState<string | null>(null)

  // ── View controls ──
  const [viewMode, setViewMode]           = useState<'table' | 'chart'>('table')
  const [showPerQuery, setShowPerQuery]   = useState(false)
  const [showOverlap, setShowOverlap]     = useState(false)
  const [showHistory, setShowHistory]     = useState(false)
  const [chartMetric, setChartMetric]     = useState<keyof CompareRow>('avg_query_latency_ms')

  // Feature 1 — Chunk Preview
  const [previewRow, setPreviewRow] = useState<ComparePerQueryRow | null>(null)

  // ── Sorting ──
  const [sortKey, setSortKey] = useState<SortKey>('avg_query_latency_ms')
  const [sortAsc, setSortAsc] = useState(true)
  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(key === 'errors' || key.includes('latency') || key === 'ingest_time_s') }
  }
  const sortedRows = useMemo(() => [...summaryRows].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    if (typeof av === 'string' && typeof bv === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av)
    return sortAsc ? (av as number) - (bv as number) : (bv as number) - (av as number)
  }), [summaryRows, sortKey, sortAsc])

  // ── Feature 5: Run History ──
  const [history, setHistory]   = useState<SavedRun[]>(loadHistory)
  const [diffPair, setDiffPair] = useState<[SavedRun, SavedRun] | null>(null)

  const saveRun = useCallback((summary: CompareRow[], perQuery: ComparePerQueryRow[]) => {
    const queries = [...new Set(perQuery.map(r => r.query))].slice(0, 2).join(', ')
    const backends = [...new Set(summary.map(r => r.backend))].join('+')
    const run: SavedRun = {
      id: crypto.randomUUID(),
      label: `${backends} · ${queries.slice(0, 40)}${queries.length > 40 ? '…' : ''}`,
      ts: Date.now(),
      summary,
      per_query: perQuery,
    }
    const next = [run, ...history].slice(0, MAX_HISTORY)
    setHistory(next)
    saveHistory(next)
  }, [history])

  // ── Run ──
  const handleRun = async () => {
    const queries = queriesText.split('\n').map(s => s.trim()).filter(Boolean)
    if (!queries.length) { setRunError('Enter at least one query.'); return }
    if (!selectedBackends.length) { setRunError('Select at least one backend.'); return }
    setLoading(true); setRunError(null); setSummaryRows([]); setPerQueryRows([]); setDiffPair(null)
    try {
      const payload = mode === 'existing'
        ? { collection_name: collectionName, backends: selectedBackends, queries, full_retrieval: fullRetrieval, repeat_runs: repeatRuns, compare_graph_ab: compareGraphAb }
        : { corpus_text: corpusText || undefined, backends: selectedBackends, queries, full_retrieval: fullRetrieval, repeat_runs: repeatRuns, compare_graph_ab: compareGraphAb }
      const result = await startComparison(payload)
      setSummaryRows(result.summary)
      setPerQueryRows(result.per_query)
      saveRun(result.summary, result.per_query)
    } catch (err: unknown) {
      setRunError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  // ── Export ──
  const exportCSV = useCallback(() => {
    const header = SUMMARY_COLS.map(c => c.key).join(',')
    const body   = sortedRows.map(r => SUMMARY_COLS.map(c => r[c.key]).join(',')).join('\n')
    download('comparison.csv', header + '\n' + body, 'text/csv')
  }, [sortedRows])
  const exportJSON = useCallback(() => {
    download('comparison.json', JSON.stringify({ summary: summaryRows, per_query: perQueryRows }, null, 2), 'application/json')
  }, [summaryRows, perQueryRows])

  const hasResults = summaryRows.length > 0
  const queryCount = queriesText.split('\n').filter(Boolean).length

  return (
    <div className="space-y-5">

      {/* Feature 1 — Chunk Preview Modal */}
      {previewRow && (
        <ChunkPreviewPanel
          chunks={previewRow.chunks || []}
          backend={previewRow.backend}
          query={previewRow.query}
          graphEntities={previewRow.graph_entities}
          graphPaths={previewRow.graph_paths}
          methodContributions={previewRow.method_contributions}
          onClose={() => setPreviewRow(null)}
        />
      )}

      {/* ── Page header ──────────────────────────────────────────────── */}
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <div>
            <h1 className="text-xl font-semibold">Backend Comparison</h1>
            <p className="text-xs text-gray-400 mt-0.5">
              Run the same queries across multiple vector databases. Query latency is the primary differentiator — similarity scores are identical when using the same embedding model.
            </p>
          </div>
          <button
            onClick={() => setShowGuide(true)}
            title="How does Backend Comparison work?"
            className="w-6 h-6 mt-0.5 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-300 hover:text-white text-xs font-bold flex items-center justify-center transition-colors shrink-0"
          >
            ?
          </button>
        </div>
        <button onClick={() => navigate('/search')} className="text-xs text-sky-400 hover:text-sky-300 underline whitespace-nowrap">← Search Lab</button>
      </div>

      {/* ── Step 1: Data source ───────────────────────────────────────── */}
      <Section title="Step 1 — Data source">
        <div className="flex gap-3 mb-3">
          <ModeButton active={mode === 'existing'} onClick={() => setMode('existing')}>📂 Existing collection</ModeButton>
          <ModeButton active={mode === 'paste'}    onClick={() => setMode('paste')}>📋 Paste text</ModeButton>
        </div>
        {mode === 'existing' ? (
          <div className="space-y-2">
            <p className="text-xs text-gray-400">Query an already-ingested collection. Backends showing "✗ not found" return empty results.</p>
            <div className="flex gap-2 items-center">
              <select value={collectionName} onChange={e => setCollectionName(e.target.value)}
                className="flex-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-sky-500">
                {!allCollectionNames.length && <option value="">Loading…</option>}
                {allCollectionNames.map(n => <option key={n} value={n}>{n}</option>)}
              </select>
              <input value={collectionName} onChange={e => setCollectionName(e.target.value)}
                placeholder="or type a name"
                className="w-40 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-sky-500" />
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <p className="text-xs text-gray-400">Paste text to ingest into each backend for this run only.</p>
            <textarea value={corpusText} onChange={e => setCorpusText(e.target.value)} rows={5}
              placeholder="Paste document text here… (leave blank for built-in sample)"
              className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-2 resize-y focus:outline-none focus:border-sky-500" />
          </div>
        )}
      </Section>

      {/* ── Step 2: Backends ─────────────────────────────────────────── */}
      <Section title="Step 2 — Select backends">
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-0.5">
          {ALL_BACKENDS.map(b => (
            <BackendAvailability key={b} backend={b} selected={selectedBackends.includes(b)}
              collections={backendCollections} collectionName={mode === 'existing' ? collectionName : ''}
              onToggle={toggleBackend} />
          ))}
        </div>
      </Section>

      {/* ── Step 3: Queries & options ─────────────────────────────────── */}
      <Section title="Step 3 — Queries & options">
        <textarea value={queriesText} onChange={e => setQueriesText(e.target.value)} rows={4}
          placeholder={"One query per line:\nWhat are the main topics?\nWho are the key entities?"}
          className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-2 resize-y focus:outline-none focus:border-sky-500" />
        <div className="flex items-center gap-3 mt-1 flex-wrap">
          <button onClick={loadSamples} className="text-xs text-sky-400 hover:text-sky-300 underline">Load sample queries</button>
          {queriesText && <span className="text-xs text-gray-500">{queryCount} quer{queryCount === 1 ? 'y' : 'ies'}</span>}
        </div>
        <div className="mt-3 space-y-2">
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input type="checkbox" checked={fullRetrieval} onChange={e => setFullRetrieval(e.target.checked)}
              className="rounded border-gray-600 bg-gray-800" />
            Enable Full Retrieval comparison (LLM methods ON, requires LM Studio)
          </label>
          <label className="flex items-center gap-2 text-sm text-gray-400 cursor-pointer select-none">
            <input type="checkbox" checked={compareGraphAb} onChange={e => setCompareGraphAb(e.target.checked)}
              className="rounded border-gray-600 bg-gray-800" />
            🕸️ Graph A/B comparison — run each query with Knowledge Graph ON vs OFF, show Δ score + graph trail
            <Tooltip text="Each query runs twice: once with the Knowledge Graph enabled (Dense+BM25+Graph) and once with only Dense+BM25. The score delta shows whether the graph improved retrieval." />
          </label>
          {/* Feature 6 — Repeat runs for P50/P95 */}
          <div className="flex items-center gap-3">
            <label className="text-sm text-gray-400 select-none">
              Repeat runs for latency percentiles:
              <Tooltip text="Each query is run N times. P50/P95 latency is computed from all runs. Higher = more accurate latency measurement but slower overall." />
            </label>
            <div className="flex gap-1">
              {[1, 3, 5].map(n => (
                <button key={n} onClick={() => setRepeatRuns(n)}
                  className={`px-3 py-1 rounded text-xs font-medium ${repeatRuns === n ? 'bg-sky-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'}`}>
                  {n === 1 ? '1× (off)' : `${n}×`}
                </button>
              ))}
            </div>
          </div>
        </div>
      </Section>

      {/* ── Run button & toolbar ──────────────────────────────────────── */}
      <div className="flex gap-2 flex-wrap items-center">
        <button onClick={handleRun} disabled={loading || !selectedBackends.length}
          className="px-5 py-2 bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white rounded text-sm font-medium transition-colors">
          {loading ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24" fill="none">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
              </svg>
              Running{repeatRuns > 1 ? ` (${repeatRuns}× per query)` : ''}…
            </span>
          ) : '▶ Run Comparison'}
        </button>
        {hasResults && (
          <>
            <button onClick={() => setViewMode(v => v === 'table' ? 'chart' : 'table')}
              className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm">
              {viewMode === 'table' ? '📊 Chart view' : '📋 Table view'}
            </button>
            <button onClick={() => setShowPerQuery(v => !v)} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm">
              {showPerQuery ? 'Hide per-query' : 'Per-query detail'}
            </button>
            <button onClick={() => setShowOverlap(v => !v)} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm">
              {showOverlap ? 'Hide overlap' : '🔀 Overlap matrix'}
            </button>
            <button onClick={exportCSV}  className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm">Export CSV</button>
            <button onClick={exportJSON} className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm">Export JSON</button>
          </>
        )}
        <button onClick={() => setShowHistory(v => !v)}
          className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm ml-auto">
          🕐 History {history.length > 0 && `(${history.length})`}
        </button>
      </div>

      {runError && <div className="bg-red-900/30 border border-red-700 rounded p-3 text-sm text-red-300">{runError}</div>}

      {/* Feature 5 — History panel */}
      {showHistory && (
        <Section title="Run History">
          <HistoryPanel
            history={history}
            onLoad={run => { setSummaryRows(run.summary); setPerQueryRows(run.per_query) }}
            onClear={() => { setHistory([]); saveHistory([]) }}
            onDiff={(a, b) => { setDiffPair([a, b]); setShowHistory(false) }}
          />
        </Section>
      )}

      {/* Feature 5 — Diff view */}
      {diffPair && (
        <Section title={`Diff: ${diffPair[0].label} → ${diffPair[1].label}`}
          action={<button onClick={() => setDiffPair(null)} className="text-xs text-gray-500 hover:text-gray-300">✕ Close diff</button>}>
          <DiffTable runA={diffPair[0]} runB={diffPair[1]} />
        </Section>
      )}

      {/* ── Results ───────────────────────────────────────────────────── */}
      {(hasResults || loading) && (
        <Section title="Results — Backend Summary"
          action={hasResults && viewMode === 'chart' ? (
            <select value={chartMetric as string} onChange={e => setChartMetric(e.target.value as keyof CompareRow)}
              className="text-xs bg-gray-800 border border-gray-700 rounded px-2 py-1 text-gray-300">
              <option value="avg_query_latency_ms">Avg Latency</option>
              <option value="latency_p95_ms">P95 Latency</option>
              <option value="base_top_score">Top Score</option>
              <option value="avg_score">Avg Score</option>
              <option value="total_result_count">Result Count</option>
              <option value="ingest_time_s">Ingest Time</option>
            </select>
          ) : undefined}>
          {loading ? (
            <p className="text-sm text-gray-500 py-4 text-center">Running queries across backends…</p>
          ) : (
            <>
              {/* Feature 4 — Winner Badges */}
              <WinnerBadges rows={summaryRows} graphAb={compareGraphAb} />

              {viewMode === 'chart' ? (
                /* Feature 2 — Bar Chart */
                <BarChart rows={sortedRows} metric={chartMetric}
                  label={`${SUMMARY_COLS.find(c => c.key === chartMetric)?.label ?? chartMetric} per backend`} />
              ) : (
                <>
                  <p className="text-xs text-gray-500 mb-2">
                    <span className="text-yellow-400">ℹ</span> Similarity scores are identical across backends when using the same embedding model. <strong>Query Latency</strong> is the primary real differentiator. Click column headers to sort.
                    {repeatRuns > 1 && <> · P50/P95 from {repeatRuns} repeated runs per query.</>}
                  </p>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm border-collapse">
                      <thead>
                        <tr className="border-b border-gray-700">
                          {SUMMARY_COLS.map(col => (
                            <th key={col.key} onClick={() => handleSort(col.key)}
                              className="px-3 py-2 text-left text-gray-400 cursor-pointer hover:text-white select-none whitespace-nowrap"
                              title={col.tip}>
                              {col.label}{sortKey === col.key ? (sortAsc ? ' ▲' : ' ▼') : ''}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {sortedRows.map(row => (
                          <tr key={row.backend} className="border-b border-gray-800 hover:bg-gray-800/50">
                            {SUMMARY_COLS.map(col => {
                              const val = row[col.key]
                              return (
                                <td key={col.key} className={`px-3 py-2 ${scoreColor(col.key, val as number)}`}>
                                  {col.key === 'backend' ? (
                                    <span className="flex items-center gap-1.5">
                                      <span className={`w-2 h-2 rounded-sm ${BACKEND_COLORS[val as string] ?? 'bg-gray-500'}`} />
                                      {val}
                                    </span>
                                  ) : col.key.includes('latency') && typeof val === 'number' ? fmtLatency(val)
                                   : col.key === 'total_result_count' && typeof val === 'number' ? val.toString()
                                   : col.key === 'errors' && typeof val === 'number' ? val.toString()
                                   : typeof val === 'number' ? val.toFixed(3) : val}
                                </td>
                              )
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </>
          )}
        </Section>
      )}

      {/* Feature 3 — Overlap Matrix */}
      {showOverlap && perQueryRows.length > 0 && (
        <Section title="Result Overlap Matrix">
          <OverlapMatrix perQuery={perQueryRows} backends={selectedBackends} />
        </Section>
      )}

      {/* ── Per-query breakdown ───────────────────────────────────────── */}
      {showPerQuery && perQueryRows.length > 0 && (
        <Section title="Per-Query Breakdown">
          <p className="text-xs text-gray-400 mb-2">
            Click any row to preview retrieved chunks and graph trail. {repeatRuns > 1 && 'P50/P95 from repeated runs.'}
            {compareGraphAb && ' 🕸️ Δ = score with graph ON minus score with graph OFF.'}
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="px-3 py-2 text-left text-gray-400">Backend</th>
                  <th className="px-3 py-2 text-left text-gray-400">Query</th>
                  <th className="px-3 py-2 text-left text-gray-400">P50 Latency</th>
                  {repeatRuns > 1 && <th className="px-3 py-2 text-left text-gray-400">P95 Latency</th>}
                  <th className="px-3 py-2 text-left text-gray-400">Top Score</th>
                  {compareGraphAb && <th className="px-3 py-2 text-left text-gray-400" title="Score without Knowledge Graph">No-Graph Score</th>}
                  {compareGraphAb && <th className="px-3 py-2 text-left text-gray-400" title="Score delta: graph ON minus graph OFF. Positive = graph helped.">Δ Graph</th>}
                  <th className="px-3 py-2 text-left text-gray-400">KW Hits</th>
                  <th className="px-3 py-2 text-left text-gray-400">Results</th>
                  <th className="px-3 py-2 text-left text-gray-400">Error</th>
                </tr>
              </thead>
              <tbody>
                {perQueryRows.map((row, idx) => (
                  <tr key={idx}
                    className={`border-b border-gray-800 hover:bg-gray-800/60 cursor-pointer ${row.chunks?.length ? 'hover:border-sky-800' : ''}`}
                    onClick={() => (row.chunks?.length || row.graph_entities?.length || row.graph_paths?.length) && setPreviewRow(row)}
                    title={(row.chunks?.length || row.graph_entities?.length) ? 'Click to preview chunks & graph trail' : undefined}>
                    <td className="px-3 py-1.5">
                      <span className="flex items-center gap-1">
                        <span className={`w-2 h-2 rounded-sm flex-shrink-0 ${BACKEND_COLORS[row.backend] ?? 'bg-gray-500'}`} />
                        <span className={`capitalize ${BACKEND_TEXT[row.backend] ?? 'text-gray-300'}`}>{row.backend}</span>
                      </span>
                    </td>
                    <td className="px-3 py-1.5 text-gray-300 max-w-xs truncate" title={row.query}>{row.query}</td>
                    <td className={`px-3 py-1.5 ${scoreColor('latency', row.latency_p50_ms)}`}>{fmtLatency(row.latency_p50_ms)}</td>
                    {repeatRuns > 1 && <td className={`px-3 py-1.5 ${scoreColor('latency', row.latency_p95_ms)}`}>{fmtLatency(row.latency_p95_ms)}</td>}
                    <td className={`px-3 py-1.5 ${scoreColor('base_top_score', row.top_score)}`}>{fmtScore(row.top_score)}</td>
                    {compareGraphAb && (
                      <td className={`px-3 py-1.5 ${scoreColor('base_top_score', row.score_no_graph ?? 0)}`}>
                        {row.score_no_graph != null ? fmtScore(row.score_no_graph) : '—'}
                      </td>
                    )}
                    {compareGraphAb && (
                      <td className={`px-3 py-1.5 font-mono ${
                        (row.score_delta ?? 0) > 0 ? 'text-green-400' :
                        (row.score_delta ?? 0) < 0 ? 'text-red-400' : 'text-gray-500'
                      }`}>
                        {row.score_delta != null
                          ? `${row.score_delta > 0 ? '+' : ''}${row.score_delta.toFixed(4)}`
                          : '—'}
                      </td>
                    )}
                    <td className={`px-3 py-1.5 ${scoreColor('base_kw_hits', row.kw_hits)}`}>{row.kw_hits}</td>
                    <td className={`px-3 py-1.5 ${scoreColor('result_count', row.result_count)}`}>
                      {row.result_count ?? '—'}
                      {row.chunks?.length ? <span className="ml-1 text-sky-500 text-xs">👁</span> : null}
                      {(row.graph_entities?.length ?? 0) > 0 ? <span className="ml-0.5 text-indigo-400 text-xs" title={`${row.graph_entities!.length} graph entities`}>🕸️</span> : null}
                    </td>
                    <td className="px-3 py-1.5 text-red-400 max-w-xs truncate" title={row.error}>{row.error ?? ''}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
      {showGuide && <CompareGuideModal onClose={() => setShowGuide(false)} />}
    </div>
  )
}

