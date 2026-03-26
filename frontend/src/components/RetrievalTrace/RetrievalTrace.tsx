import { useState } from 'react'
import type { MethodContribution } from '../../api/search'

export interface RetrievalStep {
  method: string
  candidates_before: number
  candidates_after: number
  score_min?: number
  score_max?: number
}

export interface MethodContributionStat {
  candidates_before?: number
  candidates_after?: number
  delta?: number
  chunks_contributed?: number
  contribution_pct?: number
}

export interface TracedChunk {
  chunk_id: string
  text: string
  score: number
  method_lineage?: MethodContribution[]
  metadata?: Record<string, unknown>
}

interface RetrievalTraceProps {
  /** Pipeline phase-by-phase filtering steps */
  steps?: RetrievalStep[]
  /** Aggregate per-method contribution stats */
  methodContributions?: Record<string, MethodContributionStat>
  /** Chunks with their per-chunk method lineage */
  chunks?: TracedChunk[]
  /** The LLM-generated answer (for answer attribution) */
  answer?: string
  /** Label shown on the toggle button */
  label?: string
}

const METHOD_COLORS: Record<string, string> = {
  'Dense Vector':         '#3b82f6',
  'BM25 Keyword':         '#8b5cf6',
  'SPLADE Sparse Neural': '#ec4899',
  'Knowledge Graph':      '#10b981',
  'RAPTOR':               '#f59e0b',
  'Cross-Encoder Rerank': '#06b6d4',
  'MMR Diversity':        '#84cc16',
  'Query Rewriting':      '#f97316',
  'HyDE Expansion':       '#6366f1',
  'Multi-Query':          '#14b8a6',
}

function getMethodColor(method: string): string {
  return METHOD_COLORS[method] ?? '#9ca3af'
}

type Tab = 'pipeline' | 'contributions' | 'chunks' | 'attribution'

export default function RetrievalTrace({
  steps = [],
  methodContributions = {},
  chunks = [],
  answer = '',
  label = 'Retrieval Trace',
}: RetrievalTraceProps) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<Tab>('contributions')
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set())

  function toggleChunk(id: string) {
    setExpandedChunks(prev => {
      const next = new Set(prev)
      next.has(id) ? next.delete(id) : next.add(id)
      return next
    })
  }

  const hasSteps = steps.length > 0
  const hasContribs = Object.keys(methodContributions).length > 0
  const hasChunks = chunks.length > 0
  const hasAnswer = answer.trim().length > 0

  if (!hasSteps && !hasContribs && !hasChunks) return null

  // Compute answer attribution: word overlap between chunk text and answer
  const answerWords = new Set(answer.toLowerCase().split(/\W+/).filter(w => w.length > 3))
  const attribution: Record<string, number> = {}
  for (const chunk of chunks) {
    const chunkWords = new Set(chunk.text.toLowerCase().split(/\W+/).filter(w => w.length > 3))
    const overlap = [...chunkWords].filter(w => answerWords.has(w)).length
    const overlapPct = answerWords.size > 0 ? overlap / answerWords.size : 0
    for (const m of chunk.method_lineage ?? []) {
      attribution[m.method] = (attribution[m.method] ?? 0) + overlapPct
    }
  }
  const totalAttrib = Object.values(attribution).reduce((a, b) => a + b, 0) || 1
  const attribPct = Object.fromEntries(
    Object.entries(attribution).map(([m, v]) => [m, (v / totalAttrib) * 100])
  )

  const sortedContribs = Object.entries(methodContributions)
    .sort(([, a], [, b]) => (b.contribution_pct ?? 0) - (a.contribution_pct ?? 0))

  const sortedAttrib = Object.entries(attribPct)
    .sort(([, a], [, b]) => b - a)

  const tabClass = (t: Tab) =>
    `px-3 py-1 text-xs font-medium rounded-t cursor-pointer transition-colors ${
      tab === t ? 'bg-gray-800 text-white' : 'text-gray-400 hover:text-gray-200'
    }`

  return (
    <div className="border border-gray-700 rounded-lg bg-gray-900 text-sm">
      {/* Toggle header */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-gray-300 hover:text-white"
      >
        <span className="flex items-center gap-2">
          🔬 <span className="font-medium">{label}</span>
          {hasContribs && (
            <span className="text-xs text-gray-500">
              ({sortedContribs.filter(([, s]) => (s.contribution_pct ?? 0) > 0).length} active methods)
            </span>
          )}
        </span>
        <span className="text-gray-500">{open ? '▼' : '▶'}</span>
      </button>

      {open && (
        <div className="border-t border-gray-700">
          {/* Tab bar */}
          <div className="flex gap-1 px-3 pt-2 border-b border-gray-700">
            {hasContribs && <button className={tabClass('contributions')} onClick={() => setTab('contributions')}>📊 Method Contributions</button>}
            {hasChunks && <button className={tabClass('chunks')} onClick={() => setTab('chunks')}>📄 Per-Chunk Attribution</button>}
            {hasAnswer && <button className={tabClass('attribution')} onClick={() => setTab('attribution')}>🎯 Answer Attribution</button>}
            {hasSteps && <button className={tabClass('pipeline')} onClick={() => setTab('pipeline')}>⚙️ Pipeline Steps</button>}
          </div>

          <div className="p-4">
            {/* ── Method Contributions tab ──────────────────────────────── */}
            {tab === 'contributions' && hasContribs && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500 mb-3">
                  Each bar shows the % of final result chunks that this method contributed.
                  Methods with 0% were enabled but did not surface any unique chunks.
                </p>
                {sortedContribs.map(([method, stats]) => (
                  <div key={method}>
                    <div className="flex items-center justify-between mb-0.5">
                      <span className="text-xs font-medium" style={{ color: getMethodColor(method) }}>
                        {method}
                      </span>
                      <div className="flex items-center gap-3 text-xs text-gray-400">
                        <span>{stats.chunks_contributed ?? 0} chunks</span>
                        <span className="font-semibold text-white w-10 text-right">
                          {(stats.contribution_pct ?? 0).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div
                        className="h-2 rounded-full transition-all"
                        style={{
                          width: `${Math.min(stats.contribution_pct ?? 0, 100)}%`,
                          backgroundColor: getMethodColor(method),
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* ── Per-Chunk Attribution tab ─────────────────────────────── */}
            {tab === 'chunks' && hasChunks && (
              <div className="space-y-3 max-h-[600px] overflow-y-auto pr-1">
                <p className="text-xs text-gray-500 mb-2">
                  Each card shows the full retrieved chunk text, its RRF score, source metadata,
                  and which retrieval methods surfaced it. Click a card to expand/collapse.
                </p>
                {chunks.map((chunk, i) => {
                  const isExpanded = expandedChunks.has(chunk.chunk_id)
                  const src = chunk.metadata?.source as string | undefined
                  const section = chunk.metadata?.section_title as string | undefined
                  const page = chunk.metadata?.page as number | undefined
                  // Highlight words that appear in the answer
                  const answerWordSet = new Set(answer.toLowerCase().split(/\W+/).filter(w => w.length > 3))
                  const chunkWords = chunk.text.split(/(\W+)/)
                  return (
                    <div
                      key={chunk.chunk_id}
                      className="border border-gray-700 rounded-lg overflow-hidden"
                    >
                      {/* Card header — always visible */}
                      <button
                        onClick={() => toggleChunk(chunk.chunk_id)}
                        className="w-full flex items-start justify-between p-3 text-left hover:bg-gray-800 transition-colors"
                      >
                        <div className="flex flex-col gap-1 flex-1 min-w-0">
                          {/* Rank + score */}
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-white bg-gray-700 rounded px-1.5 py-0.5">
                              #{i + 1}
                            </span>
                            <span className="text-xs text-gray-400">
                              RRF score: <span className="text-white font-mono">{chunk.score.toFixed(5)}</span>
                            </span>
                            {src && (
                              <span className="text-xs text-gray-500 truncate max-w-[200px]" title={src}>
                                📄 {src.split('/').pop()}
                              </span>
                            )}
                            {section && (
                              <span className="text-xs text-gray-500 italic truncate max-w-[180px]" title={section}>
                                § {section}
                              </span>
                            )}
                            {page != null && (
                              <span className="text-xs text-gray-500">p.{page}</span>
                            )}
                          </div>
                          {/* Method badges */}
                          <div className="flex flex-wrap gap-1 mt-1">
                            {(chunk.method_lineage ?? []).map(m => (
                              <span
                                key={m.method}
                                className="text-xs px-1.5 py-0.5 rounded font-medium"
                                style={{
                                  backgroundColor: `${getMethodColor(m.method)}22`,
                                  color: getMethodColor(m.method),
                                  border: `1px solid ${getMethodColor(m.method)}55`,
                                }}
                                title={`Rank assigned by this method: ${m.rank} · RRF term: 1/(60+${m.rank}) = ${m.rrf_contribution.toFixed(5)}`}
                              >
                                {m.method.replace(' Vector', '').replace(' Keyword', '').replace(' Sparse Neural', '')}
                                <span className="opacity-70 ml-1">#{m.rank} · {m.rrf_contribution.toFixed(4)}</span>
                              </span>
                            ))}
                            {(chunk.method_lineage ?? []).length === 0 && (
                              <span className="text-xs text-gray-600 italic">no lineage recorded</span>
                            )}
                          </div>
                          {/* Preview when collapsed */}
                          {!isExpanded && (
                            <p className="text-xs text-gray-400 mt-1 line-clamp-2 leading-relaxed">
                              {chunk.text}
                            </p>
                          )}
                        </div>
                        <span className="text-gray-600 ml-3 mt-0.5 flex-shrink-0">
                          {isExpanded ? '▲' : '▼'}
                        </span>
                      </button>

                      {/* Full chunk text — shown when expanded */}
                      {isExpanded && (
                        <div className="border-t border-gray-700 bg-gray-950 p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
                              Retrieved Text
                            </span>
                            <span className="text-xs text-gray-600 font-mono">{chunk.chunk_id}</span>
                          </div>
                          <p className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
                            {answerWordSet.size > 0
                              ? chunkWords.map((word, wi) => {
                                  const clean = word.toLowerCase().replace(/\W/g, '')
                                  const isMatch = clean.length > 3 && answerWordSet.has(clean)
                                  return isMatch
                                    ? <mark key={wi} className="bg-yellow-900/60 text-yellow-200 rounded-sm">{word}</mark>
                                    : <span key={wi}>{word}</span>
                                })
                              : chunk.text
                            }
                          </p>
                          {/* Per-method RRF detail table */}
                          {(chunk.method_lineage ?? []).length > 0 && (
                            <div className="mt-3 border-t border-gray-800 pt-2">
                              <p className="text-xs font-semibold text-gray-500 mb-1.5">RRF Score Breakdown</p>
                              <table className="w-full text-xs">
                                <thead>
                                  <tr className="text-gray-600">
                                    <th className="text-left font-medium pb-1">Method</th>
                                    <th className="text-right font-medium pb-1">Rank</th>
                                    <th className="text-right font-medium pb-1">RRF Term</th>
                                    <th className="text-right font-medium pb-1">Share</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {chunk.method_lineage.map(m => {
                                    const sharePct = chunk.score > 0
                                      ? (m.rrf_contribution / chunk.score) * 100
                                      : 0
                                    return (
                                      <tr key={m.method} className="border-t border-gray-800">
                                        <td className="py-0.5" style={{ color: getMethodColor(m.method) }}>{m.method}</td>
                                        <td className="text-right text-gray-400 font-mono">#{m.rank}</td>
                                        <td className="text-right text-gray-300 font-mono">{m.rrf_contribution.toFixed(5)}</td>
                                        <td className="text-right text-gray-400">{sharePct.toFixed(1)}%</td>
                                      </tr>
                                    )
                                  })}
                                  <tr className="border-t border-gray-700 font-semibold">
                                    <td className="py-0.5 text-gray-300">Total (RRF)</td>
                                    <td />
                                    <td className="text-right text-white font-mono">{chunk.score.toFixed(5)}</td>
                                    <td className="text-right text-white">100%</td>
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>
            )}

            {/* ── Answer Attribution tab ────────────────────────────────── */}
            {tab === 'attribution' && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500 mb-3">
                  Estimated contribution to the final answer based on word overlap between
                  retrieved chunks and the generated answer text.
                </p>
                {sortedAttrib.length === 0 ? (
                  <p className="text-xs text-gray-500">No answer text available for attribution.</p>
                ) : (
                  sortedAttrib.map(([method, pct]) => (
                    <div key={method}>
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-xs font-medium" style={{ color: getMethodColor(method) }}>{method}</span>
                        <span className="text-xs font-semibold text-white w-10 text-right">{pct.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className="h-2 rounded-full transition-all"
                          style={{ width: `${pct}%`, backgroundColor: getMethodColor(method) }}
                        />
                      </div>
                    </div>
                  ))
                )}
                {hasAnswer && (
                  <details className="mt-3">
                    <summary className="text-xs text-gray-500 cursor-pointer">Show answer text</summary>
                    <p className="text-xs text-gray-400 mt-1 bg-gray-800 rounded p-2 max-h-32 overflow-y-auto">{answer}</p>
                  </details>
                )}
              </div>
            )}

            {/* ── Pipeline Steps tab ───────────────────────────────────── */}
            {tab === 'pipeline' && hasSteps && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500 mb-2">
                  Phase-by-phase candidate filtering through the retrieval pipeline.
                </p>
                {steps.map((step, i) => {
                  const maxBefore = Math.max(...steps.map(s => s.candidates_before), 1)
                  const beforePct = (step.candidates_before / maxBefore) * 100
                  const afterPct = (step.candidates_after / maxBefore) * 100
                  return (
                    <div key={i}>
                      <div className="flex items-center justify-between text-xs text-gray-400 mb-0.5">
                        <span className="font-medium text-gray-300 w-40 truncate">{step.method}</span>
                        <span>{step.candidates_before} → {step.candidates_after}</span>
                      </div>
                      <div className="relative w-full h-3 bg-gray-700 rounded">
                        <div className="absolute h-3 bg-gray-600 rounded" style={{ width: `${beforePct}%` }} />
                        <div className="absolute h-3 bg-brand-500 rounded" style={{ width: `${afterPct}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
