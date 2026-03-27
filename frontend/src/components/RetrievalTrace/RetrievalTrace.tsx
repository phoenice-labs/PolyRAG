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
  chunks_contributed?: number   // final chunks this method appears in (lineage presence)
  unique_chunks_added?: number  // chunks ONLY this method found (not in Dense/BM25)
  rrf_boost_total?: number      // total RRF score this method added across all final chunks
  avg_rank?: number             // average rank this method assigned to scored chunks
  contribution_pct?: number
  status?: 'active' | 'zero' | 'disabled'
  reason?: string
}

export interface TracedChunk {
  chunk_id: string
  text: string
  score: number
  method_lineage?: MethodContribution[]
  metadata?: Record<string, unknown>
}

interface RetrievalTraceProps {
  steps?: RetrievalStep[]
  methodContributions?: Record<string, MethodContributionStat>
  chunks?: TracedChunk[]
  answer?: string
  /** entities extracted from the query by the knowledge graph */
  graphEntities?: string[]
  /** traversal paths found in the knowledge graph */
  graphPaths?: string[]
  /** query expansion variants (rewritten, HyDE, paraphrases, stepback) */
  queryVariants?: { rewritten?: string; paraphrases?: string[]; hyde_text?: string; stepback?: string }
  label?: string
}

const METHOD_COLORS: Record<string, string> = {
  'Dense Vector':           '#3b82f6',
  'BM25 Keyword':           '#8b5cf6',
  'SPLADE Sparse Neural':   '#ec4899',
  'Knowledge Graph':        '#10b981',
  'RAPTOR':                 '#f59e0b',
  'Cross-Encoder Rerank':   '#06b6d4',
  'MMR Diversity':          '#84cc16',
  'Query Rewriting':        '#f97316',
  'HyDE Expansion':         '#6366f1',
  'Multi-Query':            '#14b8a6',
  'Contextual Rerank':      '#a78bfa',
  'LLM Graph Extraction':   '#34d399',
}

function getColor(method: string): string {
  return METHOD_COLORS[method] ?? '#9ca3af'
}

type Tab = 'contributions' | 'chunks' | 'attribution' | 'graph' | 'query' | 'pipeline'

const STATUS_CONFIG = {
  active:   { icon: '✅', label: 'Active',    bg: 'bg-green-900/30',  border: 'border-green-700/40',  text: 'text-green-400' },
  zero:     { icon: '⚠️', label: 'Zero contribution', bg: 'bg-yellow-900/20', border: 'border-yellow-700/30', text: 'text-yellow-400' },
  disabled: { icon: '⭕', label: 'Disabled',  bg: 'bg-gray-800/40',   border: 'border-gray-700/30',   text: 'text-gray-500' },
}

export default function RetrievalTrace({
  steps = [],
  methodContributions = {},
  chunks = [],
  answer = '',
  graphEntities = [],
  graphPaths = [],
  queryVariants = {},
  label = 'Method Traceability',
}: RetrievalTraceProps) {
  const [open, setOpen] = useState(false)
  const [tab, setTab] = useState<Tab>('contributions')
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set())
  const [expandedMethods, setExpandedMethods] = useState<Set<string>>(new Set())

  function toggleChunk(id: string) {
    setExpandedChunks(prev => { const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n })
  }
  function toggleMethod(m: string) {
    setExpandedMethods(prev => { const n = new Set(prev); n.has(m) ? n.delete(m) : n.add(m); return n })
  }

  const hasContribs = Object.keys(methodContributions).length > 0
  const hasChunks = chunks.length > 0
  const hasAnswer = answer.trim().length > 0
  const hasGraph = graphEntities.length > 0 || graphPaths.length > 0
  const hasQuery = Object.values(queryVariants).some(v => v && (typeof v === 'string' ? v.length : v.length > 0))

  if (!hasContribs && !hasChunks && steps.length === 0) return null

  // Partition methods by status
  const active   = Object.entries(methodContributions).filter(([, s]) => s.status === 'active')
  const zero     = Object.entries(methodContributions).filter(([, s]) => s.status === 'zero')
  const disabled = Object.entries(methodContributions).filter(([, s]) => s.status === 'disabled')

  // Answer attribution via word-overlap
  const answerWords = new Set(answer.toLowerCase().split(/\W+/).filter(w => w.length > 3))
  const attribution: Record<string, number> = {}
  for (const chunk of chunks) {
    const cw = new Set(chunk.text.toLowerCase().split(/\W+/).filter(w => w.length > 3))
    const overlap = [...cw].filter(w => answerWords.has(w)).length
    const pct = answerWords.size > 0 ? overlap / answerWords.size : 0
    for (const m of chunk.method_lineage ?? []) {
      attribution[m.method] = (attribution[m.method] ?? 0) + pct
    }
  }
  const totalAttrib = Object.values(attribution).reduce((a, b) => a + b, 0) || 1
  const attribPct = Object.fromEntries(Object.entries(attribution).map(([m, v]) => [m, (v / totalAttrib) * 100]))
  const sortedAttrib = Object.entries(attribPct).sort(([, a], [, b]) => b - a)

  const tabBtn = (t: Tab, icon: string, lbl: string, show = true) => show ? (
    <button
      key={t}
      onClick={() => setTab(t)}
      className={`px-3 py-1.5 text-xs font-medium rounded-t transition-colors whitespace-nowrap ${
        tab === t ? 'bg-gray-800 text-white border-b-2 border-brand-500' : 'text-gray-400 hover:text-gray-200'
      }`}
    >{icon} {lbl}</button>
  ) : null

  return (
    <div className="border border-gray-700 rounded-lg bg-gray-900 text-sm">
      {/* Header toggle */}
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-2.5 text-gray-300 hover:text-white"
      >
        <span className="flex items-center gap-2">
          🔬 <span className="font-medium">{label}</span>
          <span className="text-xs text-gray-500">
            {active.length} active · {zero.length} zero · {disabled.length} disabled
          </span>
        </span>
        <span className="text-gray-500">{open ? '▼' : '▶'}</span>
      </button>

      {open && (
        <div className="border-t border-gray-700">
          {/* Tab bar */}
          <div className="flex gap-0.5 px-3 pt-2 border-b border-gray-700 overflow-x-auto">
            {tabBtn('contributions', '📊', 'All Methods')}
            {tabBtn('chunks', '📄', 'Retrieved Chunks', hasChunks)}
            {tabBtn('attribution', '🎯', 'Answer Attribution', hasAnswer)}
            {tabBtn('graph', '🕸️', 'Graph Trace', hasGraph)}
            {tabBtn('query', '✏️', 'Query Expansion', hasQuery)}
            {tabBtn('pipeline', '⚙️', 'Pipeline Steps', steps.length > 0)}
          </div>

          <div className="p-4 space-y-4">

            {/* ── ALL METHODS tab ─────────────────────────────────────── */}
            {tab === 'contributions' && (
              <div className="space-y-4">
                <p className="text-xs text-gray-500">
                  All 10 retrieval methods listed — <span className="text-green-400">✅ active</span>,{' '}
                  <span className="text-yellow-400">⚠️ zero unique chunks</span> (may still boost RRF scores), or{' '}
                  <span className="text-gray-500">⭕ not enabled</span>. Click any row for full diagnostics including RRF boost, avg rank, and unique chunk count.
                </p>

                {/* Active methods */}
                {active.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-green-400 uppercase tracking-wide mb-2">✅ Active Methods</p>
                    <div className="space-y-2">
                      {active.sort(([, a], [, b]) => (b.contribution_pct ?? 0) - (a.contribution_pct ?? 0)).map(([method, stats]) => (
                        <MethodCard key={method} method={method} stats={stats} expanded={expandedMethods.has(method)} onToggle={() => toggleMethod(method)} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Zero contribution methods */}
                {zero.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-yellow-400 uppercase tracking-wide mb-2">⚠️ Enabled but Zero Contribution — click to diagnose</p>
                    <div className="space-y-2">
                      {zero.map(([method, stats]) => (
                        <MethodCard key={method} method={method} stats={stats} expanded={expandedMethods.has(method)} onToggle={() => toggleMethod(method)} />
                      ))}
                    </div>
                  </div>
                )}

                {/* Disabled methods */}
                {disabled.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">⭕ Not Enabled — toggle ON in Method Settings to test</p>
                    <div className="space-y-1">
                      {disabled.map(([method, stats]) => (
                        <MethodCard key={method} method={method} stats={stats} expanded={expandedMethods.has(method)} onToggle={() => toggleMethod(method)} />
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── RETRIEVED CHUNKS tab ────────────────────────────────── */}
            {tab === 'chunks' && hasChunks && (
              <div className="space-y-2 max-h-[600px] overflow-y-auto pr-1">
                <p className="text-xs text-gray-500 mb-2">
                  Each card shows the full retrieved chunk, its RRF score, source, and exactly which methods surfaced it.
                  Words highlighted in yellow appear in the final answer.
                </p>
                {chunks.map((chunk, i) => {
                  const exp = expandedChunks.has(chunk.chunk_id)
                  const src = chunk.metadata?.source as string | undefined
                  const section = chunk.metadata?.section_title as string | undefined
                  const page = chunk.metadata?.page as number | undefined
                  const chunkWords = chunk.text.split(/(\W+)/)
                  return (
                    <div key={chunk.chunk_id} className="border border-gray-700 rounded-lg overflow-hidden">
                      <button onClick={() => toggleChunk(chunk.chunk_id)}
                        className="w-full flex items-start justify-between p-3 text-left hover:bg-gray-800 transition-colors"
                      >
                        <div className="flex flex-col gap-1 flex-1 min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <span className="text-xs font-bold text-white bg-gray-700 rounded px-1.5 py-0.5">#{i + 1}</span>
                            <span className="text-xs text-gray-400">RRF <span className="text-white font-mono">{chunk.score.toFixed(5)}</span></span>
                            {src && <span className="text-xs text-gray-500 truncate max-w-[150px]" title={src}>📄 {src.split('/').pop()}</span>}
                            {section && <span className="text-xs text-gray-500 italic truncate max-w-[150px]" title={section}>§ {section}</span>}
                            {page != null && <span className="text-xs text-gray-500">p.{page}</span>}
                          </div>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {(chunk.method_lineage ?? []).map(m => (
                              <span key={m.method} className="text-xs px-1.5 py-0.5 rounded font-medium"
                                style={{ backgroundColor: `${getColor(m.method)}22`, color: getColor(m.method), border: `1px solid ${getColor(m.method)}55` }}
                                title={`Rank: ${m.rank} · RRF: ${m.rrf_contribution.toFixed(5)}`}
                              >
                                {m.method.replace(' Vector','').replace(' Keyword','').replace(' Sparse Neural','')} #{m.rank}
                              </span>
                            ))}
                            {(chunk.method_lineage ?? []).length === 0 && (
                              <span className="text-xs text-gray-600 italic">no method lineage</span>
                            )}
                          </div>
                          {!exp && <p className="text-xs text-gray-400 mt-1 line-clamp-2 leading-relaxed">{chunk.text}</p>}
                        </div>
                        <span className="text-gray-600 ml-3 mt-0.5 flex-shrink-0">{exp ? '▲' : '▼'}</span>
                      </button>
                      {exp && (
                        <div className="border-t border-gray-700 bg-gray-950 p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Retrieved Text</span>
                            <span className="text-xs text-gray-600 font-mono">{chunk.chunk_id}</span>
                          </div>
                          <p className="text-xs text-gray-300 leading-relaxed whitespace-pre-wrap">
                            {answerWords.size > 0
                              ? chunkWords.map((word, wi) => {
                                  const clean = word.toLowerCase().replace(/\W/g, '')
                                  return clean.length > 3 && answerWords.has(clean)
                                    ? <mark key={wi} className="bg-yellow-900/60 text-yellow-200 rounded-sm">{word}</mark>
                                    : <span key={wi}>{word}</span>
                                })
                              : chunk.text
                            }
                          </p>
                          {(chunk.method_lineage ?? []).length > 0 && (
                            <div className="mt-3 border-t border-gray-800 pt-2">
                              <p className="text-xs font-semibold text-gray-500 mb-1.5">RRF Score Breakdown</p>
                              <table className="w-full text-xs">
                                <thead><tr className="text-gray-600">
                                  <th className="text-left pb-1">Method</th>
                                  <th className="text-right pb-1">Rank</th>
                                  <th className="text-right pb-1">RRF Term</th>
                                  <th className="text-right pb-1">Share</th>
                                </tr></thead>
                                <tbody>
                                  {chunk.method_lineage!.map(m => (
                                    <tr key={m.method} className="border-t border-gray-800">
                                      <td className="py-0.5" style={{ color: getColor(m.method) }}>{m.method}</td>
                                      <td className="text-right text-gray-400 font-mono">#{m.rank}</td>
                                      <td className="text-right text-gray-300 font-mono">{m.rrf_contribution.toFixed(5)}</td>
                                      <td className="text-right text-gray-400">{chunk.score > 0 ? ((m.rrf_contribution / chunk.score) * 100).toFixed(1) : 0}%</td>
                                    </tr>
                                  ))}
                                  <tr className="border-t border-gray-700 font-semibold">
                                    <td className="py-0.5 text-gray-300">Total</td><td /><td className="text-right text-white font-mono">{chunk.score.toFixed(5)}</td><td className="text-right text-white">100%</td>
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

            {/* ── ANSWER ATTRIBUTION tab ──────────────────────────────── */}
            {tab === 'attribution' && (
              <div className="space-y-3">
                <p className="text-xs text-gray-500 mb-2">
                  Estimated contribution of each retrieval method to the final generated answer,
                  based on word-overlap between retrieved chunks and the answer text.
                </p>
                {sortedAttrib.length === 0
                  ? <p className="text-xs text-gray-500">No overlap found — LM Studio may be offline or answer is empty.</p>
                  : sortedAttrib.map(([method, pct]) => (
                    <div key={method}>
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-xs font-medium" style={{ color: getColor(method) }}>{method}</span>
                        <span className="text-xs font-semibold text-white w-10 text-right">{pct.toFixed(1)}%</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div className="h-2 rounded-full" style={{ width: `${pct}%`, backgroundColor: getColor(method) }} />
                      </div>
                    </div>
                  ))
                }
                {hasAnswer && (
                  <details className="mt-3">
                    <summary className="text-xs text-gray-500 cursor-pointer">Show full answer</summary>
                    <p className="text-xs text-gray-300 mt-1 bg-gray-800 rounded p-2 max-h-40 overflow-y-auto leading-relaxed">{answer}</p>
                  </details>
                )}
              </div>
            )}

            {/* ── GRAPH TRACE tab ─────────────────────────────────────── */}
            {tab === 'graph' && (
              <div className="space-y-4">
                <p className="text-xs text-gray-500">
                  Knowledge Graph activity during this search — entities extracted from the query
                  and traversal paths followed. 0 entities means graph retrieval could not augment results.
                </p>

                {/* Entities */}
                <div>
                  <p className="text-xs font-semibold text-emerald-400 mb-2">
                    🏷️ Entities Extracted from Query ({graphEntities.length})
                  </p>
                  {graphEntities.length === 0 ? (
                    <div className="text-xs text-gray-500 bg-gray-800 rounded p-2">
                      <p className="font-medium text-yellow-400 mb-1">⚠️ No entities extracted</p>
                      <p>Possible causes:</p>
                      <ul className="list-disc list-inside mt-1 space-y-0.5 text-gray-400">
                        <li>Knowledge Graph not enabled (enable_graph=false)</li>
                        <li>Graph snapshot not found — re-ingest with ER enabled</li>
                        <li>spaCy NER found no named entities in the query text</li>
                        <li>LLM Graph Extraction offline (if enable_llm_graph=true)</li>
                      </ul>
                    </div>
                  ) : (
                    <div className="flex flex-wrap gap-1.5">
                      {graphEntities.map((e, i) => (
                        <span key={i} className="text-xs px-2 py-0.5 rounded-full bg-emerald-900/40 text-emerald-300 border border-emerald-700/40">
                          {e}
                        </span>
                      ))}
                    </div>
                  )}
                </div>

                {/* Paths */}
                <div>
                  <p className="text-xs font-semibold text-emerald-400 mb-2">
                    🔗 Graph Traversal Paths ({graphPaths.length})
                  </p>
                  {graphPaths.length === 0 ? (
                    <div className="text-xs text-gray-500 bg-gray-800 rounded p-2">
                      <p className="font-medium text-yellow-400 mb-1">⚠️ No paths traversed</p>
                      <p>Even when entities are extracted, paths may be empty if:</p>
                      <ul className="list-disc list-inside mt-1 space-y-0.5 text-gray-400">
                        <li>Entities are not connected in the knowledge graph</li>
                        <li>Kuzu graph DB has no edges for these entity types</li>
                        <li>Graph was built without entity-relation (ER) extraction</li>
                      </ul>
                    </div>
                  ) : (
                    <div className="space-y-1 max-h-64 overflow-y-auto">
                      {graphPaths.map((path, i) => (
                        <div key={i} className="text-xs text-gray-300 bg-gray-800 rounded px-2 py-1 font-mono break-all">
                          <span className="text-emerald-500 mr-1">{i + 1}.</span>{path}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Graph chunk attribution */}
                {hasChunks && (() => {
                  const graphChunks = chunks.filter(c => c.method_lineage?.some(m => m.method === 'Knowledge Graph'))
                  return (
                    <div>
                      <p className="text-xs font-semibold text-emerald-400 mb-2">
                        📄 Chunks Attributed to Knowledge Graph ({graphChunks.length} / {chunks.length})
                      </p>
                      {graphChunks.length === 0
                        ? <p className="text-xs text-gray-500 bg-gray-800 rounded p-2">No final chunks were sourced via Knowledge Graph traversal.</p>
                        : <div className="space-y-1">
                            {graphChunks.map((c, i) => (
                              <div key={c.chunk_id} className="text-xs border border-emerald-900/50 rounded p-2">
                                <div className="flex justify-between mb-1">
                                  <span className="text-emerald-400 font-medium">Graph chunk #{i + 1}</span>
                                  <span className="text-gray-500 font-mono">{c.score.toFixed(5)}</span>
                                </div>
                                <p className="text-gray-300 line-clamp-2">{c.text}</p>
                              </div>
                            ))}
                          </div>
                      }
                    </div>
                  )
                })()}
              </div>
            )}

            {/* ── QUERY EXPANSION tab ─────────────────────────────────── */}
            {tab === 'query' && (
              <div className="space-y-3">
                <p className="text-xs text-gray-500">
                  Query expansion variants generated before retrieval. Each variant was used
                  to retrieve candidates independently; results were merged before RRF fusion.
                </p>
                {!hasQuery && (
                  <p className="text-xs text-gray-500 bg-gray-800 rounded p-2">
                    No query expansion used. Enable Query Rewriting, HyDE, or Multi-Query in Method Settings.
                  </p>
                )}
                {queryVariants.rewritten && (
                  <VariantBlock icon="✏️" label="Rewritten Query" color="text-orange-400" text={queryVariants.rewritten} />
                )}
                {queryVariants.hyde_text && (
                  <VariantBlock icon="💭" label="HyDE — Hypothetical Document" color="text-indigo-400" text={queryVariants.hyde_text} />
                )}
                {queryVariants.stepback && (
                  <VariantBlock icon="↩️" label="Step-Back Query" color="text-cyan-400" text={queryVariants.stepback} />
                )}
                {queryVariants.paraphrases && queryVariants.paraphrases.length > 0 && (
                  <div>
                    <p className="text-xs font-semibold text-teal-400 mb-1">🔀 Multi-Query Paraphrases ({queryVariants.paraphrases.length})</p>
                    <div className="space-y-1">
                      {queryVariants.paraphrases.map((p, i) => (
                        <div key={i} className="text-xs text-gray-300 bg-gray-800 rounded px-2 py-1">
                          <span className="text-teal-500 mr-1">#{i + 1}</span>{p}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* ── PIPELINE STEPS tab ──────────────────────────────────── */}
            {tab === 'pipeline' && steps.length > 0 && (
              <div className="space-y-2">
                <p className="text-xs text-gray-500 mb-2">Phase-by-phase candidate filtering.</p>
                {steps.map((step, i) => {
                  const maxB = Math.max(...steps.map(s => s.candidates_before), 1)
                  return (
                    <div key={i}>
                      <div className="flex items-center justify-between text-xs text-gray-400 mb-0.5">
                        <span className="font-medium text-gray-300 w-48 truncate">{step.method}</span>
                        <span>{step.candidates_before} → {step.candidates_after}</span>
                      </div>
                      <div className="relative w-full h-3 bg-gray-700 rounded">
                        <div className="absolute h-3 bg-gray-600 rounded" style={{ width: `${(step.candidates_before / maxB) * 100}%` }} />
                        <div className="absolute h-3 bg-brand-500 rounded" style={{ width: `${(step.candidates_after / maxB) * 100}%` }} />
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

// ── Sub-components ─────────────────────────────────────────────────────────────

function MethodCard({ method, stats, expanded, onToggle }: {
  method: string
  stats: MethodContributionStat
  expanded: boolean
  onToggle: () => void
}) {
  const status = stats.status ?? 'disabled'
  const cfg = STATUS_CONFIG[status]
  const color = getColor(method)
  const pct = stats.contribution_pct ?? 0
  const rrf = stats.rrf_boost_total ?? 0
  const unique = stats.unique_chunks_added ?? 0
  const avgRank = stats.avg_rank ?? 0

  return (
    <div className={`border rounded-lg overflow-hidden ${cfg.border} ${cfg.bg}`}>
      <button onClick={onToggle} className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-white/5 transition-colors">
        <span className="text-base leading-none">{cfg.icon}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-semibold" style={{ color }}>{method}</span>
            {status === 'active' && (
              <>
                <span className="text-xs text-gray-400">{stats.chunks_contributed ?? 0} chunks</span>
                <div className="flex-1 max-w-[100px] bg-gray-700 rounded-full h-1.5">
                  <div className="h-1.5 rounded-full" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
                </div>
                <span className="text-xs font-bold text-white">{pct.toFixed(1)}%</span>
                {rrf > 0 && (
                  <span className="text-xs text-blue-400" title="Total RRF score boost added to final chunks">
                    +{rrf.toFixed(4)} RRF
                  </span>
                )}
              </>
            )}
            {status === 'zero' && rrf > 0 && (
              <span className="text-xs text-amber-400" title="SPLADE boosted scores of existing chunks even though it added no new ones">
                +{rrf.toFixed(4)} RRF boost (no new chunks)
              </span>
            )}
            {status !== 'active' && (
              <span className={`text-xs ${cfg.text}`}>{cfg.label}</span>
            )}
          </div>
        </div>
        <span className="text-gray-600 text-xs">{expanded ? '▲' : '▼'}</span>
      </button>
      {expanded && (
        <div className="px-3 pb-3 pt-1 border-t border-white/5 space-y-2">
          {stats.reason && (
            <p className="text-xs text-gray-300 leading-relaxed">{stats.reason}</p>
          )}
          {status !== 'disabled' && (
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-gray-500 mt-1">
              <span>Chunks in result: <span className="text-white">{stats.chunks_contributed ?? 0}</span></span>
              <span>Unique (new): <span className={unique > 0 ? 'text-green-400' : 'text-gray-400'}>{unique}</span></span>
              <span>RRF boost total: <span className={rrf > 0 ? 'text-blue-400' : 'text-gray-400'}>{rrf > 0 ? `+${rrf.toFixed(5)}` : '0'}</span></span>
              <span>Avg rank assigned: <span className="text-white">{avgRank > 0 ? avgRank : '—'}</span></span>
              <span>Candidates before: <span className="text-white">{stats.candidates_before ?? 0}</span></span>
              <span>Candidates after: <span className="text-white">{stats.candidates_after ?? 0}</span></span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function VariantBlock({ icon, label, color, text }: { icon: string; label: string; color: string; text: string }) {
  return (
    <div>
      <p className={`text-xs font-semibold mb-1 ${color}`}>{icon} {label}</p>
      <p className="text-xs text-gray-300 bg-gray-800 rounded px-2 py-1.5 leading-relaxed">{text}</p>
    </div>
  )
}
