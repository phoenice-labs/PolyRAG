import { useState } from 'react'
import { api } from '../../api/client'
import type { SearchResultItem } from '../../api/search'

interface ResultCardProps {
  result: SearchResultItem
  backend?: string
}

const BACKEND_BADGE_COLORS: Record<string, string> = {
  faiss: 'bg-blue-600', chromadb: 'bg-purple-600', qdrant: 'bg-red-600',
  weaviate: 'bg-green-600', milvus: 'bg-yellow-600', pgvector: 'bg-indigo-600',
}

// Colour coding per retrieval signal type
const METHOD_COLORS: Record<string, string> = {
  'Dense Vector':      'bg-blue-700 text-blue-100',
  'BM25 Keyword':      'bg-amber-700 text-amber-100',
  'Knowledge Graph':   'bg-emerald-700 text-emerald-100',
  'RAPTOR':            'bg-purple-700 text-purple-100',
  'Multi-Query':       'bg-rose-700 text-rose-100',
  'MMR Diversity':     'bg-teal-700 text-teal-100',
  'Cross-Encoder Rerank': 'bg-orange-700 text-orange-100',
  'Contextual Rerank': 'bg-pink-700 text-pink-100',
}
const DEFAULT_METHOD_COLOR = 'bg-gray-700 text-gray-200'

export default function ResultCard({ result, backend }: ResultCardProps) {
  const [expanded, setExpanded] = useState(false)
  const [metaOpen, setMetaOpen] = useState(false)
  const [lineageOpen, setLineageOpen] = useState(false)
  const [feedback, setFeedback] = useState<'up' | 'down' | null>(null)

  const meta = result.metadata ?? {}
  const badgeColor = BACKEND_BADGE_COLORS[backend ?? (meta.backend as string) ?? ''] ?? 'bg-gray-600'
  const scorePercent = Math.min(100, Math.round(Math.abs(result.score) * 100))
  const displayText = expanded ? result.text : result.text.slice(0, 220) + (result.text.length > 220 ? '...' : '')

  const lineage = result.method_lineage ?? []
  const postProcessors = result.post_processors ?? []
  const hasLineage = lineage.length > 0 || postProcessors.length > 0

  const handleFeedback = async (type: 'up' | 'down') => {
    setFeedback(type)
    try {
      await api.post('/feedback', { chunk_id: result.chunk_id, feedback: type })
    } catch { /* ignore */ }
  }

  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-2.5 border border-gray-700 hover:border-gray-500 transition-colors">
      {/* Header: backend badge + score bar */}
      <div className="flex items-center justify-between">
        {backend && (
          <span className={`text-xs px-2 py-0.5 rounded ${badgeColor} text-white font-medium`}>
            {backend}
          </span>
        )}
        <div className="flex items-center gap-2 text-xs text-gray-400 ml-auto">
          <span>Score</span>
          <div className="w-20 bg-gray-700 rounded-full h-1.5" data-testid="score-bar">
            <div className="bg-sky-500 h-1.5 rounded-full" style={{ width: `${scorePercent}%` }} />
          </div>
          <span className="tabular-nums">{result.score.toFixed(3)}</span>
        </div>
      </div>

      {/* Method lineage pills — always visible at a glance */}
      {hasLineage && (
        <div className="flex flex-wrap gap-1 items-center">
          {lineage.map((c, i) => (
            <span key={i}
              className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${METHOD_COLORS[c.method] ?? DEFAULT_METHOD_COLOR}`}
              title={`Rank #${c.rank} · RRF +${c.rrf_contribution.toFixed(5)}`}
            >
              {c.method} #{c.rank}
            </span>
          ))}
          {postProcessors.map((p, i) => (
            <span key={`pp-${i}`}
              className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${METHOD_COLORS[p] ?? DEFAULT_METHOD_COLOR}`}
              title="Post-processor (reranker)"
            >
              ↻ {p}
            </span>
          ))}
        </div>
      )}

      {/* Text */}
      <p className="text-sm text-gray-200 leading-relaxed">{displayText}</p>
      {result.text.length > 220 && (
        <button onClick={() => setExpanded(!expanded)} className="text-xs text-sky-500 hover:underline">
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}

      {/* Lineage detail */}
      {hasLineage && (
        <button onClick={() => setLineageOpen(!lineageOpen)} className="text-xs text-emerald-500 hover:text-emerald-300 flex items-center gap-1">
          <span>{lineageOpen ? '▼' : '▶'}</span> Method Lineage ({lineage.length} signal{lineage.length !== 1 ? 's' : ''}{postProcessors.length > 0 ? ` + ${postProcessors.length} reranker` : ''})
        </button>
      )}

      {lineageOpen && (
        <div className="bg-gray-900 rounded p-3 space-y-2">
          <div className="text-[10px] font-semibold text-gray-500 uppercase tracking-wide">RRF Signal Contributions</div>
          {/* Mini bar chart */}
          {lineage.length > 0 && (() => {
            const maxContrib = Math.max(...lineage.map((c) => c.rrf_contribution), 0.0001)
            return lineage.map((c, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium shrink-0 ${METHOD_COLORS[c.method] ?? DEFAULT_METHOD_COLOR}`}>
                  {c.method}
                </span>
                <div className="flex-1 bg-gray-700 rounded-full h-2">
                  <div
                    className="h-2 rounded-full bg-emerald-500"
                    style={{ width: `${Math.round((c.rrf_contribution / maxContrib) * 100)}%` }}
                  />
                </div>
                <span className="text-gray-400 tabular-nums shrink-0">
                  rank #{c.rank} · +{c.rrf_contribution.toFixed(5)}
                </span>
              </div>
            ))
          })()}
          {postProcessors.length > 0 && (
            <div className="pt-1 border-t border-gray-700">
              <div className="text-[10px] text-gray-500 mb-1">Post-processors (reranked final order):</div>
              <div className="flex flex-wrap gap-1">
                {postProcessors.map((p, i) => (
                  <span key={i} className={`text-[10px] px-1.5 py-0.5 rounded ${METHOD_COLORS[p] ?? DEFAULT_METHOD_COLOR}`}>↻ {p}</span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Metadata toggle */}
      <button onClick={() => setMetaOpen(!metaOpen)} className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1">
        <span>{metaOpen ? '▼' : '▶'}</span> Provenance & Metadata
      </button>

      {metaOpen && (
        <div className="text-xs space-y-1 text-gray-400 pl-3 border-l border-gray-600 bg-gray-900/40 rounded p-2">
          <div><span className="text-gray-500">Chunk ID:</span> <span className="font-mono">{result.chunk_id}</span></div>
          {result.provenance && <div><span className="text-gray-500">Provenance:</span> {result.provenance}</div>}
          {result.confidence != null && (
            <div><span className="text-gray-500">Confidence:</span> {(result.confidence * 100).toFixed(1)}%</div>
          )}
          {Object.entries(meta).map(([k, v]) => (
            <div key={k}><span className="text-gray-500">{k}:</span> {String(v)}</div>
          ))}
        </div>
      )}

      {/* Feedback */}
      <div className="flex gap-1.5">
        <button onClick={() => handleFeedback('up')} aria-label="thumbs up"
          className={`text-sm px-2 py-0.5 rounded ${feedback === 'up' ? 'bg-green-700 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>
          👍
        </button>
        <button onClick={() => handleFeedback('down')} aria-label="thumbs down"
          className={`text-sm px-2 py-0.5 rounded ${feedback === 'down' ? 'bg-red-700 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}>
          👎
        </button>
      </div>
    </div>
  )
}


