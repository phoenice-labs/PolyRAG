import { useState } from 'react'
import type { RetrievalStep } from '../../api/search'

interface RetrievalTraceProps {
  steps: RetrievalStep[]
}

const MAX_WIDTH = 280

export default function RetrievalTrace({ steps }: RetrievalTraceProps) {
  const [open, setOpen] = useState(false)

  if (!steps || steps.length === 0) return null

  const maxCandidates = Math.max(...steps.map((s) => s.candidates_before), 1)

  return (
    <div className="border border-gray-700 rounded-lg bg-gray-900">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between p-3 text-sm text-gray-300 hover:text-white"
      >
        <span>🔍 Retrieval Trace ({steps.length} steps)</span>
        <span>{open ? '▼' : '▶'}</span>
      </button>
      {open && (
        <div className="p-4 space-y-3 border-t border-gray-700">
          <svg width={MAX_WIDTH + 100} height={steps.length * 50 + 20} className="overflow-visible">
            {steps.map((step, i) => {
              const beforeW = (step.candidates_before / maxCandidates) * MAX_WIDTH
              const afterW = (step.candidates_after / maxCandidates) * MAX_WIDTH
              const y = i * 50 + 10
              return (
                <g key={i}>
                  <rect x={0} y={y} width={beforeW} height={20} rx={3} className="fill-blue-800" />
                  <rect x={0} y={y} width={afterW} height={20} rx={3} className="fill-brand-500" />
                  <text x={beforeW + 6} y={y + 14} className="fill-gray-400 text-xs" fontSize={11}>
                    {step.method}: {step.candidates_before}→{step.candidates_after}
                  </text>
                </g>
              )
            })}
          </svg>
          <div className="space-y-1">
            {steps.map((step, i) => (
              <div key={i} className="flex items-center gap-3 text-xs text-gray-400">
                <span className="text-gray-500 w-24 truncate">{step.method}</span>
                <span>{step.candidates_before} → {step.candidates_after}</span>
                {step.score_min !== undefined && step.score_max !== undefined && (
                  <span className="text-gray-600">
                    score: [{step.score_min.toFixed(3)}, {step.score_max.toFixed(3)}]
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
