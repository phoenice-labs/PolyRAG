interface IngestionFlowProps {
  activeStep?: string
  completedSteps?: Set<string>
}

const STEPS = [
  { id: 'upload', label: '📁 Upload' },
  { id: 'chunk',  label: '✂ Chunk'  },
  { id: 'embed',  label: '🧬 Embed'  },
  { id: 'graph',  label: '🕸 KG'     },
  { id: 'upsert', label: '💾 Upsert' },
]

export default function IngestionFlow({ activeStep, completedSteps }: IngestionFlowProps) {
  return (
    <div className="flex items-center gap-0 w-full py-4 px-2 bg-gray-900 rounded border border-gray-700 overflow-x-auto">
      {STEPS.map((step, i) => {
        const isActive    = activeStep === step.id
        const isCompleted = completedSteps?.has(step.id) && !isActive
        return (
          <div key={step.id} className="flex items-center min-w-0 flex-1">
            <div
              className={`flex-1 text-center px-3 py-2 rounded-lg border text-sm font-medium transition-all duration-300 whitespace-nowrap ${
                isActive
                  ? 'border-sky-500 bg-sky-500/20 text-white shadow-lg shadow-sky-500/20 animate-pulse'
                  : isCompleted
                  ? 'border-green-600 bg-green-600/15 text-green-300'
                  : 'border-gray-600 bg-gray-800 text-gray-400'
              }`}
            >
              {isCompleted ? '✓ ' : ''}{step.label}
            </div>
            {i < STEPS.length - 1 && (
              <div className={`flex-shrink-0 mx-1 text-lg transition-colors duration-300 ${
                isCompleted ? 'text-green-500' : isActive ? 'text-sky-500' : 'text-gray-600'
              }`}>
                →
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

