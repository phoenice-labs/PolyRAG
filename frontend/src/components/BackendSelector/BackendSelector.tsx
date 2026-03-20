import { useStore } from '../../store'

const BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']

const BACKEND_COLORS: Record<string, string> = {
  faiss: 'bg-blue-500',
  chromadb: 'bg-purple-500',
  qdrant: 'bg-red-500',
  weaviate: 'bg-green-500',
  milvus: 'bg-yellow-500',
  pgvector: 'bg-indigo-500',
}

function StatusDot({ status }: { status: 'ok' | 'error' | 'unknown' }) {
  const colors = { ok: 'bg-green-400', error: 'bg-red-400', unknown: 'bg-gray-500' }
  return (
    <span
      data-testid={`status-dot-${status}`}
      className={`inline-block w-2 h-2 rounded-full ${colors[status]}`}
    />
  )
}

export default function BackendSelector() {
  const { selectedBackends, setSelectedBackends, backendStatuses } = useStore()

  const toggle = (name: string) => {
    if (selectedBackends.includes(name)) {
      setSelectedBackends(selectedBackends.filter((b) => b !== name))
    } else {
      setSelectedBackends([...selectedBackends, name])
    }
  }

  return (
    <div className="space-y-1">
      <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Backends</div>
      {BACKENDS.map((name) => {
        const selected = selectedBackends.includes(name)
        const status = (backendStatuses[name] as 'ok' | 'error' | 'unknown') ?? 'unknown'
        return (
          <label
            key={name}
            className="flex items-center gap-2 cursor-pointer group"
          >
            <input
              type="checkbox"
              checked={selected}
              onChange={() => toggle(name)}
              className="rounded border-gray-600 bg-gray-800 text-brand-500"
              aria-label={name}
            />
            <span
              className={`w-2 h-2 rounded-sm ${BACKEND_COLORS[name]}`}
            />
            <span className="text-sm text-gray-300 group-hover:text-white flex-1 capitalize">
              {name}
            </span>
            <StatusDot status={status} />
          </label>
        )
      })}
    </div>
  )
}
