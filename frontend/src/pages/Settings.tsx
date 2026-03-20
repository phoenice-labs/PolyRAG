import { useState } from 'react'
import { useStore, EMBEDDING_MODELS, type EmbeddingModelId } from '../store'

interface BackendConfig {
  host: string
  port: string
  apiKey: string
}

const BACKENDS = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']
const STORAGE_KEY = 'polyrag-settings'

function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
  } catch {
    return {}
  }
}

export default function Settings() {
  const { embeddingModel, setEmbeddingModel } = useStore()

  const [backendConfigs, setBackendConfigs] = useState<Record<string, BackendConfig>>(() => {
    const saved = loadSettings()
    const defaults: Record<string, BackendConfig> = {}
    BACKENDS.forEach((b) => { defaults[b] = saved.backends?.[b] ?? { host: 'localhost', port: '', apiKey: '' } })
    return defaults
  })
  const [lmUrl, setLmUrl] = useState(() => loadSettings().lmUrl ?? 'http://localhost:1234')
  const [lmModel, setLmModel] = useState(() => loadSettings().lmModel ?? '')
  const [saved, setSaved] = useState(false)
  const [lmStatus, setLmStatus] = useState<'idle' | 'ok' | 'error'>('idle')

  const updateBackend = (name: string, field: keyof BackendConfig, value: string) => {
    setBackendConfigs((prev) => ({ ...prev, [name]: { ...prev[name], [field]: value } }))
  }

  const handleSave = () => {
    const current = loadSettings()
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...current, backends: backendConfigs, lmUrl, lmModel }))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const testLmStudio = async () => {
    setLmStatus('idle')
    try {
      const res = await fetch(`${lmUrl}/v1/models`)
      setLmStatus(res.ok ? 'ok' : 'error')
    } catch {
      setLmStatus('error')
    }
  }

  return (
    <div className="space-y-6 max-w-2xl">
      <h1 className="text-xl font-semibold">Settings</h1>

      <div className="bg-gray-900 rounded-lg p-4 space-y-4">
        <h2 className="text-sm font-semibold text-gray-300">Backend Connections</h2>
        {BACKENDS.map((name) => {
          const cfg = backendConfigs[name]
          return (
            <div key={name} className="space-y-2">
              <div className="text-xs text-gray-400 font-medium uppercase">{name}</div>
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <label className="text-xs text-gray-500">Host</label>
                  <input
                    value={cfg.host}
                    onChange={(e) => updateBackend(name, 'host', e.target.value)}
                    className="w-full mt-0.5 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500">Port</label>
                  <input
                    value={cfg.port}
                    onChange={(e) => updateBackend(name, 'port', e.target.value)}
                    placeholder="default"
                    className="w-full mt-0.5 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-500">API Key</label>
                  <input
                    value={cfg.apiKey}
                    onChange={(e) => updateBackend(name, 'apiKey', e.target.value)}
                    type="password"
                    placeholder="optional"
                    className="w-full mt-0.5 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <div className="bg-gray-900 rounded-lg p-4 space-y-3">
        <h2 className="text-sm font-semibold text-gray-300">LM Studio</h2>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-400">URL</label>
            <input
              value={lmUrl}
              onChange={(e) => setLmUrl(e.target.value)}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400">Model Name</label>
            <input
              value={lmModel}
              onChange={(e) => setLmModel(e.target.value)}
              placeholder="e.g. llama-3-8b"
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={testLmStudio}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm"
          >
            Test Connection
          </button>
          {lmStatus === 'ok' && <span className="text-green-400 text-sm">✓ Connected</span>}
          {lmStatus === 'error' && <span className="text-red-400 text-sm">✗ Failed</span>}
        </div>
      </div>

      {/* ── Embedding Model ─────────────────────────────────────────────────── */}
      <div className="bg-gray-900 rounded-lg p-4 space-y-3">
        <div>
          <h2 className="text-sm font-semibold text-gray-300">Embedding Model</h2>
          <p className="text-xs text-gray-500 mt-1">
            Each model uses its own isolated collection — switching model requires re-ingestion.
            BGE models are downloaded on first use (cached locally).
          </p>
        </div>
        <select
          value={embeddingModel}
          onChange={(e) => setEmbeddingModel(e.target.value as EmbeddingModelId)}
          className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-brand-500"
        >
          {EMBEDDING_MODELS.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
        <div className="text-xs text-gray-500 flex gap-4">
          <span>Selected: <span className="text-gray-300">{embeddingModel}</span></span>
          <span>Collection suffix: <span className="text-sky-400">
            {embeddingModel === 'all-MiniLM-L6-v2' ? 'minilm'
              : embeddingModel === 'BAAI/bge-base-en-v1.5' ? 'bge-base'
              : 'bge-large'}
          </span></span>
        </div>
      </div>

      <button
        onClick={handleSave}
        className="px-4 py-2 bg-brand-500 hover:bg-sky-600 text-white rounded text-sm font-medium"
      >
        {saved ? '✓ Saved!' : 'Save Settings'}
      </button>
    </div>
  )
}
