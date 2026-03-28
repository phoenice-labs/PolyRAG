import { useState, useEffect } from 'react'
import { useStore, EMBEDDING_MODELS, type EmbeddingModelId } from '../store'
import {
  getLLMProviders,
  getLLMConfig,
  updateLLMConfig,
  testLLMConnection,
  type LLMProvider,
  type LLMConfigUpdate,
} from '../api/config'

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
  const [saved, setSaved] = useState(false)

  // ── LLM Config (API-backed) ──────────────────────────────────────────────────
  const [providers, setProviders] = useState<LLMProvider[]>([])
  const [llmLoading, setLlmLoading] = useState(true)
  const [llmForm, setLlmForm] = useState<LLMConfigUpdate>({
    provider: 'lm_studio',
    base_url: 'http://localhost:1234/v1',
    api_key: '',
    model: '',        // populated from server on mount — no hardcoded model name
    temperature: 0.2,
    max_tokens: 512,
    timeout: 60,
  })
  const [apiKeySet, setApiKeySet] = useState(false)    // tracks server-side key presence
  const [llmSaving, setLlmSaving] = useState(false)
  const [llmSaved, setLlmSaved] = useState(false)
  const [llmError, setLlmError] = useState<string | null>(null)
  const [testStatus, setTestStatus] = useState<'idle' | 'ok' | 'error'>('idle')
  const [testError, setTestError] = useState<string | null>(null)

  // Load providers + current LLM config from backend on mount
  useEffect(() => {
    getLLMProviders().then(setProviders).catch(() => {})
    getLLMConfig()
      .then((cfg) => {
        setLlmForm({
          provider: cfg.provider,
          base_url: cfg.base_url,
          api_key: '',               // server never returns the real key
          model: cfg.model,
          temperature: cfg.temperature,
          max_tokens: cfg.max_tokens,
          timeout: cfg.timeout,
        })
        setApiKeySet(cfg.api_key_set)
      })
      .catch(() => {})
      .finally(() => setLlmLoading(false))
  }, [])

  // Auto-fill default base_url when provider changes
  const handleProviderChange = (providerId: string) => {
    const prov = providers.find((p) => p.id === providerId)
    setLlmForm((f) => ({
      ...f,
      provider: providerId,
      base_url: prov?.default_base_url ?? f.base_url,
    }))
  }

  const handleLlmSave = async () => {
    setLlmSaving(true)
    setLlmError(null)
    try {
      const result = await updateLLMConfig(llmForm)
      setApiKeySet(result.api_key_set)
      setLlmSaved(true)
      setTimeout(() => setLlmSaved(false), 2000)
    } catch (e: unknown) {
      const err = e as { response?: { data?: { detail?: string } }; message?: string }
      setLlmError(err?.response?.data?.detail ?? err?.message ?? 'Failed to save')
    } finally {
      setLlmSaving(false)
    }
  }

  const handleTestConnection = async () => {
    setTestStatus('idle')
    setTestError(null)
    try {
      const result = await testLLMConnection()
      setTestStatus(result.reachable ? 'ok' : 'error')
      setTestError(result.error ?? null)
    } catch {
      setTestStatus('error')
      setTestError('Request failed')
    }
  }

  const selectedProvider = providers.find((p) => p.id === llmForm.provider)

  const updateBackend = (name: string, field: keyof BackendConfig, value: string) => {
    setBackendConfigs((prev) => ({ ...prev, [name]: { ...prev[name], [field]: value } }))
  }

  const handleSave = () => {
    const current = loadSettings()
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ ...current, backends: backendConfigs }))
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
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
        <button
          onClick={handleSave}
          className="px-4 py-2 bg-brand-500 hover:bg-sky-600 text-white rounded text-sm font-medium"
        >
          {saved ? '✓ Saved!' : 'Save Backend Settings'}
        </button>
      </div>

      {/* ── LLM Provider (API-backed) ─────────────────────────────────────────── */}
      <div className="bg-gray-900 rounded-lg p-4 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-sm font-semibold text-gray-300">LLM Provider</h2>
            <p className="text-xs text-gray-500 mt-1">
              Saved to server — takes effect immediately on all RAG queries without restart.
            </p>
          </div>
          {llmLoading && <span className="text-xs text-gray-500 animate-pulse">Loading…</span>}
        </div>

        {/* Provider selector */}
        <div>
          <label className="text-xs text-gray-400">Provider</label>
          <select
            value={llmForm.provider}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-brand-500"
          >
            {providers.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
          {selectedProvider && (
            <p className="text-xs text-gray-500 mt-1">{selectedProvider.notes}</p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-gray-400">Base URL</label>
            <input
              value={llmForm.base_url}
              onChange={(e) => setLlmForm((f) => ({ ...f, base_url: e.target.value }))}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400">Model</label>
            <input
              value={llmForm.model}
              onChange={(e) => setLlmForm((f) => ({ ...f, model: e.target.value }))}
              placeholder="e.g. mistralai/ministral-3b or llama-3-8b"
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>

        {/* API Key — only shown for providers that require one */}
        {selectedProvider?.requires_api_key && (
          <div>
            <label className="text-xs text-gray-400">
              API Key
              {apiKeySet && <span className="ml-2 text-green-400">✓ key stored</span>}
            </label>
            <input
              type="password"
              value={llmForm.api_key}
              onChange={(e) => setLlmForm((f) => ({ ...f, api_key: e.target.value }))}
              placeholder={apiKeySet ? '(leave blank to keep existing key)' : 'sk-...'}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
        )}

        <div className="grid grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-gray-400">Temperature</label>
            <input
              type="number"
              min={0} max={2} step={0.05}
              value={llmForm.temperature}
              onChange={(e) => setLlmForm((f) => ({ ...f, temperature: parseFloat(e.target.value) || 0 }))}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400">Max Tokens</label>
            <input
              type="number"
              min={64} max={8192} step={64}
              value={llmForm.max_tokens}
              onChange={(e) => setLlmForm((f) => ({ ...f, max_tokens: parseInt(e.target.value) || 512 }))}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400">Timeout (s)</label>
            <input
              type="number"
              min={5} max={600}
              value={llmForm.timeout}
              onChange={(e) => setLlmForm((f) => ({ ...f, timeout: parseInt(e.target.value) || 60 }))}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>

        {llmError && (
          <p className="text-red-400 text-xs">{llmError}</p>
        )}

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={handleLlmSave}
            disabled={llmSaving}
            className="px-4 py-2 bg-brand-500 hover:bg-sky-600 disabled:opacity-50 text-white rounded text-sm font-medium"
          >
            {llmSaved ? '✓ Saved!' : llmSaving ? 'Saving…' : 'Save LLM Config'}
          </button>
          <button
            onClick={handleTestConnection}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-gray-200 rounded text-sm"
          >
            Test Connection
          </button>
          {testStatus === 'ok' && <span className="text-green-400 text-sm">✓ Connected</span>}
          {testStatus === 'error' && (
            <span className="text-red-400 text-sm">✗ {testError ?? 'Failed'}</span>
          )}
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
    </div>
  )
}

