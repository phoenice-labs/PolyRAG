/**
 * PromptEditor — manage all LLM prompts from the UI without code changes.
 *
 * Shows each prompt with:
 *   - Which retrieval method it belongs to (method_name, method_id)
 *   - Where in the pipeline it's called (pipeline_stage)
 *   - What it does (description)
 *   - Editable prompt template (textarea + Save + Reset buttons)
 *
 * Changes persist to config/prompts.yaml and take effect on the next search.
 */
import { useEffect, useState } from 'react'
import { api } from '../api/client'

interface PromptEntry {
  key: string
  method_name: string
  method_id: number
  pipeline_stage: string
  description: string
  template: string
}

const STAGE_COLORS: Record<string, string> = {
  'Query Expansion':          'bg-blue-900 text-blue-300',
  'Re-ranking':               'bg-purple-900 text-purple-300',
  'Index Build (Post-Ingestion)': 'bg-amber-900 text-amber-300',
  'Index Build (Ingestion)':  'bg-orange-900 text-orange-300',
  'Answer Synthesis':         'bg-green-900 text-green-300',
}

function stageColor(stage: string): string {
  return STAGE_COLORS[stage] ?? 'bg-gray-800 text-gray-300'
}

// ── Prompt Editor Guide Modal ─────────────────────────────────────────────────

function PromptGuideModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[620px] max-h-[88vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg">✏️</span>
            <h2 className="text-base font-semibold text-white">Prompt Editor — Guide</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* Purpose */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">What is the Prompt Editor?</h3>
            <p className="text-gray-300 leading-relaxed">
              The Prompt Editor lets you <span className="text-white font-medium">customise every LLM instruction</span> used in
              the retrieval pipeline — directly from the UI, without touching any code. Changes persist to{' '}
              <code className="text-sky-400 bg-gray-800 px-1 rounded text-xs">config/prompts.yaml</code> and take effect
              on the very next search, with no restart required.
            </p>
          </section>

          {/* What prompts exist */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Prompts & Pipeline Stages</h3>
            <div className="space-y-1.5 text-xs">
              {[
                { stage: 'Query Expansion', color: 'bg-blue-900 text-blue-300', methods: ['Query Rewrite', 'Multi-Query', 'HyDE'], desc: 'Prompts that transform the user query before retrieval to improve recall.' },
                { stage: 'Re-ranking', color: 'bg-purple-900 text-purple-300', methods: ['Contextual Rerank'], desc: 'Prompts that score or re-order retrieved chunks by relevance to the query.' },
                { stage: 'Index Build (Ingestion)', color: 'bg-orange-900 text-orange-300', methods: ['RAPTOR'], desc: 'Prompts used during document ingestion to build hierarchical summaries.' },
                { stage: 'Answer Synthesis', color: 'bg-green-900 text-green-300', methods: ['LLM Graph Extract', 'LLM Entity Extraction'], desc: 'Prompts that extract entities, relations, or generate final answers from retrieved context.' },
              ].map(({ stage, color, methods, desc }) => (
                <div key={stage} className="bg-gray-800 rounded p-2.5 border border-gray-700">
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`text-xs px-2 py-0.5 rounded font-medium ${color}`}>{stage}</span>
                    <span className="text-gray-500">→ {methods.join(', ')}</span>
                  </div>
                  <div className="text-gray-400 leading-relaxed">{desc}</div>
                </div>
              ))}
            </div>
          </section>

          {/* How to edit */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">How to Edit a Prompt</h3>
            <ol className="space-y-2 text-gray-300 leading-relaxed text-xs">
              {[
                'Find the method you want to customise in the Retrieval Methods table at the top, then click "Edit ▼" to expand it. Alternatively, click any method card directly.',
                'Read the Description to understand what the prompt does and what variables it receives (e.g., {query}, {context}, {chunks}).',
                'Edit the System Prompt in the text area. The card turns amber and shows "unsaved" while there are uncommitted changes.',
                'Click Save to persist changes to prompts.yaml. The confirmation message confirms the change is live.',
                'If a prompt breaks retrieval, click "Reset to Default" to restore the original factory prompt instantly.',
              ].map((text, i) => (
                <li key={i} className="flex gap-3">
                  <span className="w-5 h-5 rounded-full bg-brand-500 text-white text-xs flex items-center justify-center shrink-0 mt-0.5">{i + 1}</span>
                  <span>{text}</span>
                </li>
              ))}
            </ol>
          </section>

          {/* Template variables */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Common Template Variables</h3>
            <div className="space-y-1.5 text-xs">
              {[
                { var: '{query}', desc: 'The original user query exactly as typed.' },
                { var: '{context}', desc: 'The retrieved chunks, joined and formatted, passed to the LLM as context.' },
                { var: '{chunks}', desc: 'List of individual chunk texts (used in re-ranking and contextual prompts).' },
                { var: '{entities}', desc: 'Named entities extracted from the query or chunks (graph prompts).' },
                { var: '{summary}', desc: 'Existing RAPTOR summary node text (used during hierarchical index build).' },
              ].map(({ var: v, desc }) => (
                <div key={v} className="flex gap-2">
                  <code className="text-sky-400 bg-gray-800 px-1.5 py-0.5 rounded shrink-0 font-mono">{v}</code>
                  <span className="text-gray-400">{desc}</span>
                </div>
              ))}
            </div>
            <p className="text-xs text-yellow-400 mt-2">⚠ Always keep required variables in your edited prompt — removing them will break the corresponding retrieval method.</p>
          </section>

          {/* Tips */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Tips</h3>
            <ul className="space-y-1.5 text-xs text-gray-400 list-disc list-inside">
              <li>Changes take effect <span className="text-gray-200">immediately on the next search</span> — no restart, no reload needed.</li>
              <li>Copy the default prompt text somewhere safe before editing — "Reset to Default" is your safety net.</li>
              <li>For Query Rewrite prompts, be explicit about the <span className="text-gray-200">desired output format</span> (e.g., "Return only the rewritten query, no explanation").</li>
              <li>For Contextual Rerank prompts, instruct the LLM to return a <span className="text-gray-200">numeric score 0–1</span> only — extra text breaks the parser.</li>
              <li>Test changes on the <span className="text-gray-200">Search Lab</span> after saving — compare results before and after to verify improvement.</li>
            </ul>
          </section>

        </div>
      </div>
    </div>
  )
}

export default function PromptEditor() {
  const [prompts, setPrompts] = useState<PromptEntry[]>([])
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [saving, setSaving] = useState<Record<string, boolean>>({})
  const [resetting, setResetting] = useState<Record<string, boolean>>({})
  const [status, setStatus] = useState<Record<string, 'saved' | 'reset' | 'error' | null>>({})
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [expandedKey, setExpandedKey] = useState<string | null>(null)
  const [showGuide, setShowGuide] = useState(false)

  useEffect(() => {
    api.get<PromptEntry[]>('/prompts')
      .then((r) => {
        setPrompts(r.data)
        const initial: Record<string, string> = {}
        r.data.forEach((p) => { initial[p.key] = p.template })
        setDrafts(initial)
      })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  const handleSave = async (key: string) => {
    setSaving((s) => ({ ...s, [key]: true }))
    setStatus((s) => ({ ...s, [key]: null }))
    try {
      await api.put(`/prompts/${key}`, { template: drafts[key] })
      setPrompts((prev) => prev.map((p) => p.key === key ? { ...p, template: drafts[key] } : p))
      setStatus((s) => ({ ...s, [key]: 'saved' }))
    } catch {
      setStatus((s) => ({ ...s, [key]: 'error' }))
    } finally {
      setSaving((s) => ({ ...s, [key]: false }))
      setTimeout(() => setStatus((s) => ({ ...s, [key]: null })), 3000)
    }
  }

  const handleReset = async (key: string) => {
    setResetting((s) => ({ ...s, [key]: true }))
    setStatus((s) => ({ ...s, [key]: null }))
    try {
      const r = await api.post<{ template: string }>(`/prompts/${key}/reset`)
      const defaultTemplate = r.data.template
      setDrafts((d) => ({ ...d, [key]: defaultTemplate }))
      setPrompts((prev) => prev.map((p) => p.key === key ? { ...p, template: defaultTemplate } : p))
      setStatus((s) => ({ ...s, [key]: 'reset' }))
    } catch {
      setStatus((s) => ({ ...s, [key]: 'error' }))
    } finally {
      setResetting((s) => ({ ...s, [key]: false }))
      setTimeout(() => setStatus((s) => ({ ...s, [key]: null })), 3000)
    }
  }

  const isDirty = (key: string, template: string) => drafts[key] !== template

  if (loading) return <div className="text-gray-500 text-sm mt-8 text-center">Loading prompts…</div>
  if (error)   return <div className="text-red-400 text-sm mt-8 text-center">Error: {error}</div>

  return (
    <div className="max-w-4xl mx-auto space-y-4">
      <div className="mb-6 flex items-start gap-3">
        <div>
          <h1 className="text-xl font-bold text-gray-100 mb-1">Prompt Editor</h1>
          <p className="text-sm text-gray-400">
            Edit LLM prompts for each retrieval method. Changes persist to{' '}
            <code className="text-sky-400 bg-gray-800 px-1 rounded">config/prompts.yaml</code>{' '}
            and take effect on the next search — no restart required.
          </p>
        </div>
        <button
          onClick={() => setShowGuide(true)}
          title="How does the Prompt Editor work?"
          className="w-6 h-6 mt-0.5 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-300 hover:text-white text-xs font-bold flex items-center justify-center transition-colors shrink-0"
        >
          ?
        </button>
      </div>
      {showGuide && <PromptGuideModal onClose={() => setShowGuide(false)} />}

      {/* Method → Prompt mapping table */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden mb-6">
        <div className="px-4 py-3 border-b border-gray-800 text-xs font-semibold text-gray-400 uppercase tracking-wider">
          Retrieval Methods → Prompts
        </div>
        <table className="w-full text-sm">
          <thead className="bg-gray-800/50">
            <tr>
              <th className="text-left px-4 py-2 text-xs text-gray-500">Method</th>
              <th className="text-left px-4 py-2 text-xs text-gray-500">Stage</th>
              <th className="text-left px-4 py-2 text-xs text-gray-500">Description</th>
              <th className="text-left px-4 py-2 text-xs text-gray-500">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {prompts.map((p) => (
              <tr key={p.key} className="hover:bg-gray-800/30 transition-colors">
                <td className="px-4 py-3">
                  <div className="font-medium text-gray-200">{p.method_name}</div>
                  {p.method_id > 0 && (
                    <div className="text-xs text-gray-600">Method #{p.method_id}</div>
                  )}
                </td>
                <td className="px-4 py-3">
                  <span className={`text-xs px-2 py-0.5 rounded font-medium ${stageColor(p.pipeline_stage)}`}>
                    {p.pipeline_stage}
                  </span>
                </td>
                <td className="px-4 py-3 text-xs text-gray-400 max-w-xs">{p.description.slice(0, 120)}…</td>
                <td className="px-4 py-3">
                  <button
                    onClick={() => setExpandedKey(expandedKey === p.key ? null : p.key)}
                    className="text-xs text-sky-400 hover:text-sky-300 underline"
                  >
                    {expandedKey === p.key ? 'Close ▲' : 'Edit ▼'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Prompt editors */}
      {prompts.map((p) => (
        <div
          key={p.key}
          className={`bg-gray-900 rounded-lg border transition-all ${
            expandedKey === p.key ? 'border-sky-700' : 'border-gray-800'
          }`}
        >
          {/* Header — always visible */}
          <button
            onClick={() => setExpandedKey(expandedKey === p.key ? null : p.key)}
            className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-800/30 transition-colors"
          >
            <span className="text-sm font-semibold text-gray-200 flex-1">{p.method_name}</span>
            <span className={`text-xs px-2 py-0.5 rounded font-medium ${stageColor(p.pipeline_stage)}`}>
              {p.pipeline_stage}
            </span>
            {isDirty(p.key, p.template) && (
              <span className="text-xs text-amber-400 bg-amber-900/30 px-2 py-0.5 rounded">unsaved</span>
            )}
            <span className="text-gray-600 text-xs">{expandedKey === p.key ? '▲' : '▼'}</span>
          </button>

          {/* Body — only when expanded */}
          {expandedKey === p.key && (
            <div className="px-4 pb-4 border-t border-gray-800">
              {/* Description */}
              <div className="mt-3 mb-3 text-xs text-gray-400 leading-relaxed">{p.description}</div>

              {/* Prompt textarea */}
              <label className="block text-xs text-gray-500 mb-1">System Prompt</label>
              <textarea
                className="w-full bg-gray-800 text-gray-200 rounded border border-gray-700 px-3 py-2 text-xs font-mono
                           focus:outline-none focus:border-sky-500 resize-y min-h-32"
                rows={8}
                value={drafts[p.key] ?? p.template}
                onChange={(e) => setDrafts((d) => ({ ...d, [p.key]: e.target.value }))}
              />

              {/* Action row */}
              <div className="flex items-center gap-3 mt-2">
                <button
                  onClick={() => handleSave(p.key)}
                  disabled={saving[p.key] || !isDirty(p.key, p.template)}
                  className="px-3 py-1.5 bg-sky-600 hover:bg-sky-500 disabled:opacity-40 text-white rounded text-xs font-medium"
                >
                  {saving[p.key] ? 'Saving…' : 'Save'}
                </button>
                <button
                  onClick={() => handleReset(p.key)}
                  disabled={resetting[p.key]}
                  className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 disabled:opacity-40 text-gray-300 rounded text-xs"
                >
                  {resetting[p.key] ? 'Resetting…' : 'Reset to Default'}
                </button>

                {status[p.key] === 'saved' && (
                  <span className="text-xs text-green-400">✓ Saved — takes effect on next search</span>
                )}
                {status[p.key] === 'reset' && (
                  <span className="text-xs text-amber-400">↺ Reset to factory default</span>
                )}
                {status[p.key] === 'error' && (
                  <span className="text-xs text-red-400">✕ Save failed — check server logs</span>
                )}
              </div>
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
