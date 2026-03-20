import { useState } from 'react'
import { useStore } from '../../store'

const METHOD_INFO: Record<string, string> = {
  'Dense Vector':
    'Embeds your query into a high-dimensional vector and retrieves the closest chunks by cosine similarity. Best at capturing semantic meaning — finds relevant passages even when exact keywords differ.',
  'BM25 Keyword':
    'Classic probabilistic keyword search (Best Match 25). Scores documents by term frequency and inverse document frequency. Excellent for exact-word matches, names, codes, and specific terminology.',
  'SPLADE Sparse Neural':
    'Sparse Lexical And Dense Expansion: a BERT-based model (naver/splade-v3, Apache 2.0) that learns which vocabulary terms are important for a passage and expands queries with related terms. Beats BM25 2× on MS MARCO benchmarks while staying fully lexical and interpretable. First use downloads ~440 MB model; subsequent queries use pre-encoded disk cache. Runs on CPU.',
  'Knowledge Graph':
    'Extracts named entities (people, places, concepts) from your query, traverses a graph of entity–relation edges built from the corpus, and fetches chunks that share those entities. Finds related content even without explicit textual overlap.',
  'LLM Entity Extraction':
    'Uses the LLM to extract richer entities and concepts from your query — including verbs, implicit concepts, and relationships that spaCy NER misses (e.g. "divide" → CONCEPT:division). Produces better graph traversal seeds for semantic queries. Falls back to spaCy automatically if LLM is offline.',
  'Cross-Encoder Rerank':
    'Passes each (query, chunk) pair through a dedicated neural cross-encoder model that scores their relevance jointly. More accurate than cosine similarity alone — expensive, so applied after the initial retrieval pool is narrowed.',
  'MMR Diversity':
    'Maximal Marginal Relevance: balances relevance with diversity. Iteratively picks the next chunk that is relevant to the query but dissimilar to already-selected chunks. Reduces repetitive results when multiple chunks say the same thing.',
  'Query Rewrite':
    'Uses an LLM to rephrase the raw query before retrieval: expands acronyms, removes conversational filler, adds synonyms. Turns a casual question into a retrieval-optimised form without changing its intent.',
  'Multi-Query':
    'Generates N paraphrases of the (rewritten) query via LLM, runs retrieval for each, then fuses all result lists with RRF. Increases recall by covering multiple angles of the same information need. Requires Query Rewrite to be enabled.',
  'HyDE':
    'Hypothetical Document Embeddings: asks the LLM to write a short ideal answer passage, embeds that passage, and uses its embedding for retrieval. Often outperforms embedding the raw question for factoid lookups.',
  'RAPTOR':
    'Recursive Abstractive Processing for Tree-Organised Retrieval: builds a hierarchy of LLM-generated summaries of your corpus clusters. Retrieves from both leaf chunks and higher-level summaries — great for multi-hop or document-level questions.',
  'Contextual Rerank':
    'Re-scores the final candidate chunks by asking the LLM "how relevant is this passage to the query?" and using the LLM\'s ranking. The most expensive step; catches subtle relevance misses that vector similarity and cross-encoder miss.',
}

const ALWAYS_AVAILABLE: Array<{
  key: string
  label: string
  children?: Array<{ key: string; label: string }>
}> = [
  { key: 'enable_dense', label: 'Dense Vector' },
  { key: 'enable_bm25', label: 'BM25 Keyword' },
  { key: 'enable_splade', label: 'SPLADE Sparse Neural' },
  {
    key: 'enable_graph',
    label: 'Knowledge Graph',
    children: [{ key: 'enable_llm_graph', label: 'LLM Entity Extraction' }],
  },
  { key: 'enable_rerank', label: 'Cross-Encoder Rerank' },
  { key: 'enable_mmr', label: 'MMR Diversity' },
]

// Groups with dependency: parent must be enabled before children can be used.
const LLM_GROUPS: Array<{
  key: string
  label: string
  children?: Array<{ key: string; label: string }>
}> = [
  {
    key: 'enable_rewrite',
    label: 'Query Rewrite',
    children: [{ key: 'enable_multi_query', label: 'Multi-Query' }],
  },
  { key: 'enable_hyde', label: 'HyDE' },
  { key: 'enable_raptor', label: 'RAPTOR' },
  { key: 'enable_contextual_rerank', label: 'Contextual Rerank' },
]

function InfoIcon({ tip }: { tip: string }) {
  const [visible, setVisible] = useState(false)
  return (
    <span className="relative flex-shrink-0">
      <button
        type="button"
        onMouseEnter={() => setVisible(true)}
        onMouseLeave={() => setVisible(false)}
        onFocus={() => setVisible(true)}
        onBlur={() => setVisible(false)}
        className="text-gray-500 hover:text-sky-400 focus:outline-none transition-colors"
        aria-label="More info"
      >
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-3.5 h-3.5">
          <path fillRule="evenodd" d="M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0Zm-7-4a1 1 0 1 1-2 0 1 1 0 0 1 2 0ZM9 9a.75.75 0 0 0 0 1.5h.253a.25.25 0 0 1 .244.304l-.459 2.066A1.75 1.75 0 0 0 10.747 15H11a.75.75 0 0 0 0-1.5h-.253a.25.25 0 0 1-.244-.304l.459-2.066A1.75 1.75 0 0 0 9.253 9H9Z" clipRule="evenodd" />
        </svg>
      </button>
      {visible && (
        <div className="absolute left-5 top-0 z-50 w-64 rounded-lg bg-gray-800 border border-gray-600 shadow-xl p-3 text-xs text-gray-300 leading-relaxed pointer-events-none">
          {tip}
        </div>
      )}
    </span>
  )
}

function Toggle({
  label,
  enabled,
  onChange,
  disabled,
  autoEnabled,
}: {
  label: string
  enabled: boolean
  onChange: () => void
  disabled?: boolean
  autoEnabled?: boolean
}) {
  const tip = METHOD_INFO[label]
  return (
    <div className={`flex items-center justify-between py-1 gap-1 ${disabled ? 'opacity-50' : ''}`}>
      <span className="flex items-center gap-1 text-sm text-gray-300 min-w-0">
        {tip && <InfoIcon tip={tip} />}
        <span className="truncate">{label}</span>
        {autoEnabled && (
          <span className="text-[10px] text-amber-400 ml-1" title="Auto-enabled by dependency">⚡</span>
        )}
      </span>
      <button
        role="switch"
        aria-checked={enabled}
        aria-label={label}
        onClick={disabled ? undefined : onChange}
        disabled={disabled}
        className={`relative inline-flex h-5 w-9 shrink-0 rounded-full border-2 border-transparent transition-colors focus:outline-none ${
          disabled ? 'cursor-not-allowed' : 'cursor-pointer'
        } ${enabled ? 'bg-brand-500' : 'bg-gray-700'}`}
      >
        <span
          className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow transition-transform ${
            enabled ? 'translate-x-4' : 'translate-x-0'
          }`}
        />
      </button>
    </div>
  )
}

export default function MethodToggle() {
  const { retrievalMethods, setRetrievalMethod, autoEnabledMethods } = useStore()

  return (
    <div className="space-y-3">
      {/* Independent methods — freely combinable */}
      <div>
        <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">Always Available</div>
        <div className="space-y-1">
          {ALWAYS_AVAILABLE.map(({ key, label, children }) => (
            <div key={key}>
              <Toggle
                label={label}
                enabled={retrievalMethods[key] ?? false}
                onChange={() => setRetrievalMethod(key, !retrievalMethods[key])}
                autoEnabled={autoEnabledMethods[key] ?? false}
              />
              {children && children.map((child) => {
                const parentEnabled = retrievalMethods[key] ?? false
                const childEnabled = retrievalMethods[child.key] ?? false
                return (
                  <div key={child.key} className="flex items-stretch ml-2">
                    <div className="flex flex-col items-center mr-1.5">
                      <div className="w-px flex-1 bg-gray-700" />
                      <div className="w-2 h-px bg-gray-700" />
                    </div>
                    <div className="flex-1">
                      <Toggle
                        label={child.label}
                        enabled={childEnabled}
                        onChange={() => setRetrievalMethod(child.key, !childEnabled)}
                        disabled={!parentEnabled}
                        autoEnabled={autoEnabledMethods[child.key] ?? false}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>

      {/* LLM-required with dependency groups */}
      <div className="border-t border-gray-700 pt-3">
        <div className="text-xs text-gray-400 uppercase tracking-wider mb-1">LLM-Required</div>
        <div className="space-y-1">
          {LLM_GROUPS.map(({ key, label, children }) => (
            <div key={key}>
              <Toggle
                label={label}
                enabled={retrievalMethods[key] ?? false}
                onChange={() => setRetrievalMethod(key, !retrievalMethods[key])}
                autoEnabled={autoEnabledMethods[key] ?? false}
              />
              {/* Children indented with connector line */}
              {children && children.map((child) => {
                const parentEnabled = retrievalMethods[key] ?? false
                const childEnabled = retrievalMethods[child.key] ?? false
                return (
                  <div key={child.key} className="flex items-stretch ml-2">
                    {/* Visual connector */}
                    <div className="flex flex-col items-center mr-1.5">
                      <div className="w-px flex-1 bg-gray-700" />
                      <div className="w-2 h-px bg-gray-700" />
                    </div>
                    <div className="flex-1">
                      <Toggle
                        label={child.label}
                        enabled={childEnabled}
                        onChange={() => setRetrievalMethod(child.key, !childEnabled)}
                        disabled={!parentEnabled}
                      />
                    </div>
                  </div>
                )
              })}
            </div>
          ))}
        </div>
      </div>
      <div className="text-[10px] text-gray-600 border-t border-gray-800 pt-2">
        ⚡ = auto-enabled as a required dependency
      </div>
    </div>
  )
}
