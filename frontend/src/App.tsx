import { useState } from 'react'
import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import BackendHealthBar from './components/BackendHealthBar/BackendHealthBar'
import IngestionStudio from './pages/IngestionStudio'
import SearchLab from './pages/SearchLab'
import ComparisonMatrix from './pages/ComparisonMatrix'
import KnowledgeGraph from './pages/KnowledgeGraph'
import DocumentLibrary from './pages/DocumentLibrary'
import EvaluationStudio from './pages/EvaluationStudio'
import Settings from './pages/Settings'
import JobHistory from './pages/JobHistory'
import PromptEditor from './pages/PromptEditor'
import { useStore } from './store'

const queryClient = new QueryClient()

const NAV = [
  { to: '/', label: '⬆ Ingest', exact: true },
  { to: '/search', label: '🔍 Search' },
  { to: '/compare', label: '📊 Compare' },
  { to: '/graph', label: '🕸 Graph' },
  { to: '/library', label: '📚 Library' },
  { to: '/evaluate', label: '⚖ Evaluate' },
  { to: '/jobs', label: '🗂 Jobs' },
  { to: '/prompts', label: '🧠 Prompts' },
  { to: '/settings', label: '⚙ Settings' },
]

// ── PolyRAG About Modal ───────────────────────────────────────────────────────

function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[620px] max-h-[88vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-3">
            <span className="text-2xl font-bold text-brand-500">PolyRAG</span>
            <span className="text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">v0.1.0</span>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* Tagline */}
          <section>
            <p className="text-brand-400 text-base font-medium italic leading-relaxed">
              "Write your RAG orchestration once. Run it on any vector store."
            </p>
          </section>

          {/* The Idea */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">The Idea</h3>
            <p className="text-gray-300 leading-relaxed">
              Most RAG (Retrieval-Augmented Generation) systems are <span className="text-white font-medium">hard-wired to one vector database</span> —
              switching backends means rewriting ingestion, retrieval, and search logic from scratch. PolyRAG inverts this:
              a single unified pipeline sits above all backends, and you swap the store by changing <span className="text-white font-medium">one config line</span>.
            </p>
          </section>

          {/* Why "Poly" */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Why "Poly"?</h3>
            <div className="grid grid-cols-2 gap-2 text-xs">
              {[
                { icon: '🗄️', title: 'Poly-Store', desc: '6 vector databases — FAISS, ChromaDB, Qdrant, Weaviate, Milvus, PGVector — all behind one interface. Pick any, mix any.' },
                { icon: '🔍', title: 'Poly-Retrieval', desc: '11 retrieval methods from BM25 keyword to SPLADE sparse neural to RAPTOR hierarchical — combined via RRF fusion.' },
                { icon: '🧩', title: 'Poly-Encoder', desc: 'Multiple embedding models (MiniLM, BGE-base, BGE-large) with automatic collection isolation — no data cross-contamination.' },
                { icon: '⚗️', title: 'Poly-Pipeline', desc: '11 composable phases — chunking, embedding, retrieval, reranking, graph, confidence, provenance, RBAC — all wired together, all independently configurable.' },
              ].map(({ icon, title, desc }) => (
                <div key={title} className="bg-gray-800 rounded p-2.5 border border-gray-700">
                  <div className="text-white font-medium mb-1">{icon} {title}</div>
                  <div className="text-gray-400 leading-relaxed">{desc}</div>
                </div>
              ))}
            </div>
          </section>

          {/* The Problem it Solves */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">The Problem It Solves</h3>
            <p className="text-gray-300 leading-relaxed">
              Building a production RAG system means making hard early bets on a vector store, an embedding model, and a retrieval strategy —
              then discovering later that your choice doesn't scale, or a newer method outperforms it.
              PolyRAG is built for <span className="text-white font-medium">experimentation without lock-in</span>: swap backends, compare them live, tune retrieval methods, and measure quality objectively — all from this UI.
            </p>
          </section>

          {/* The 11-Phase Pipeline */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">The 11-Phase Pipeline</h3>
            <div className="grid grid-cols-2 gap-1.5 text-xs">
              {[
                { n: '1', label: 'Vector Store Abstraction', desc: '6 pluggable adapters behind one interface' },
                { n: '2', label: 'Semantic Chunking', desc: 'Fixed, sentence-boundary, semantic strategies' },
                { n: '3', label: 'Hybrid Search', desc: 'Dense + BM25 + SPLADE fused via RRF' },
                { n: '4', label: 'Multi-Stage Reranking', desc: 'Cross-encoder over top candidates' },
                { n: '5', label: 'Query Intelligence', desc: 'Rewrite, Multi-Query, HyDE via LLM' },
                { n: '6', label: 'Provenance & Citations', desc: 'Full traceability of every returned chunk' },
                { n: '7', label: 'Confidence Signals', desc: '7-signal confidence aggregator + verdict' },
                { n: '8', label: 'Temporal / RBAC Filters', desc: 'Lifecycle, access policies, classification' },
                { n: '9', label: 'Noise & Quality', desc: 'Deduplication, quality scoring, observability' },
                { n: '10', label: 'Knowledge Graph', desc: 'spaCy NER + Kuzu graph + 3-way hybrid' },
                { n: '11', label: 'RAPTOR + MMR + Contextual', desc: 'Hierarchical indexing, diversity, LLM rerank' },
              ].map(({ n, label, desc }) => (
                <div key={n} className="flex gap-2 bg-gray-800 rounded p-2 border border-gray-700/50">
                  <span className="w-5 h-5 rounded-full bg-brand-500/20 text-brand-400 text-xs flex items-center justify-center shrink-0 font-bold">{n}</span>
                  <div>
                    <div className="text-gray-200 font-medium">{label}</div>
                    <div className="text-gray-500 leading-relaxed mt-0.5">{desc}</div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* Design Principles */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Design Principles</h3>
            <ul className="space-y-1.5 text-xs text-gray-400 list-disc list-inside">
              <li><span className="text-gray-200">No GPU required</span> — all embeddings run on CPU; works on any developer laptop.</li>
              <li><span className="text-gray-200">One config file</span> — <code className="bg-gray-800 px-1 rounded">config/config.yaml</code> drives all 11 phases and all backends.</li>
              <li><span className="text-gray-200">Graceful degradation</span> — LM Studio, Neo4j, and Docker backends are all optional; the core pipeline always runs.</li>
              <li><span className="text-gray-200">All data regenerable</span> — delete <code className="bg-gray-800 px-1 rounded">data/</code> at any time; re-ingest rebuilds everything.</li>
              <li><span className="text-gray-200">Measure, don't guess</span> — the Compare and Evaluate screens exist so you make data-driven decisions, not vendor-driven ones.</li>
            </ul>
          </section>

          {/* Stack */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Tech Stack</h3>
            <div className="flex flex-wrap gap-2 text-xs">
              {['FastAPI', 'Python 3.10+', 'React 19', 'TypeScript', 'Vite', 'Tailwind CSS',
                'sentence-transformers', 'spaCy', 'Kuzu', 'rank-bm25', 'FAISS', 'ChromaDB',
                'Qdrant', 'Weaviate', 'Milvus', 'PGVector', 'D3.js', 'ReactFlow'].map((t) => (
                <span key={t} className="bg-gray-800 border border-gray-700 text-gray-300 px-2 py-0.5 rounded">{t}</span>
              ))}
            </div>
          </section>

          {/* API / Developer */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">Developer & Agentic Integration</h3>
            <div className="bg-gray-800 rounded p-3 border border-gray-700 text-xs text-gray-300 space-y-2">
              <p>PolyRAG exposes a full REST API for integration into agentic AI workflows, LLM pipelines, and external systems.</p>

              {/* Unified endpoint callout */}
              <div className="bg-gray-900 rounded p-2.5 border border-brand-500/40">
                <p className="text-brand-400 font-semibold mb-1">⭐ Unified Agentic Endpoint</p>
                <p className="text-gray-300 mb-1">
                  <code className="bg-gray-800 px-1 rounded text-green-400">POST /api/rag</code> — single production call. Pass a query + your saved profile ID. Returns a fully traceable answer with confidence verdict, per-chunk lineage, and pipeline audit.
                </p>
                <p className="text-gray-400 text-[11px]">
                  Developer journey: <span className="text-gray-300">Ingest → Search → Evaluate → Compare → Save Profile → Call /api/rag</span>
                </p>
              </div>

              {/* Profile system */}
              <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                <div className="bg-gray-900 rounded p-2 border border-gray-700">
                  <p className="text-gray-300 font-medium mb-0.5">Profile System</p>
                  <p className="text-gray-400">Save your tested config (backend + model + methods) as a named profile. Reference it by ID in every agent call — no config noise in production.</p>
                </div>
                <div className="bg-gray-900 rounded p-2 border border-gray-700">
                  <p className="text-gray-300 font-medium mb-0.5">Full Traceability</p>
                  <p className="text-gray-400">Every response carries: confidence verdict, per-chunk method lineage (which retrievers found it + RRF scores), reranker chain, pipeline funnel counts, and LLM call traces.</p>
                </div>
              </div>

              <div className="flex flex-wrap gap-3 pt-1">
                <a href="http://localhost:8000/docs" target="_blank" rel="noreferrer"
                  className="text-sky-400 hover:text-sky-300 underline">
                  📖 Swagger UI (localhost:8000/docs)
                </a>
                <span className="text-gray-600">·</span>
                <span className="text-gray-400">See <code className="bg-gray-900 px-1 rounded">API_GUIDE.md</code> for full integration guide, agentic patterns, and method recipes.</span>
              </div>
            </div>
          </section>

        </div>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AppShell />
      </BrowserRouter>
    </QueryClientProvider>
  )
}

function AppShell() {
  const [showAbout, setShowAbout] = useState(false)
  const activeIngestJobs = useStore((s) => s.activeIngestJobs)
  const clearActiveIngestJobs = useStore((s) => s.clearActiveIngestJobs)
  const setActiveIngestJob = useStore((s) => s.setActiveIngestJob)

  // Poll backend for running jobs — syncs Zustand store with server truth.
  const { data: runningJobs } = useQuery<{ id: string; status: string; backend: string }[]>({
    queryKey: ['runningJobs'],
    queryFn: async () => {
      const res = await fetch('/api/ingest/jobs')
      if (!res.ok) return []
      return res.json()
    },
    refetchInterval: 5000,
    select: (jobs) => jobs.filter((j) => j.status === 'running' || j.status === 'pending'),
  })

  // Keep Zustand store in sync: clear jobs that are no longer running on the server.
  // Must be in a useEffect — calling setState during render causes React warning.
  useEffect(() => {
    if (runningJobs === undefined) return
    const runningIds = new Set(runningJobs.map((j) => j.id))
    const next: Record<string, string> = {}
    for (const [backend, jobId] of Object.entries(activeIngestJobs)) {
      if (runningIds.has(jobId)) next[backend] = jobId
    }
    if (Object.keys(next).length !== Object.keys(activeIngestJobs).length) {
      clearActiveIngestJobs()
      for (const [b, id] of Object.entries(next)) setActiveIngestJob(b, id)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runningJobs])

  const runningCount = runningJobs?.length ?? 0

  return (
    <div className="flex flex-col h-screen">
      <BackendHealthBar />
      <div className="flex flex-1 overflow-hidden">
        <nav className="w-48 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-1 shrink-0">
          <div className="flex items-center gap-1.5 mb-4">
            <span className="text-brand-500 font-bold text-lg">PolyRAG</span>
            <button
              onClick={() => setShowAbout(true)}
              title="About PolyRAG"
              className="w-4 h-4 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-400 hover:text-white text-[10px] font-bold flex items-center justify-center transition-colors shrink-0"
            >
              i
            </button>
          </div>
          {NAV.map(({ to, label }) => (
            <NavLink key={to} to={to} end={to==='/'} className={({isActive}) =>
              `px-3 py-2 rounded text-sm transition-colors ${isActive ? 'bg-brand-500 text-white' : 'text-gray-400 hover:text-gray-100 hover:bg-gray-800'}`
            }>{label}</NavLink>
          ))}
          {runningCount > 0 && (
            <NavLink
              to="/jobs"
              className="mt-auto flex items-center gap-1.5 px-3 py-2 rounded text-xs bg-amber-600/20 text-amber-400 hover:bg-amber-600/30 transition-colors animate-pulse"
            >
              <span className="w-2 h-2 rounded-full bg-amber-400 shrink-0" />
              {runningCount} job{runningCount > 1 ? 's' : ''} running
            </NavLink>
          )}
        </nav>
        <main className="flex-1 overflow-auto p-6">
          <Routes>
            <Route path="/" element={<IngestionStudio />} />
            <Route path="/search" element={<SearchLab />} />
            <Route path="/compare" element={<ComparisonMatrix />} />
            <Route path="/graph" element={<KnowledgeGraph />} />
            <Route path="/library" element={<DocumentLibrary />} />
            <Route path="/evaluate" element={<EvaluationStudio />} />
            <Route path="/jobs" element={<JobHistory />} />
            <Route path="/prompts" element={<PromptEditor />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
    </div>
  )
}
