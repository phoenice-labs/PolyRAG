import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import BackendSelector from '../components/BackendSelector/BackendSelector'
import IngestionFlow from '../components/IngestionFlow/IngestionFlow'
import LogStream from '../components/LogStream/LogStream'
import { useStore } from '../store'
import { ingestText, previewChunks, type ChunkPreview } from '../api/ingest'
import { streamIngestLogs } from '../api/client'
import { deleteCollection } from '../api/backends'

const CHUNK_STRATEGIES = ['section', 'sliding', 'sentence', 'paragraph']

type SourceMode = 'text' | 'file' | 'server'

// ── Chunking Guide Modal ────────────────────────────────────────────────────

function ChunkingGuideModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-xl border border-gray-700 w-[640px] max-h-[85vh] overflow-y-auto shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-700">
          <div className="flex items-center gap-2">
            <span className="text-lg">📐</span>
            <h2 className="text-base font-semibold text-white">Chunking Strategy Guide</h2>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-white text-xl leading-none">✕</button>
        </div>

        <div className="px-6 py-5 space-y-6 text-sm">

          {/* Strategies */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Strategies</h3>
            <div className="space-y-3">
              {[
                {
                  name: 'sentence', badge: '✅ Best for most cases',
                  color: 'text-green-400', border: 'border-green-700',
                  desc: 'Splits on sentence boundaries. Preserves full thoughts — a sentence never gets cut in half. Great for prose, articles, documentation, and Shakespeare-style text.',
                  use: 'General-purpose RAG, Q&A, factual retrieval',
                },
                {
                  name: 'section', badge: 'Good for structured docs',
                  color: 'text-sky-400', border: 'border-sky-700',
                  desc: 'Splits on headings, blank lines, and paragraph breaks. Keeps logical sections together. Best when your document has clear headings or markdown structure.',
                  use: 'Technical docs, READMEs, reports, manuals',
                },
                {
                  name: 'paragraph', badge: 'Good for narrative text',
                  color: 'text-purple-400', border: 'border-purple-700',
                  desc: 'Splits on paragraph breaks (double newline). Keeps related ideas in one chunk. Sits between sentence and section in granularity.',
                  use: 'Books, essays, long-form articles',
                },
                {
                  name: 'sliding', badge: 'Use with caution',
                  color: 'text-amber-400', border: 'border-amber-700',
                  desc: 'Fixed-size window slides across the text character by character. Ignores sentence or paragraph boundaries — chunks may start or end mid-sentence. Highest recall for keyword search but lower semantic coherence.',
                  use: 'Dense keyword search, code files, structured data',
                },
              ].map(({ name, badge, color, border, desc, use }) => (
                <div key={name} className={`rounded-lg border ${border} bg-gray-800/60 p-3`}>
                  <div className="flex items-center gap-2 mb-1">
                    <span className="font-mono font-semibold text-white">{name}</span>
                    <span className={`text-xs ${color}`}>{badge}</span>
                  </div>
                  <p className="text-gray-300 leading-relaxed">{desc}</p>
                  <p className="text-gray-500 text-xs mt-1.5">Best for: <span className="text-gray-400">{use}</span></p>
                </div>
              ))}
            </div>
          </section>

          {/* Chunk Size */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Chunk Size</h3>
            <div className="space-y-2 text-gray-300 leading-relaxed">
              <p>Chunk size is the <span className="text-white font-medium">maximum character length</span> of a single chunk (not word count).</p>
              <div className="grid grid-cols-3 gap-2 mt-2">
                {[
                  { range: '128 – 256', label: 'Short', desc: 'Pinpoint precision. Good for FAQs, short answers. Loses wider context.' },
                  { range: '400 – 512', label: 'Sweet spot ✅', desc: 'Best balance of precision and context. Works with all strategies.' },
                  { range: '768 – 1024', label: 'Long', desc: 'More context per result. Can dilute relevance scores. Use for long docs.' },
                ].map(({ range, label, desc }) => (
                  <div key={range} className="bg-gray-800 rounded-lg p-2.5 text-center">
                    <div className="text-xs text-gray-500 font-mono">{range}</div>
                    <div className="text-xs font-semibold text-gray-200 mt-0.5">{label}</div>
                    <div className="text-xs text-gray-500 mt-1 leading-tight">{desc}</div>
                  </div>
                ))}
              </div>
              <p className="text-xs text-amber-400 mt-1">⚠ Chunks larger than 512 chars can reduce BM25 and SPLADE precision — longer text dilutes term-frequency signals.</p>
            </div>
          </section>

          {/* Overlap */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">Overlap</h3>
            <div className="space-y-2 text-gray-300 leading-relaxed">
              <p>Overlap repeats the <span className="text-white font-medium">last N characters</span> of each chunk at the start of the next. This ensures answers that span a boundary aren't lost.</p>
              <div className="grid grid-cols-3 gap-2 mt-2">
                {[
                  { range: '0', label: 'No overlap', desc: 'Fast. Risk of missing cross-boundary answers.' },
                  { range: '50 – 100', label: 'Recommended ✅', desc: 'Covers most boundary cases without bloating the index.' },
                  { range: '200+', label: 'High overlap', desc: 'Safest for long docs. Increases index size and ingest time.' },
                ].map(({ range, label, desc }) => (
                  <div key={range} className="bg-gray-800 rounded-lg p-2.5 text-center">
                    <div className="text-xs text-gray-500 font-mono">{range}</div>
                    <div className="text-xs font-semibold text-gray-200 mt-0.5">{label}</div>
                    <div className="text-xs text-gray-500 mt-1 leading-tight">{desc}</div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* Quick-pick recommendations */}
          <section className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">🎯 Quick Recommendations</h3>
            <div className="space-y-2">
              {[
                { use: 'General docs / Q&A', rec: 'sentence · 400–512 · overlap 64' },
                { use: 'Technical docs / manuals', rec: 'section · 512 · overlap 64' },
                { use: 'Books / long narrative', rec: 'paragraph · 512–768 · overlap 100' },
                { use: 'Code / logs / structured data', rec: 'sliding · 256–400 · overlap 32' },
                { use: 'Dense keyword retrieval', rec: 'sliding · 256 · overlap 0' },
              ].map(({ use, rec }) => (
                <div key={use} className="flex items-baseline gap-2 text-xs">
                  <span className="text-gray-400 w-44 shrink-0">{use}</span>
                  <span className="text-sky-300 font-mono">{rec}</span>
                </div>
              ))}
            </div>
          </section>

          {/* How chunking affects search */}
          <section>
            <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">How Chunking Affects Search Quality</h3>
            <ul className="space-y-1.5 text-gray-300 text-xs leading-relaxed list-none">
              <li>🔵 <span className="font-medium text-white">Dense (vector) search</span> — benefits from larger chunks with full context. Semantic meaning is preserved even in 512-char chunks.</li>
              <li>🟡 <span className="font-medium text-white">BM25 keyword search</span> — prefers smaller chunks. Larger chunks dilute term frequency, reducing keyword precision.</li>
              <li>🟠 <span className="font-medium text-white">SPLADE sparse neural</span> — similar to BM25. Smaller, focused chunks give sharper term-importance scores.</li>
              <li>🕸️ <span className="font-medium text-white">Knowledge Graph (ER)</span> — works best with sentence or paragraph strategy. Splitting mid-sentence breaks entity co-occurrence.</li>
              <li>📏 <span className="font-medium text-white">Cross-encoder reranker</span> — insensitive to chunk size; it re-scores candidates after retrieval.</li>
            </ul>
          </section>

          <div className="text-xs text-gray-600 border-t border-gray-700 pt-3">
            Tip: use <span className="font-mono text-gray-400">Preview Chunks</span> (paste text mode) to see exactly how your settings split your document before committing to a full ingest.
          </div>
        </div>
      </div>
    </div>
  )
}

export default function IngestionStudio() {
  const { selectedBackends, activeCollection, setActiveCollection } = useStore()
  const navigate = useNavigate()

  // Source mode
  const [sourceMode, setSourceMode] = useState<SourceMode>('server')
  const [text, setText] = useState('')
  const [serverPath, setServerPath] = useState('data/shakespeare.txt')
  const [dragging, setDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // Chunking config
  const [strategy, setStrategy] = useState('sentence')
  const [chunkSize, setChunkSize] = useState(512)
  const [overlap, setOverlap] = useState(64)
  const [extractEntities, setExtractEntities] = useState(false)
  const [showChunkingGuide, setShowChunkingGuide] = useState(false)

  // State
  const [logs, setLogs] = useState<string[]>([])
  const [activeStep, setActiveStep] = useState<string | undefined>()
  const [completedSteps, setCompletedSteps] = useState<Set<string>>(new Set())
  const [loading, setLoading] = useState(false)
  const [jobResults, setJobResults] = useState<Record<string, string>>({})  // backend → status
  const [erUsed, setErUsed] = useState(false)  // true when last ingest used ER
  const [ingestCollection, setIngestCollection] = useState('')  // collection used in last ingest
  const [preview, setPreview] = useState<ChunkPreview | null>(null)
  const [showPreview, setShowPreview] = useState(false)
  const [clearFirst, setClearFirst] = useState(false)

  // Step order for auto-advancing completed steps
  const STEP_ORDER = ['upload', 'chunk', 'embed', 'graph', 'upsert']

  const detectStep = (rawLine: string): string | undefined => {
    // Strip the [backend] prefix added by the frontend before matching
    const line = rawLine.replace(/^\[[\w]+\]\s*/, '').toLowerCase()

    // Order matters: more specific matches first
    if (/upsert|upserting|ingested \d+ chunk|stored|done\./.test(line)) return 'upsert'
    if (/graph|entity|entities|knowledge/.test(line)) return 'graph'
    if (/embed|embedding/.test(line)) return 'embed'
    if (/chunk|chunking|splitting/.test(line)) return 'chunk'
    if (/start|read|load|pipeline|corpus|upload/.test(line)) return 'upload'
    return undefined
  }

  const advanceStep = (step: string) => {
    const idx = STEP_ORDER.indexOf(step)
    if (idx < 0) return
    // Mark all prior steps as completed, set current as active
    setCompletedSteps((prev) => {
      const next = new Set(prev)
      STEP_ORDER.slice(0, idx).forEach((s) => next.add(s))
      return next
    })
    setActiveStep(step)
  }

  const markAllComplete = () => {
    setCompletedSteps(new Set(STEP_ORDER))
    setActiveStep(undefined)
  }

  // ── File drag-and-drop handlers ──────────────────────────────────────────
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true) }
  const handleDragLeave = () => setDragging(false)
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) readFile(file)
  }
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) readFile(file)
  }
  const readFile = (file: File) => {
    const reader = new FileReader()
    reader.onload = (ev) => {
      setText(ev.target?.result as string ?? '')
      setSourceMode('file')
    }
    reader.readAsText(file)
  }

  // ── Get effective text/path for the request ──────────────────────────────
  const buildPayload = (): { text?: string; corpus_path?: string } => {
    if (sourceMode === 'server') return { corpus_path: serverPath.trim() }
    return { text: text.trim() }
  }

  const hasContent = () => {
    if (sourceMode === 'server') return serverPath.trim().length > 0
    return text.trim().length > 0
  }

  const handleIngest = async () => {
    if (!hasContent() || selectedBackends.length === 0) return
    setLoading(true)
    setLogs([])
    setJobResults({})
    setErUsed(false)
    setCompletedSteps(new Set())
    advanceStep('upload')

    // Clear existing collection in each selected backend if requested
    if (clearFirst) {
      for (const backend of selectedBackends) {
        try {
          setLogs((prev) => [...prev, `[${backend}] Clearing collection "${activeCollection}"...`])
          await deleteCollection(backend, activeCollection)
          setLogs((prev) => [...prev, `[${backend}] Collection cleared ✓`])
        } catch {
          setLogs((prev) => [...prev, `[${backend}] Warning: could not clear collection (may not exist yet)`])
        }
      }
    }

    try {
      const response = await ingestText({
        ...buildPayload(),
        backends: selectedBackends,
        collection_name: activeCollection,
        chunk_strategy: strategy,
        chunk_size: chunkSize,
        overlap,
        enable_er: extractEntities,
      })

      const jobIds = response.job_ids  // { backend: job_id }
      const backends = Object.keys(jobIds)
      let doneCount = 0

      // Stream logs from all backend jobs simultaneously
      backends.forEach((backend) => {
        const jobId = jobIds[backend]
        streamIngestLogs(
          jobId,
          (line) => {
            const prefixed = `[${backend}] ${line}`
            setLogs((prev) => [...prev, prefixed])
            const step = detectStep(prefixed)
            if (step) advanceStep(step)
          },
          () => {
            doneCount++
            setJobResults((prev) => ({ ...prev, [backend]: 'done' }))
            if (doneCount === backends.length) {
              markAllComplete()
              setLoading(false)
              if (extractEntities) {
                setErUsed(true)
                setIngestCollection(activeCollection)
              }
            }
          }
        )
      })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      setLogs((prev) => [...prev, `ERROR: ${msg}`])
      setLoading(false)
    }
  }

  // ── Preview ───────────────────────────────────────────────────────────────
  const handlePreview = useCallback(async () => {
    const payload = buildPayload()
    const previewText = payload.text ?? ''
    if (!previewText && !payload.corpus_path) return

    // For server path, tell user we can only preview pasted text
    if (sourceMode === 'server' && !previewText) {
      alert('Chunk preview only works with pasted or uploaded text, not server paths.\nSwitch to Text or File mode to use Preview.')
      return
    }
    try {
      const result = await previewChunks(previewText, strategy, chunkSize, overlap)
      setPreview(result)
      setShowPreview(true)
    } catch (err) {
      alert(`Preview failed: ${err}`)
    }
  }, [text, serverPath, sourceMode, strategy, chunkSize, overlap])

  return (
    <div className="flex gap-4 h-full">
      {/* Left panel */}
      <div className="w-72 shrink-0 space-y-4">

        {/* Source panel */}
        <div className="bg-gray-900 rounded-lg p-4 space-y-3">
          <h2 className="text-sm font-semibold text-gray-300">Source</h2>

          {/* Mode tabs */}
          <div className="flex rounded overflow-hidden border border-gray-700 text-xs">
            {(['text', 'file', 'server'] as SourceMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setSourceMode(m)}
                className={`flex-1 py-1 capitalize ${sourceMode === m ? 'bg-brand-500 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
              >
                {m === 'server' ? 'Server Path' : m === 'file' ? 'Upload File' : 'Paste Text'}
              </button>
            ))}
          </div>

          {/* Paste text */}
          {sourceMode === 'text' && (
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your document text here..."
              className="w-full h-36 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 p-2 resize-none focus:outline-none focus:border-brand-500"
            />
          )}

          {/* Upload file (drag & drop + browse) */}
          {sourceMode === 'file' && (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className={`h-36 flex flex-col items-center justify-center rounded border-2 border-dashed cursor-pointer transition-colors ${
                dragging ? 'border-brand-500 bg-brand-500/10' : 'border-gray-600 bg-gray-800 hover:border-gray-500'
              }`}
            >
              <input ref={fileInputRef} type="file" accept=".txt,.md,.pdf,.csv" className="hidden" onChange={handleFileChange} />
              {text ? (
                <div className="text-center px-3">
                  <p className="text-green-400 text-xs font-medium mb-1">✓ File loaded</p>
                  <p className="text-gray-400 text-xs">{text.length.toLocaleString()} characters</p>
                  <p className="text-gray-500 text-xs mt-1">Click or drop to replace</p>
                </div>
              ) : (
                <div className="text-center px-3">
                  <p className="text-gray-400 text-sm mb-1">Drop a file here</p>
                  <p className="text-gray-500 text-xs">or click to browse</p>
                  <p className="text-gray-600 text-xs mt-2">.txt .md .pdf .csv</p>
                </div>
              )}
            </div>
          )}

          {/* Server-side path (e.g. data/shakespeare.txt) */}
          {sourceMode === 'server' && (
            <div className="space-y-2">
              <label className="text-xs text-gray-400">File path on server</label>
              <input
                value={serverPath}
                onChange={(e) => setServerPath(e.target.value)}
                placeholder="data/shakespeare.txt"
                className="w-full bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1.5 focus:outline-none focus:border-brand-500 font-mono"
              />
              <p className="text-xs text-gray-500">
                Path relative to project root. The API will read the file server-side.
              </p>
              <div className="space-y-1">
                {['data/shakespeare.txt', 'data/shakespeare_hamlet.txt'].map((p) => (
                  <button
                    key={p}
                    onClick={() => setServerPath(p)}
                    className="block w-full text-left text-xs text-brand-500 hover:text-sky-300 font-mono px-1"
                  >
                    {p}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Collection name */}
          <div>
            <label className="text-xs text-gray-400">Collection name</label>
            <input
              value={activeCollection}
              onChange={(e) => setActiveCollection(e.target.value)}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
          </div>
        </div>

        {/* Chunking config */}
        <div className="bg-gray-900 rounded-lg p-4 space-y-3">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-semibold text-gray-300">Chunking</h2>
            <button
              onClick={() => setShowChunkingGuide(true)}
              title="Chunking strategy guide"
              className="w-5 h-5 rounded-full bg-gray-700 hover:bg-brand-500 text-gray-300 hover:text-white text-xs font-bold flex items-center justify-center transition-colors"
            >
              ?
            </button>
          </div>
          <div>
            <label className="text-xs text-gray-400">Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full mt-1 bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none"
            >
              {CHUNK_STRATEGIES.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400">Chunk Size: {chunkSize}</label>
            <input type="range" min={64} max={2048} step={64} value={chunkSize}
              onChange={(e) => setChunkSize(+e.target.value)} className="w-full" />
            {chunkSize > 512 && (
              <p className="text-xs text-amber-400 mt-1">
                ⚠ Large chunks ({chunkSize} chars) dilute phrase recall — specific word/phrase searches may miss. Recommended: ≤ 512.
              </p>
            )}
          </div>
          <div>
            <label className="text-xs text-gray-400">Overlap: {overlap}</label>
            <input type="range" min={0} max={512} step={16} value={overlap}
              onChange={(e) => setOverlap(+e.target.value)} className="w-full" />
          </div>
          <label className="flex items-center gap-2 text-xs text-gray-400 cursor-pointer">
            <input type="checkbox" checked={extractEntities}
              onChange={(e) => setExtractEntities(e.target.checked)}
              className="rounded border-gray-600 bg-gray-800" />
            Extract Entities (ER)
          </label>
        </div>

        {/* Backend selector */}
        <div className="bg-gray-900 rounded-lg p-4">
          <BackendSelector />
        </div>

        {/* Ingest options + action buttons */}
        <div className="space-y-2">
          <label className="flex items-center gap-2 text-xs text-amber-400 cursor-pointer px-1">
            <input
              type="checkbox"
              checked={clearFirst}
              onChange={(e) => setClearFirst(e.target.checked)}
              className="rounded border-gray-600 bg-gray-800 accent-amber-500"
            />
            Clear collection before ingesting (replace mode)
          </label>
          <div className="flex gap-2">
            <button onClick={handlePreview}
              className="flex-1 py-2 text-sm bg-gray-700 hover:bg-gray-600 rounded text-gray-200">
              Preview Chunks
            </button>
            <button
              onClick={handleIngest}
              disabled={loading || !hasContent() || selectedBackends.length === 0}
              className={`flex-1 py-2 text-sm disabled:opacity-50 rounded font-medium text-white ${
                clearFirst ? 'bg-amber-700 hover:bg-amber-600' : 'bg-brand-500 hover:bg-sky-600'
              }`}
            >
              {loading ? 'Running...' : clearFirst ? '⚠ Clear & Ingest' : 'Start Ingestion'}
            </button>
          </div>
        </div>

        {/* Per-backend job status */}
        {Object.keys(jobResults).length > 0 && (
          <div className="bg-gray-900 rounded-lg p-3 space-y-1">
            <h3 className="text-xs font-semibold text-gray-400 mb-2">Job Results</h3>
            {Object.entries(jobResults).map(([backend, status]) => (
              <div key={backend} className="flex items-center justify-between text-xs">
                <span className="text-gray-300 font-mono">{backend}</span>
                <span className={status === 'done' ? 'text-green-400' : 'text-red-400'}>{status}</span>
              </div>
            ))}
            {/* ER entity graph shortcut — shown only after an ER-enabled ingest */}
            {erUsed && (
              <div className="mt-3 pt-3 border-t border-gray-700 space-y-2">
                <button
                  onClick={() => navigate(`/graph?collection=${encodeURIComponent(ingestCollection)}`)}
                  className="w-full py-2 text-xs font-medium rounded bg-purple-700 hover:bg-purple-600 text-white flex items-center justify-center gap-2"
                >
                  <span>🕸</span>
                  <span>View Entity Graph — {ingestCollection}</span>
                  <span className="text-purple-300">→</span>
                </button>
                <p className="text-gray-500 text-xs text-center">
                  {Object.keys(jobResults).length} backend{Object.keys(jobResults).length > 1 ? 's' : ''} · click a node to see relations &amp; chunk provenance
                </p>

                {/* LLM Graph Enhancement — redirect to Library */}
                <div className="pt-2 border-t border-gray-700">
                  <p className="text-xs text-gray-400 mb-1.5">
                    🧠 <span className="font-semibold text-indigo-300">Enhance Graph with LLM</span>
                    <span className="text-gray-500 ml-1">— richer entity extraction via LM Studio</span>
                  </p>
                  <button
                    onClick={() => navigate('/library')}
                    className="w-full py-2 text-xs font-medium rounded border border-indigo-700 text-indigo-300 hover:bg-indigo-900/40 flex items-center justify-center gap-2 transition-colors"
                  >
                    <span>📚</span>
                    <span>Go to Document Library → Enhance Graph</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Center — pipeline DAG + live logs */}
      <div className="flex-1 space-y-4">
        <div className="bg-gray-900 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-3">Pipeline</h2>
          <IngestionFlow activeStep={activeStep} completedSteps={completedSteps} />
        </div>
        <div className="bg-gray-900 rounded-lg p-4">
          <h2 className="text-sm font-semibold text-gray-300 mb-2">Live Logs</h2>
          <LogStream lines={logs} />
        </div>
      </div>

      {/* Chunk Preview Modal */}
      {showPreview && preview && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">
          <div className="bg-gray-900 rounded-xl p-6 w-2/3 max-h-[80vh] overflow-y-auto border border-gray-700">
            <div className="flex justify-between mb-4">
              <h2 className="text-lg font-semibold">Chunk Preview ({preview.total} chunks)</h2>
              <button onClick={() => setShowPreview(false)} className="text-gray-400 hover:text-white text-xl">✕</button>
            </div>
            <div className="space-y-2">
              {preview.chunks.map((chunk) => (
                <div key={chunk.index} className="bg-gray-800 rounded p-3 text-sm text-gray-200 border-l-2 border-brand-500">
                  <span className="text-xs text-gray-500 mr-2">[{chunk.index + 1}]</span>
                  {chunk.text}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Chunking Guide Modal */}
      {showChunkingGuide && <ChunkingGuideModal onClose={() => setShowChunkingGuide(false)} />}
    </div>
  )
}

