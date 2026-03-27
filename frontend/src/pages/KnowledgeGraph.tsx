import { useEffect, useRef, useState, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'
import * as d3 from 'd3'
import { api } from '../api/client'

interface ChunkRef {
  chunk_id: string
  snippet: string
}

interface NodeRelation {
  target_id: string
  target_label: string
  relation: string
  weight: number
}

interface GraphNode extends d3.SimulationNodeDatum {
  id: string
  label: string
  type: 'PERSON' | 'ORG' | 'LOC' | 'CONCEPT' | 'OTHER'
  frequency: number
  sources?: string[]
  chunks?: ChunkRef[]
  relations?: NodeRelation[]
}

interface GraphLink extends d3.SimulationLinkDatum<GraphNode> {
  relation: string
  weight?: number
}

const NODE_COLORS: Record<string, string> = {
  PERSON: '#3b82f6',
  ORG: '#f97316',
  LOC: '#22c55e',
  CONCEPT: '#a855f7',
  OTHER: '#6b7280',
}

const ENTITY_TYPES = ['PERSON', 'ORG', 'LOC', 'CONCEPT', 'OTHER'] as const

type TabId = 'visual' | 'entities' | 'relations' | 'chunks'

export default function KnowledgeGraph() {
  const [searchParams] = useSearchParams()
  const svgRef = useRef<SVGSVGElement>(null)
  const [collections, setCollections] = useState<string[]>([])
  const [collection, setCollection] = useState(searchParams.get('collection') ?? '')
  const [nodes, setNodes] = useState<GraphNode[]>([])
  const [links, setLinks] = useState<GraphLink[]>([])
  const [filter, setFilter] = useState<Set<string>>(new Set(ENTITY_TYPES))
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<GraphNode | null>(null)
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState<TabId>('visual')
  const [tableSearch, setTableSearch] = useState('')
  const [sortKey, setSortKey] = useState<'label' | 'type' | 'frequency' | 'chunks' | 'relations'>('frequency')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')

  useEffect(() => {
    api.get<string[]>('/graph').then((r) => {
      const names = r.data ?? []
      setCollections(names)
      if (names.length > 0 && !collection) setCollection(names[0])
    }).catch(() => {})
  }, [])  // eslint-disable-line react-hooks/exhaustive-deps

  const refreshCollections = useCallback(async () => {
    try {
      const names = (await api.get<string[]>('/graph')).data ?? []
      setCollections(names)
      // If current selection was cleared, auto-select first available
      if (names.length > 0 && !names.includes(collection)) {
        setCollection(names[0])
      } else if (names.length === 0) {
        setNodes([])
        setLinks([])
        setCollection('')
      }
    } catch {}
  }, [collection])

  const loadGraph = useCallback(async () => {
    if (!collection) return
    setLoading(true)
    try {
      const data = await api.get(`/graph/${collection}`).then((r) => r.data)
      setNodes(data.nodes ?? [])
      setLinks(data.edges ?? [])
    } catch {
      setNodes([])
      setLinks([])
    } finally {
      setLoading(false)
    }
  }, [collection])

  useEffect(() => { loadGraph() }, [loadGraph])

  // ── D3 visual ────────────────────────────────────────────────────────────────
  useEffect(() => {
    if (tab !== 'visual') return
    const svg = svgRef.current
    if (!svg) return

    const width = svg.clientWidth || 800
    const height = svg.clientHeight || 500

    const filteredNodes = nodes.filter(
      (n) => filter.has(n.type) && n.label.toLowerCase().includes(search.toLowerCase())
    )
    const nodeIds = new Set(filteredNodes.map((n) => n.id))
    const filteredLinks = links.filter(
      (l) => nodeIds.has((l.source as GraphNode).id ?? l.source as string) &&
             nodeIds.has((l.target as GraphNode).id ?? l.target as string)
    )

    const sel = d3.select(svg)
    sel.selectAll('*').remove()

    const g = sel.append('g')
    sel.call(
      d3.zoom<SVGSVGElement, unknown>().on('zoom', (event) => {
        g.attr('transform', event.transform.toString())
      })
    )

    const sim = d3.forceSimulation<GraphNode>(filteredNodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(filteredLinks).id((d) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))

    const link = g.append('g').selectAll('line').data(filteredLinks).join('line')
      .attr('stroke', '#4b5563').attr('stroke-width', (d) => Math.min(4, 1 + (d.weight ?? 1) * 0.3))

    const linkLabel = g.append('g').selectAll('text').data(filteredLinks).join('text')
      .attr('font-size', 9).attr('fill', '#6b7280').text((d) => d.relation)

    const node = g.append('g').selectAll('circle').data(filteredNodes).join('circle')
      .attr('r', (d) => 6 + Math.sqrt(d.frequency ?? 1) * 2)
      .attr('fill', (d) => NODE_COLORS[d.type] ?? '#6b7280')
      .attr('cursor', 'pointer')
      .on('click', (_, d) => { setSelected(d); setTab('visual') })
      .call(
        d3.drag<SVGCircleElement, GraphNode>()
          .on('start', (event, d) => { if (!event.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y })
          .on('drag', (event, d) => { d.fx = event.x; d.fy = event.y })
          .on('end', (event, d) => { if (!event.active) sim.alphaTarget(0); d.fx = null; d.fy = null })
      )

    const label = g.append('g').selectAll('text').data(filteredNodes).join('text')
      .attr('font-size', 10).attr('fill', '#d1d5db').attr('dx', 10).text((d) => d.label)

    sim.on('tick', () => {
      link
        .attr('x1', (d) => (d.source as GraphNode).x ?? 0)
        .attr('y1', (d) => (d.source as GraphNode).y ?? 0)
        .attr('x2', (d) => (d.target as GraphNode).x ?? 0)
        .attr('y2', (d) => (d.target as GraphNode).y ?? 0)
      linkLabel
        .attr('x', (d) => (((d.source as GraphNode).x ?? 0) + ((d.target as GraphNode).x ?? 0)) / 2)
        .attr('y', (d) => (((d.source as GraphNode).y ?? 0) + ((d.target as GraphNode).y ?? 0)) / 2)
      node.attr('cx', (d) => d.x ?? 0).attr('cy', (d) => d.y ?? 0)
      label.attr('x', (d) => d.x ?? 0).attr('y', (d) => d.y ?? 0)
    })

    return () => { sim.stop() }
  }, [nodes, links, filter, search, tab])

  const toggleFilter = (type: string) => {
    setFilter((prev) => {
      const next = new Set(prev)
      next.has(type) ? next.delete(type) : next.add(type)
      return next
    })
  }

  // ── Table helpers ─────────────────────────────────────────────────────────────
  const sortedNodes = [...nodes]
    .filter((n) => n.label.toLowerCase().includes(tableSearch.toLowerCase()))
    .sort((a, b) => {
      let va: number | string, vb: number | string
      if (sortKey === 'chunks') { va = a.chunks?.length ?? 0; vb = b.chunks?.length ?? 0 }
      else if (sortKey === 'relations') { va = a.relations?.length ?? 0; vb = b.relations?.length ?? 0 }
      else if (sortKey === 'frequency') { va = a.frequency; vb = b.frequency }
      else if (sortKey === 'type') { va = a.type; vb = b.type }
      else { va = a.label; vb = b.label }
      if (va < vb) return sortDir === 'asc' ? -1 : 1
      if (va > vb) return sortDir === 'asc' ? 1 : -1
      return 0
    })

  const handleSort = (key: typeof sortKey) => {
    if (sortKey === key) setSortDir((d) => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const sortIcon = (key: typeof sortKey) =>
    sortKey !== key ? '⇅' : sortDir === 'desc' ? '↓' : '↑'

  // All chunk appearances (flattened)
  const allChunkLinks = nodes
    .flatMap((n) => (n.chunks ?? []).map((c) => ({ entity: n.label, type: n.type, ...c })))
    .filter((c) => c.entity.toLowerCase().includes(tableSearch.toLowerCase()) || c.snippet.toLowerCase().includes(tableSearch.toLowerCase()))

  // All edges
  const allRelations = links
    .filter((l) => {
      const src = typeof l.source === 'object' ? (l.source as GraphNode).label : String(l.source)
      const tgt = typeof l.target === 'object' ? (l.target as GraphNode).label : String(l.target)
      const q = tableSearch.toLowerCase()
      return !q || src.toLowerCase().includes(q) || tgt.toLowerCase().includes(q) || l.relation.toLowerCase().includes(q)
    })

  const TABS: { id: TabId; label: string; count?: number }[] = [
    { id: 'visual', label: '🕸 Visual' },
    { id: 'entities', label: '🔵 Entities', count: nodes.length },
    { id: 'relations', label: '↔ Relations', count: links.length },
    { id: 'chunks', label: '📄 Chunk Links', count: nodes.reduce((s, n) => s + (n.chunks?.length ?? 0), 0) },
  ]

  return (
    <div className="flex flex-col h-full gap-3">
      {/* Header bar */}
      <div className="flex gap-2 items-center flex-wrap">
        <select
          value={collection}
          onChange={(e) => setCollection(e.target.value)}
          className="bg-gray-800 text-sm text-gray-200 rounded border border-gray-700 px-2 py-1"
        >
          {collections.map((c) => <option key={c} value={c}>{c}</option>)}
          {!collections.includes(collection) && collection && <option value={collection}>{collection}</option>}
        </select>

        <button
          onClick={async () => { await refreshCollections(); await loadGraph() }}
          title="Refresh collection list and graph data from server"
          className="px-2 py-1 text-xs text-gray-400 hover:text-gray-200 bg-gray-800 rounded border border-gray-700 hover:border-gray-500 transition-colors"
        >
          ↻ Refresh
        </button>

        {/* Tab switcher */}
        <div className="flex gap-1 bg-gray-800 rounded p-0.5">
          {TABS.map(({ id, label, count }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`px-3 py-1 rounded text-xs font-medium transition-colors ${tab === id ? 'bg-brand-600 text-white' : 'text-gray-400 hover:text-gray-200'}`}
            >
              {label}{count !== undefined ? ` (${count})` : ''}
            </button>
          ))}
        </div>

        {/* Type filters — only on visual tab */}
        {tab === 'visual' && (
          <>
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Filter nodes..."
              className="w-36 bg-gray-800 text-xs text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
            />
            {ENTITY_TYPES.map((type) => (
              <button
                key={type}
                onClick={() => toggleFilter(type)}
                className={`px-2 py-1 rounded text-xs font-medium border ${filter.has(type) ? 'border-transparent text-white' : 'border-gray-600 text-gray-500'}`}
                style={{ backgroundColor: filter.has(type) ? NODE_COLORS[type] : 'transparent' }}
              >
                {type}
              </button>
            ))}
          </>
        )}

        {/* Search — on table tabs */}
        {tab !== 'visual' && (
          <input
            value={tableSearch}
            onChange={(e) => setTableSearch(e.target.value)}
            placeholder="Search..."
            className="flex-1 bg-gray-800 text-xs text-gray-200 rounded border border-gray-700 px-2 py-1 focus:outline-none focus:border-brand-500"
          />
        )}

        {loading && <span className="text-gray-500 text-xs">Loading…</span>}
      </div>

      {/* ── Visual tab ──────────────────────────────────────────────────────── */}
      {tab === 'visual' && (
        <div className="flex gap-4 flex-1 min-h-0">
          <div className="flex-1 bg-gray-900 rounded border border-gray-700 relative">
            <svg ref={svgRef} className="w-full h-full" />
            {nodes.length === 0 && !loading && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-gray-600 gap-2">
                <span>No graph data.</span>
                <span className="text-xs">Re-ingest with "Extract Entities (ER)" enabled.</span>
              </div>
            )}
          </div>

          {selected && (
            <div className="w-72 bg-gray-900 rounded border border-gray-700 p-4 space-y-4 overflow-y-auto flex-shrink-0">
              <div className="flex justify-between items-start">
                <h3 className="font-semibold text-gray-200 leading-tight">{selected.label}</h3>
                <button onClick={() => setSelected(null)} className="text-gray-500 hover:text-white ml-2">✕</button>
              </div>
              <div className="flex gap-2 items-center flex-wrap">
                <span className="px-2 py-0.5 rounded text-xs text-white font-medium"
                  style={{ backgroundColor: NODE_COLORS[selected.type] ?? '#6b7280' }}>
                  {selected.type}
                </span>
                <span className="text-gray-400 text-xs">frequency: {selected.frequency}</span>
                {selected.chunks && <span className="text-gray-500 text-xs">{selected.chunks.length} chunks</span>}
                {selected.relations && <span className="text-gray-500 text-xs">{selected.relations.length} relations</span>}
              </div>

              {selected.relations && selected.relations.length > 0 && (
                <div>
                  <div className="text-gray-300 text-xs font-semibold uppercase tracking-wide mb-2">Relations</div>
                  <div className="space-y-1 max-h-40 overflow-y-auto">
                    {selected.relations.map((r, i) => (
                      <div key={i} className="flex items-center gap-1 text-xs">
                        <span className="text-brand-400 font-mono">{r.relation}</span>
                        <span className="text-gray-600">→</span>
                        <span className="text-gray-300 truncate">{r.target_label}</span>
                        <span className="text-gray-600 ml-auto flex-shrink-0">×{r.weight.toFixed(1)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {selected.chunks && selected.chunks.length > 0 && (
                <div>
                  <div className="text-gray-300 text-xs font-semibold uppercase tracking-wide mb-2">
                    Appears in {selected.chunks.length} chunk{selected.chunks.length > 1 ? 's' : ''}
                  </div>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {selected.chunks.map((c, i) => (
                      <div key={i} className="bg-gray-800 rounded p-2 text-xs">
                        <div className="text-gray-500 font-mono mb-1 truncate text-[10px]">{c.chunk_id}</div>
                        <div className="text-gray-300 leading-relaxed">&ldquo;{c.snippet}&hellip;&rdquo;</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {(!selected.chunks || selected.chunks.length === 0) &&
               (!selected.relations || selected.relations.length === 0) && (
                <div className="text-gray-600 text-xs">No data. Re-ingest with ER enabled.</div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ── Entities table ───────────────────────────────────────────────────── */}
      {tab === 'entities' && (
        <div className="flex-1 overflow-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="sticky top-0 bg-gray-900 z-10">
              <tr>
                {([['label','Entity'], ['type','Type'], ['frequency','Freq'], ['chunks','Chunks'], ['relations','Relations']] as [typeof sortKey, string][]).map(([key, label]) => (
                  <th key={key} onClick={() => handleSort(key)}
                    className="px-3 py-2 text-gray-400 font-semibold cursor-pointer hover:text-white border-b border-gray-700 whitespace-nowrap select-none">
                    {label} {sortIcon(key)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedNodes.map((n) => (
                <tr key={n.id} className="border-b border-gray-800 hover:bg-gray-800 cursor-pointer"
                  onClick={() => { setSelected(n); setTab('visual') }}>
                  <td className="px-3 py-2 text-gray-200 font-medium">{n.label}</td>
                  <td className="px-3 py-2">
                    <span className="px-1.5 py-0.5 rounded text-white text-[10px] font-medium"
                      style={{ backgroundColor: NODE_COLORS[n.type] ?? '#6b7280' }}>
                      {n.type}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-gray-300">{n.frequency}</td>
                  <td className="px-3 py-2 text-gray-300">{n.chunks?.length ?? 0}</td>
                  <td className="px-3 py-2 text-gray-300">{n.relations?.length ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {sortedNodes.length === 0 && (
            <div className="text-center text-gray-600 py-12">No entities match your search.</div>
          )}
        </div>
      )}

      {/* ── Relations table ──────────────────────────────────────────────────── */}
      {tab === 'relations' && (
        <div className="flex-1 overflow-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="sticky top-0 bg-gray-900 z-10">
              <tr>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Source</th>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Relation</th>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Target</th>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Weight</th>
              </tr>
            </thead>
            <tbody>
              {allRelations.map((l, i) => {
                const srcNode = typeof l.source === 'object' ? l.source as GraphNode : nodes.find((n) => n.id === l.source)
                const tgtNode = typeof l.target === 'object' ? l.target as GraphNode : nodes.find((n) => n.id === l.target)
                return (
                  <tr key={i} className="border-b border-gray-800 hover:bg-gray-800">
                    <td className="px-3 py-2">
                      <span className="text-gray-200 font-medium">{srcNode?.label ?? String(l.source)}</span>
                      {srcNode && <span className="ml-1 px-1 py-0.5 rounded text-white text-[10px]"
                        style={{ backgroundColor: NODE_COLORS[srcNode.type] ?? '#6b7280' }}>{srcNode.type}</span>}
                    </td>
                    <td className="px-3 py-2 text-brand-400 font-mono">{l.relation}</td>
                    <td className="px-3 py-2">
                      <span className="text-gray-200 font-medium">{tgtNode?.label ?? String(l.target)}</span>
                      {tgtNode && <span className="ml-1 px-1 py-0.5 rounded text-white text-[10px]"
                        style={{ backgroundColor: NODE_COLORS[tgtNode.type] ?? '#6b7280' }}>{tgtNode.type}</span>}
                    </td>
                    <td className="px-3 py-2 text-gray-400">{l.weight?.toFixed(1) ?? '1.0'}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
          {allRelations.length === 0 && (
            <div className="text-center text-gray-600 py-12">
              {links.length === 0
                ? 'No relation edges yet. Re-ingest with ER enabled to populate.'
                : 'No relations match your search.'}
            </div>
          )}
        </div>
      )}

      {/* ── Chunk Links table ────────────────────────────────────────────────── */}
      {tab === 'chunks' && (
        <div className="flex-1 overflow-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead className="sticky top-0 bg-gray-900 z-10">
              <tr>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Entity</th>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Type</th>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Chunk ID</th>
                <th className="px-3 py-2 text-gray-400 font-semibold border-b border-gray-700">Snippet</th>
              </tr>
            </thead>
            <tbody>
              {allChunkLinks.map((c, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-gray-800">
                  <td className="px-3 py-2 text-gray-200 font-medium whitespace-nowrap">{c.entity}</td>
                  <td className="px-3 py-2">
                    <span className="px-1.5 py-0.5 rounded text-white text-[10px] font-medium"
                      style={{ backgroundColor: NODE_COLORS[c.type] ?? '#6b7280' }}>
                      {c.type}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-gray-500 font-mono text-[10px] max-w-[160px] truncate">{c.chunk_id}</td>
                  <td className="px-3 py-2 text-gray-300 max-w-xs truncate" title={c.snippet}>&ldquo;{c.snippet}&rdquo;</td>
                </tr>
              ))}
            </tbody>
          </table>
          {allChunkLinks.length === 0 && (
            <div className="text-center text-gray-600 py-12">
              {nodes.every((n) => !n.chunks?.length)
                ? 'No chunk provenance yet. Re-ingest with ER enabled.'
                : 'No chunk links match your search.'}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

