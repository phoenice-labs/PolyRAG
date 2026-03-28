import { useState, useEffect, useRef } from 'react'
import { listJobs, type IngestJob } from '../api/ingest'

const STATUS_COLORS: Record<IngestJob['status'], string> = {
  pending: 'text-yellow-400',
  running: 'text-blue-400',
  done: 'text-green-400',
  error: 'text-red-400',
}

const STATUS_DOT: Record<IngestJob['status'], string> = {
  pending: 'bg-yellow-400',
  running: 'bg-blue-400 animate-pulse',
  done: 'bg-green-400',
  error: 'bg-red-400',
}

type FilterStatus = 'all' | IngestJob['status']

function elapsedSecs(job: IngestJob): string {
  const start = new Date(job.created_at).getTime()
  const end = job.updated_at ? new Date(job.updated_at).getTime() : Date.now()
  const secs = Math.max(0, (end - start) / 1000)
  return secs > 0 ? `${secs.toFixed(1)}s` : '—'
}

export default function JobHistory() {
  const [jobs, setJobs] = useState<IngestJob[]>([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<FilterStatus>('all')
  const [expanded, setExpanded] = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const fetchJobs = async (silent = false) => {
    if (!silent) setLoading(true)
    try {
      const data = await listJobs()
      setJobs(data)
    } catch {
      if (!silent) setJobs([])
    } finally {
      if (!silent) setLoading(false)
    }
  }

  // Initial fetch + auto-poll every 5s while any job is running/pending
  useEffect(() => {
    fetchJobs()
  }, [])

  useEffect(() => {
    const hasActive = jobs.some((j) => j.status === 'running' || j.status === 'pending')
    if (hasActive && !pollRef.current) {
      pollRef.current = setInterval(() => fetchJobs(true), 5000)
    } else if (!hasActive && pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
    return () => {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
    }
  }, [jobs])

  const filtered = filter === 'all' ? jobs : jobs.filter((j) => j.status === filter)
  const runningCount = jobs.filter((j) => j.status === 'running' || j.status === 'pending').length

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-semibold">Job History</h1>
          {runningCount > 0 && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-blue-900/60 border border-blue-700 text-xs text-blue-300">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
              {runningCount} running · auto-refreshing
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button onClick={() => fetchJobs()} disabled={loading}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-sm text-gray-200 rounded">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
        </div>
      </div>

      <div className="flex gap-2 flex-wrap">
        {(['all', 'running', 'pending', 'done', 'error'] as FilterStatus[]).map((s) => {
          const count = s === 'all' ? jobs.length : jobs.filter((j) => j.status === s).length
          return (
            <button
              key={s}
              onClick={() => setFilter(s)}
              className={`px-3 py-1 rounded text-sm flex items-center gap-1.5 ${filter === s ? 'bg-brand-500 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
            >
              {s !== 'all' && <span className={`w-1.5 h-1.5 rounded-full ${STATUS_DOT[s as IngestJob['status']]}`} />}
              {s}
              <span className="text-xs opacity-60">({count})</span>
            </button>
          )
        })}
      </div>

      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="px-3 py-2 text-left text-gray-400">Job ID</th>
            <th className="px-3 py-2 text-left text-gray-400">Backend</th>
            <th className="px-3 py-2 text-left text-gray-400">Collection</th>
            <th className="px-3 py-2 text-left text-gray-400">Status</th>
            <th className="px-3 py-2 text-left text-gray-400">Created</th>
            <th className="px-3 py-2 text-left text-gray-400">Elapsed</th>
            <th className="px-3 py-2 text-left text-gray-400">Chunks</th>
          </tr>
        </thead>
        <tbody>
          {filtered.length === 0 ? (
            <tr>
              <td colSpan={7} className="px-3 py-8 text-center text-gray-600">No jobs found</td>
            </tr>
          ) : (
            filtered.map((job) => (
              <>
                <tr
                  key={job.job_id}
                  className="border-b border-gray-800 hover:bg-gray-800/50 cursor-pointer"
                  onClick={() => setExpanded(expanded === job.job_id ? null : job.job_id)}
                >
                  <td className="px-3 py-2 text-gray-400 font-mono text-xs">{job.job_id.slice(0, 8)}…</td>
                  <td className="px-3 py-2 text-gray-300">{job.backend}</td>
                  <td className="px-3 py-2 text-gray-400 text-xs truncate max-w-[140px]" title={job.collection_name}>{job.collection_name ?? '—'}</td>
                  <td className={`px-3 py-2 font-medium ${STATUS_COLORS[job.status]}`}>
                    <span className="flex items-center gap-1.5">
                      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${STATUS_DOT[job.status]}`} />
                      {job.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-gray-400">{new Date(job.created_at).toLocaleString()}</td>
                  <td className="px-3 py-2 text-gray-400">{elapsedSecs(job)}</td>
                  <td className="px-3 py-2 text-gray-400">{job.result?.total_chunks ?? '—'}</td>
                </tr>
                {expanded === job.job_id && (
                  <tr key={`${job.job_id}-expanded`} className="border-b border-gray-800 bg-gray-900/50">
                    <td colSpan={7} className="px-6 py-3">
                      {job.error && (
                        <div className="text-xs text-red-400 mb-2">
                          <span className="font-medium text-gray-300">Error: </span>{job.error}
                        </div>
                      )}
                      {job.log_lines && job.log_lines.length > 0 ? (
                        <div className="max-h-48 overflow-y-auto font-mono text-xs text-gray-400 bg-gray-950 rounded p-2 space-y-0.5">
                          {job.log_lines.map((line, i) => <div key={i}>{line}</div>)}
                        </div>
                      ) : (
                        <div className="text-xs text-gray-600">No log lines captured yet.</div>
                      )}
                    </td>
                  </tr>
                )}
              </>
            ))
          )}
        </tbody>
      </table>
    </div>
  )
}
