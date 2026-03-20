import { useState, useEffect } from 'react'
import { listJobs, type IngestJob } from '../api/ingest'

const STATUS_COLORS: Record<IngestJob['status'], string> = {
  pending: 'text-yellow-400',
  running: 'text-blue-400',
  done: 'text-green-400',
  error: 'text-red-400',
}

type FilterStatus = 'all' | IngestJob['status']

export default function JobHistory() {
  const [jobs, setJobs] = useState<IngestJob[]>([])
  const [loading, setLoading] = useState(false)
  const [filter, setFilter] = useState<FilterStatus>('all')
  const [expanded, setExpanded] = useState<string | null>(null)

  const fetchJobs = async () => {
    setLoading(true)
    try {
      const data = await listJobs()
      setJobs(data)
    } catch {
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { fetchJobs() }, [])

  const clearCompleted = () => setJobs((prev) => prev.filter((j) => j.status !== 'done'))

  const filtered = filter === 'all' ? jobs : jobs.filter((j) => j.status === filter)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Job History</h1>
        <div className="flex gap-2">
          <button onClick={fetchJobs} disabled={loading}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-sm text-gray-200 rounded">
            {loading ? 'Loading...' : 'Refresh'}
          </button>
          <button onClick={clearCompleted}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-sm text-gray-200 rounded">
            Clear Completed
          </button>
        </div>
      </div>

      <div className="flex gap-2">
        {(['all', 'pending', 'running', 'done', 'error'] as FilterStatus[]).map((s) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`px-3 py-1 rounded text-sm ${filter === s ? 'bg-brand-500 text-white' : 'bg-gray-800 text-gray-400 hover:text-gray-200'}`}
          >
            {s}
          </button>
        ))}
      </div>

      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="px-3 py-2 text-left text-gray-400">Job ID</th>
            <th className="px-3 py-2 text-left text-gray-400">Backend</th>
            <th className="px-3 py-2 text-left text-gray-400">Status</th>
            <th className="px-3 py-2 text-left text-gray-400">Created</th>
            <th className="px-3 py-2 text-left text-gray-400">Duration</th>
            <th className="px-3 py-2 text-left text-gray-400">Chunks</th>
            <th className="px-3 py-2 text-left text-gray-400">Errors</th>
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
                  <td className="px-3 py-2 text-gray-400 font-mono text-xs">{job.job_id.slice(0, 8)}...</td>
                  <td className="px-3 py-2 text-gray-300">{job.backend}</td>
                  <td className={`px-3 py-2 font-medium ${STATUS_COLORS[job.status]}`}>{job.status}</td>
                  <td className="px-3 py-2 text-gray-400">{new Date(job.created_at).toLocaleString()}</td>
                  <td className="px-3 py-2 text-gray-400">{job.duration ? `${job.duration.toFixed(1)}s` : '—'}</td>
                  <td className="px-3 py-2 text-gray-400">{job.chunk_count ?? '—'}</td>
                  <td className="px-3 py-2 text-gray-400">{job.errors?.length ?? 0}</td>
                </tr>
                {expanded === job.job_id && job.errors && job.errors.length > 0 && (
                  <tr key={`${job.job_id}-expanded`} className="border-b border-gray-800 bg-gray-900/50">
                    <td colSpan={7} className="px-6 py-2">
                      <div className="text-xs text-gray-400">
                        <div className="font-medium text-gray-300 mb-1">Errors:</div>
                        {job.errors.map((e, i) => (
                          <div key={i} className="text-red-400">{e}</div>
                        ))}
                      </div>
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
