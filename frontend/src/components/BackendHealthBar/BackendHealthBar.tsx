import { useEffect, useState } from 'react'
import { getBackends, type BackendStatus } from '../../api/backends'
import { useStore } from '../../store'

export default function BackendHealthBar() {
  const [backends, setBackends] = useState<BackendStatus[]>([])
  const setBackendStatus = useStore((s) => s.setBackendStatus)

  const fetchBackends = async () => {
    try {
      const data = await getBackends()
      setBackends(data)
      data.forEach((b) => setBackendStatus(b.name, b.status))
    } catch {
      // silently fail during polling
    }
  }

  useEffect(() => {
    fetchBackends()
    const interval = setInterval(fetchBackends, 10000)
    return () => clearInterval(interval)
  }, [])

  const dotColor = (status: BackendStatus['status']) => {
    if (status === 'ok') return 'bg-green-400'
    if (status === 'error') return 'bg-red-400'
    return 'bg-gray-500'
  }

  return (
    <div className="h-8 bg-gray-900 border-b border-gray-800 flex items-center px-4 gap-4 shrink-0">
      <span className="text-xs text-gray-500">Backends:</span>
      {backends.length === 0 ? (
        <span className="text-xs text-gray-600">Checking...</span>
      ) : (
        backends.map((b) => (
          <div key={b.name} className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${dotColor(b.status)}`} />
            <span className="text-xs text-gray-400">{b.name}</span>
            {b.ping_ms !== undefined && (
              <span className="text-xs text-gray-600">{b.ping_ms}ms</span>
            )}
          </div>
        ))
      )}
    </div>
  )
}
