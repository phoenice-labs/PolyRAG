import axios from 'axios'

export const api = axios.create({ baseURL: '/api' })

export const streamIngestLogs = (
  jobId: string,
  onLine: (line: string) => void,
  onDone: () => void
): (() => void) => {
  const evtSource = new EventSource(`/api/ingest/${jobId}/stream`)

  evtSource.onmessage = (e: MessageEvent) => {
    const data: string = e.data
    // Server signals completion via STATUS:done / STATUS:error
    if (data.startsWith('STATUS:')) {
      evtSource.close()
      onDone()
    } else {
      onLine(data)
    }
  }

  // onerror fires on network failure OR on normal server-close + reconnect attempt
  evtSource.onerror = () => {
    evtSource.close()
    onDone()
  }

  return () => evtSource.close()
}
