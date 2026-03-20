import { useEffect, useRef } from 'react'

interface LogStreamProps {
  lines: string[]
}

function getLineColor(line: string): string {
  const upper = line.toUpperCase()
  if (upper.includes('ERROR')) return 'text-red-400'
  if (upper.includes('WARN')) return 'text-yellow-400'
  if (upper.includes('SUCCESS') || upper.includes('DONE') || upper.includes('COMPLETE')) return 'text-green-400'
  return 'text-gray-400'
}

function formatTimestamp(): string {
  return new Date().toLocaleTimeString('en-US', { hour12: false })
}

export default function LogStream({ lines }: LogStreamProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (bottomRef.current?.scrollIntoView) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [lines])

  return (
    <div className="h-64 overflow-y-auto bg-gray-900 rounded border border-gray-700 p-2 font-mono text-xs">
      {lines.length === 0 ? (
        <div className="text-gray-600 italic">Waiting for logs...</div>
      ) : (
        lines.map((line, i) => (
          <div key={i} className={`flex gap-2 ${getLineColor(line)}`}>
            <span className="text-gray-600 shrink-0">{formatTimestamp()}</span>
            <span>{line}</span>
          </div>
        ))
      )}
      <div ref={bottomRef} />
    </div>
  )
}
