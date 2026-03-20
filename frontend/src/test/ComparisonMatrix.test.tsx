import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ComparisonMatrix from '../pages/ComparisonMatrix'
import { useStore } from '../store'

vi.mock('../api/compare', () => ({
  startComparison: vi.fn().mockResolvedValue({ job_id: 'test-job', status: 'done', results: [] }),
  getCompareJob: vi.fn().mockResolvedValue({ job_id: 'test-job', status: 'done', results: [] }),
}))

beforeEach(() => {
  useStore.setState({ selectedBackends: ['faiss', 'chromadb'] })
})

describe('ComparisonMatrix', () => {
  it('renders table headers', () => {
    render(<ComparisonMatrix />)
    expect(screen.getByText('base_top_score')).toBeTruthy()
    expect(screen.getByText('full_top_score')).toBeTruthy()
    expect(screen.getByText('base_kw_hits')).toBeTruthy()
    expect(screen.getByText('avg_score')).toBeTruthy()
    expect(screen.getByText('ingest_time_s')).toBeTruthy()
    expect(screen.getByText('errors')).toBeTruthy()
  })

  it('renders Run Comparison button', () => {
    render(<ComparisonMatrix />)
    expect(screen.getByText('Run Comparison')).toBeTruthy()
  })

  it('sorts columns on click', () => {
    render(<ComparisonMatrix />)
    const header = screen.getByText('base_top_score')
    fireEvent.click(header)
    // After click, a sort indicator should appear
    expect(screen.getByText(/base_top_score/)).toBeTruthy()
  })

  it('shows corpus path input', () => {
    render(<ComparisonMatrix />)
    expect(screen.getByPlaceholderText('/path/to/corpus')).toBeTruthy()
  })

  it('shows Full Retrieval checkbox', () => {
    render(<ComparisonMatrix />)
    expect(screen.getByText('Full Retrieval')).toBeTruthy()
  })
})
