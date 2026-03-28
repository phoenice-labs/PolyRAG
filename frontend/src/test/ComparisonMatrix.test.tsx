import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import ComparisonMatrix from '../pages/ComparisonMatrix'
import { useStore } from '../store'

vi.mock('../api/compare', () => ({
  startComparison: vi.fn().mockResolvedValue({ job_id: 'test-job', status: 'done', results: [] }),
  getCompareJob: vi.fn().mockResolvedValue({ job_id: 'test-job', status: 'done', results: [] }),
  getSampleQueries: vi.fn().mockResolvedValue([]),
}))

beforeEach(() => {
  useStore.setState({ selectedBackends: ['faiss', 'chromadb'] })
})

const renderWithRouter = (ui: React.ReactElement) =>
  render(<MemoryRouter>{ui}</MemoryRouter>)

describe('ComparisonMatrix', () => {
  it('renders page heading and step sections', () => {
    renderWithRouter(<ComparisonMatrix />)
    expect(screen.getByText('Backend Comparison')).toBeTruthy()
    expect(screen.getByText(/Step 1/)).toBeTruthy()
    expect(screen.getByText(/Step 2/)).toBeTruthy()
    expect(screen.getByText(/Step 3/)).toBeTruthy()
  })

  it('renders Run Comparison button', () => {
    renderWithRouter(<ComparisonMatrix />)
    expect(screen.getByText(/Run Comparison/)).toBeTruthy()
  })

  it('toggles between existing and paste modes', () => {
    renderWithRouter(<ComparisonMatrix />)
    const pasteBtn = screen.getByText(/Paste text/)
    fireEvent.click(pasteBtn)
    expect(screen.getByPlaceholderText(/Paste document text/)).toBeTruthy()
  })

  it('shows collection name input in existing mode', () => {
    renderWithRouter(<ComparisonMatrix />)
    expect(screen.getByPlaceholderText('or type a name')).toBeTruthy()
  })

  it('shows Full Retrieval checkbox', () => {
    renderWithRouter(<ComparisonMatrix />)
    expect(screen.getByText(/Full Retrieval/)).toBeTruthy()
  })
})
