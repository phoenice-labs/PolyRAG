import { describe, it, expect, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ResultCard from '../components/ResultCard/ResultCard'
import type { SearchResultItem } from '../api/search'

vi.mock('../api/client', () => ({
  api: {
    post: vi.fn().mockResolvedValue({ data: {} }),
    create: vi.fn(),
  },
}))

const mockResult: SearchResultItem = {
  chunk_id: 'chunk-123',
  text: 'This is the chunk text content that should be displayed in the result card.',
  score: 0.87,
  metadata: {
    source: 'document.pdf',
    version: '1.0',
    classification: 'PUBLIC',
  },
  confidence: 0.9,
}

describe('ResultCard', () => {
  it('renders chunk text', () => {
    render(<ResultCard result={mockResult} backend="faiss" />)
    expect(screen.getByText(/This is the chunk text content/)).toBeTruthy()
  })

  it('renders score bar', () => {
    render(<ResultCard result={mockResult} backend="faiss" />)
    expect(screen.getByTestId('score-bar')).toBeTruthy()
  })

  it('renders backend badge', () => {
    render(<ResultCard result={mockResult} backend="faiss" />)
    expect(screen.getByText('faiss')).toBeTruthy()
  })

  it('expands metadata accordion', () => {
    render(<ResultCard result={mockResult} backend="faiss" />)
    const metaBtn = screen.getByText(/Provenance/)
    fireEvent.click(metaBtn)
    expect(screen.getByText('document.pdf')).toBeTruthy()
  })

  it('renders thumbs up button', () => {
    render(<ResultCard result={mockResult} />)
    expect(screen.getByLabelText('thumbs up')).toBeTruthy()
  })

  it('renders thumbs down button', () => {
    render(<ResultCard result={mockResult} />)
    expect(screen.getByLabelText('thumbs down')).toBeTruthy()
  })

  it('shows score value', () => {
    render(<ResultCard result={mockResult} />)
    expect(screen.getByText('0.870')).toBeTruthy()
  })
})
