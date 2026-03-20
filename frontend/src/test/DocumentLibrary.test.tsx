import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import DocumentLibrary from '../pages/DocumentLibrary'

// Mock api/backends
vi.mock('../api/backends', () => ({
  getBackends: vi.fn().mockResolvedValue([
    { name: 'faiss', status: 'available' },
    { name: 'chromadb', status: 'available' },
  ]),
  getCollections: vi.fn().mockResolvedValue([
    { name: 'polyrag_docs', chunk_count: 333, index_type: 'faiss' },
    { name: 'hamlet', chunk_count: 42, index_type: 'faiss' },
  ]),
  deleteCollection: vi.fn().mockResolvedValue({ deleted: true }),
  clearAllCollections: vi.fn().mockResolvedValue({ deleted: ['polyrag_docs', 'hamlet'], count: 2 }),
}))

describe('DocumentLibrary', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('renders backend tabs', async () => {
    render(<DocumentLibrary />)
    expect(screen.getByText('faiss')).toBeInTheDocument()
    expect(screen.getByText('chromadb')).toBeInTheDocument()
    expect(screen.getByText('qdrant')).toBeInTheDocument()
  })

  it('shows collections after load', async () => {
    render(<DocumentLibrary />)
    await waitFor(() => {
      expect(screen.getByText('polyrag_docs')).toBeInTheDocument()
      expect(screen.getByText('hamlet')).toBeInTheDocument()
    })
  })

  it('shows chunk counts', async () => {
    render(<DocumentLibrary />)
    await waitFor(() => {
      expect(screen.getByText('333')).toBeInTheDocument()
      expect(screen.getByText('42')).toBeInTheDocument()
    })
  })

  it('shows Delete buttons for each collection', async () => {
    render(<DocumentLibrary />)
    await waitFor(() => {
      const deleteBtns = screen.getAllByText('Delete')
      expect(deleteBtns.length).toBe(2)
    })
  })

  it('shows Clear All button', async () => {
    render(<DocumentLibrary />)
    expect(screen.getByText(/Clear All/)).toBeInTheDocument()
  })

  it('shows empty state when no collections', async () => {
    const { getCollections } = await import('../api/backends')
    vi.mocked(getCollections).mockResolvedValueOnce([])
    render(<DocumentLibrary />)
    await waitFor(() => {
      expect(screen.getByText(/No collections found/)).toBeInTheDocument()
    })
  })
})
