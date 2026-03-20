import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import BackendSelector from '../components/BackendSelector/BackendSelector'
import { useStore } from '../store'

// Reset store state before each test
beforeEach(() => {
  useStore.setState({
    selectedBackends: ['faiss', 'chromadb'],
    backendStatuses: {},
  })
})

describe('BackendSelector', () => {
  it('renders all 6 backends', () => {
    render(<BackendSelector />)
    const backends = ['faiss', 'chromadb', 'qdrant', 'weaviate', 'milvus', 'pgvector']
    backends.forEach((b) => {
      expect(screen.getByLabelText(b)).toBeTruthy()
    })
  })

  it('clicking an unselected backend selects it', () => {
    render(<BackendSelector />)
    const qdrant = screen.getByLabelText('qdrant')
    expect(qdrant).not.toBeChecked()
    fireEvent.click(qdrant)
    expect(useStore.getState().selectedBackends).toContain('qdrant')
  })

  it('clicking a selected backend deselects it', () => {
    render(<BackendSelector />)
    const faiss = screen.getByLabelText('faiss')
    expect(faiss).toBeChecked()
    fireEvent.click(faiss)
    expect(useStore.getState().selectedBackends).not.toContain('faiss')
  })

  it('shows gray status dot for unknown backend status', () => {
    render(<BackendSelector />)
    const unknownDots = screen.getAllByTestId('status-dot-unknown')
    expect(unknownDots.length).toBeGreaterThan(0)
  })

  it('shows green status dot for ok backend', () => {
    useStore.setState({ backendStatuses: { faiss: 'ok' } })
    render(<BackendSelector />)
    const okDots = screen.getAllByTestId('status-dot-ok')
    expect(okDots.length).toBeGreaterThan(0)
  })

  it('shows red status dot for error backend', () => {
    useStore.setState({ backendStatuses: { chromadb: 'error' } })
    render(<BackendSelector />)
    const errorDots = screen.getAllByTestId('status-dot-error')
    expect(errorDots.length).toBeGreaterThan(0)
  })
})
