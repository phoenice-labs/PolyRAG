import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import MethodToggle from '../components/MethodToggle/MethodToggle'
import { useStore } from '../store'

beforeEach(() => {
  useStore.setState({
    retrievalMethods: {
      enable_dense: true,
      enable_bm25: true,
      enable_graph: true,
      enable_rerank: true,
      enable_mmr: true,
      enable_rewrite: false,
      enable_multi_query: false,
      enable_hyde: false,
      enable_raptor: false,
      enable_contextual_rerank: false,
    },
  })
})

describe('MethodToggle', () => {
  it('renders all 10 methods', () => {
    render(<MethodToggle />)
    const methods = [
      'Dense Vector', 'BM25 Keyword', 'Knowledge Graph', 'Cross-Encoder Rerank', 'MMR Diversity',
      'Query Rewrite', 'Multi-Query', 'HyDE', 'RAPTOR', 'Contextual Rerank',
    ]
    methods.forEach((m) => {
      expect(screen.getByLabelText(m)).toBeTruthy()
    })
  })

  it('shows "Always Available" group', () => {
    render(<MethodToggle />)
    expect(screen.getByText('Always Available')).toBeTruthy()
  })

  it('shows "LLM-Required" group', () => {
    render(<MethodToggle />)
    expect(screen.getByText('LLM-Required')).toBeTruthy()
  })

  it('toggle switch changes state', () => {
    render(<MethodToggle />)
    const rewriteToggle = screen.getByLabelText('Query Rewrite')
    expect(rewriteToggle).toHaveAttribute('aria-checked', 'false')
    fireEvent.click(rewriteToggle)
    expect(useStore.getState().retrievalMethods.enable_rewrite).toBe(true)
  })

  it('enabled method has aria-checked true', () => {
    render(<MethodToggle />)
    const denseToggle = screen.getByLabelText('Dense Vector')
    expect(denseToggle).toHaveAttribute('aria-checked', 'true')
  })
})
