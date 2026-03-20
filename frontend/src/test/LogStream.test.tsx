import { describe, it, expect } from 'vitest'
import { render, screen } from '@testing-library/react'
import LogStream from '../components/LogStream/LogStream'

describe('LogStream', () => {
  it('shows waiting message when no lines', () => {
    render(<LogStream lines={[]} />)
    expect(screen.getByText(/Waiting for logs/)).toBeTruthy()
  })

  it('renders log lines', () => {
    const lines = ['INFO: Starting process', 'INFO: Processing chunk 1']
    render(<LogStream lines={lines} />)
    expect(screen.getByText('INFO: Starting process')).toBeTruthy()
    expect(screen.getByText('INFO: Processing chunk 1')).toBeTruthy()
  })

  it('applies red color for ERROR lines', () => {
    const { container } = render(<LogStream lines={['ERROR: Something went wrong']} />)
    const errorLine = container.querySelector('.text-red-400')
    expect(errorLine).toBeTruthy()
    expect(errorLine?.textContent).toContain('ERROR: Something went wrong')
  })

  it('applies yellow color for WARN lines', () => {
    const { container } = render(<LogStream lines={['WARN: Low memory']} />)
    const warnLine = container.querySelector('.text-yellow-400')
    expect(warnLine).toBeTruthy()
    expect(warnLine?.textContent).toContain('WARN: Low memory')
  })

  it('applies green color for SUCCESS lines', () => {
    const { container } = render(<LogStream lines={['SUCCESS: Ingest complete']} />)
    const successLine = container.querySelector('.text-green-400')
    expect(successLine).toBeTruthy()
    expect(successLine?.textContent).toContain('SUCCESS: Ingest complete')
  })

  it('applies gray color for INFO lines', () => {
    const { container } = render(<LogStream lines={['INFO: Normal message']} />)
    const infoLine = container.querySelector('.text-gray-400')
    expect(infoLine).toBeTruthy()
  })
})
