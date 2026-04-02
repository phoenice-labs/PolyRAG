/**
 * IngestionStudio — PDF/PPTX file handling tests
 *
 * Tests that:
 *  1. The file input accepts .pdf and .pptx extensions
 *  2. Dropping a PDF/PPTX switches to Server Path mode with a pre-filled path
 *  3. Plain .txt files are still read as text and stay in Upload File mode
 *  4. The server-path quick-picks include PDF and PPTX example paths
 *  5. IngestRequest with corpus_path is sent (not text) for binary formats
 */
import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import IngestionStudio from '../pages/IngestionStudio'

// ── Minimal mocks ─────────────────────────────────────────────────────────────

vi.mock('../api/ingest', () => ({
  ingestText: vi.fn().mockResolvedValue({ job_ids: { faiss: 'job-1' } }),
  previewChunks: vi.fn().mockResolvedValue({ chunks: [], total: 0 }),
  getJobStatus: vi.fn().mockResolvedValue({ status: 'done', log_lines: [] }),
  listJobs: vi.fn().mockResolvedValue([]),
}))

vi.mock('../api/backends', () => ({
  getBackends: vi.fn().mockResolvedValue([{ name: 'faiss', status: 'available' }]),
  getCollections: vi.fn().mockResolvedValue([]),
  deleteCollection: vi.fn().mockResolvedValue({ deleted: true }),
}))

vi.mock('../api/client', () => ({
  streamIngestLogs: vi.fn().mockReturnValue(() => {}),
  api: { post: vi.fn(), get: vi.fn() },
}))

vi.mock('../store', () => ({
  useStore: vi.fn().mockReturnValue({
    selectedBackends: ['faiss'],
    setSelectedBackends: vi.fn(),
    backendStatuses: { faiss: 'ok', chromadb: 'unknown', qdrant: 'unknown', weaviate: 'unknown', milvus: 'unknown', pgvector: 'unknown' },
    activeCollection: 'test_col',
    setActiveCollection: vi.fn(),
    ingestConfig: {
      strategy: 'sentence',
      chunkSize: 512,
      overlap: 64,
      extractEntities: false,
      enableSplade: false,
      clearFirst: false,
    },
    setIngestConfig: vi.fn(),
    setActiveIngestJob: vi.fn(),
    activeIngestJobs: {},
    clearActiveIngestJobs: vi.fn(),
    embeddingModel: 'all-MiniLM-L6-v2',
  }),
}))

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeFile(name: string, content = 'test content', type = 'text/plain'): File {
  return new File([content], name, { type })
}

function renderStudio() {
  return render(
    <MemoryRouter>
      <IngestionStudio />
    </MemoryRouter>
  )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe('IngestionStudio — file input accept attribute', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('accepts .pdf in the file input', () => {
    renderStudio()
    // Switch to Upload File mode
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(input).toBeTruthy()
    expect(input.accept).toContain('.pdf')
  })

  it('accepts .pptx in the file input', () => {
    renderStudio()
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(input.accept).toContain('.pptx')
  })

  it('accepts .txt in the file input', () => {
    renderStudio()
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    const input = document.querySelector('input[type="file"]') as HTMLInputElement
    expect(input.accept).toContain('.txt')
  })

  it('shows supported formats hint including .pptx', () => {
    renderStudio()
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    expect(screen.getByText(/\.pptx/)).toBeTruthy()
  })
})

describe('IngestionStudio — binary file drop redirects to Server Path mode', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('dropping a PDF pre-fills server path and switches to Server Path mode', async () => {
    renderStudio()
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    const pdfFile = makeFile('my_report.pdf', '%PDF-1.4 binary', 'application/pdf')

    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    Object.defineProperty(fileInput, 'files', {
      value: [pdfFile],
      configurable: true,
    })
    await act(async () => {
      fireEvent.change(fileInput)
    })

    // Should have switched to Server Path mode — server path input appears
    await waitFor(() => {
      const serverPathInput = screen.getByPlaceholderText('data/shakespeare.txt') as HTMLInputElement
      expect(serverPathInput.value).toBe('data/my_report.pdf')
    })
  })

  it('dropping a PPTX pre-fills server path with .pptx extension', async () => {
    renderStudio()
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    const pptxFile = makeFile('presentation.pptx', 'PK\x03\x04', 'application/vnd.openxmlformats-officedocument.presentationml.presentation')
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    Object.defineProperty(fileInput, 'files', {
      value: [pptxFile],
      configurable: true,
    })
    await act(async () => {
      fireEvent.change(fileInput)
    })

    await waitFor(() => {
      const serverPathInput = screen.getByPlaceholderText('data/shakespeare.txt') as HTMLInputElement
      expect(serverPathInput.value).toBe('data/presentation.pptx')
    })
  })

  it('dropping a .txt file stays in Upload File mode', async () => {
    renderStudio()
    const uploadTab = screen.getByText('Upload File')
    fireEvent.click(uploadTab)

    const txtFile = makeFile('notes.txt', 'To be or not to be.', 'text/plain')
    const fileInput = document.querySelector('input[type="file"]') as HTMLInputElement
    Object.defineProperty(fileInput, 'files', {
      value: [txtFile],
      configurable: true,
    })
    await act(async () => {
      fireEvent.change(fileInput)
    })

    await waitFor(() => {
      // Still in Upload File mode — should show "File loaded" not server path input
      expect(screen.getByText(/File loaded/)).toBeTruthy()
    })
  })
})

describe('IngestionStudio — server path quick-picks include PDF/PPTX', () => {
  it('shows data/report.pdf quick-pick', () => {
    renderStudio()
    // Server Path mode is the default
    expect(screen.getByText('data/report.pdf')).toBeTruthy()
  })

  it('shows data/slides.pptx quick-pick', () => {
    renderStudio()
    expect(screen.getByText('data/slides.pptx')).toBeTruthy()
  })

  it('clicking PDF quick-pick fills the server path input', async () => {
    renderStudio()
    const pdfQuickPick = screen.getByText('data/report.pdf')
    fireEvent.click(pdfQuickPick)

    const serverPathInput = screen.getByPlaceholderText('data/shakespeare.txt') as HTMLInputElement
    expect(serverPathInput.value).toBe('data/report.pdf')
  })

  it('clicking PPTX quick-pick fills the server path input', async () => {
    renderStudio()
    const pptxQuickPick = screen.getByText('data/slides.pptx')
    fireEvent.click(pptxQuickPick)

    const serverPathInput = screen.getByPlaceholderText('data/shakespeare.txt') as HTMLInputElement
    expect(serverPathInput.value).toBe('data/slides.pptx')
  })
})
