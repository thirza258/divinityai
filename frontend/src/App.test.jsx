import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('App', () => {
  beforeEach(() => {
    mockFetch.mockReset();
  });

  // =========================================================================
  // Rendering tests
  // =========================================================================

  it('renders the app title', () => {
    render(<App />);
    const elements = screen.getAllByText('DivinityAI');
    expect(elements.length).toBeGreaterThanOrEqual(1);
    // The h1 heading should exist
    expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('DivinityAI');
  });

  it('renders the subtitle', () => {
    render(<App />);
    expect(screen.getByText('Quran & Hadith QA')).toBeInTheDocument();
  });

  it('renders the welcome message', () => {
    render(<App />);
    expect(
      screen.getByText(/Welcome to DivinityAI/)
    ).toBeInTheDocument();
  });

  it('renders the input field', () => {
    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    expect(input).toBeInTheDocument();
  });

  it('renders the language selector', () => {
    render(<App />);
    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
    expect(screen.getByText('EN')).toBeInTheDocument();
  });

  it('renders the submit button', () => {
    render(<App />);
    const button = screen.getByRole('button', { name: /Ask/i });
    expect(button).toBeInTheDocument();
  });

  it('renders example question buttons', () => {
    render(<App />);
    expect(
      screen.getByText('What does the Quran say about patience?')
    ).toBeInTheDocument();
    expect(
      screen.getByText('Hadith about seeking knowledge')
    ).toBeInTheDocument();
    expect(
      screen.getByText('What is the ruling on zakat?')
    ).toBeInTheDocument();
  });

  it('renders the footer disclaimer', () => {
    render(<App />);
    const elements = screen.getAllByText(/Not a fatwa-issuing system/);
    expect(elements.length).toBeGreaterThanOrEqual(1);
  });

  // =========================================================================
  // Interaction tests
  // =========================================================================

  it('disables submit button when input is empty', () => {
    render(<App />);
    const button = screen.getByRole('button', { name: /Ask/i });
    expect(button).toBeDisabled();
  });

  it('enables submit button when input has text', async () => {
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'What is patience?');
    expect(button).not.toBeDisabled();
  });

  it('updates input value when typing', async () => {
    const user = userEvent.setup();
    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);

    await user.type(input, 'Hello');
    expect(input).toHaveValue('Hello');
  });

  it('clears input after submitting', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'The Quran says...',
          sources: [],
          citations: [],
          intent: 'general',
          safety: {},
          pipeline_meta: {},
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'Test query');
    await user.click(button);

    await waitFor(() => {
      expect(input).toHaveValue('');
    });
  });

  it('clicking an example question fills the input', async () => {
    const user = userEvent.setup();
    render(<App />);
    const exampleBtn = screen.getByText('What does the Quran say about patience?');

    await user.click(exampleBtn);

    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    expect(input).toHaveValue('What does the Quran say about patience?');
  });

  it('changes language when selector is changed', async () => {
    const user = userEvent.setup();
    render(<App />);
    const select = screen.getByRole('combobox');

    await user.selectOptions(select, 'ar');
    expect(select).toHaveValue('ar');

    await user.selectOptions(select, 'id');
    expect(select).toHaveValue('id');
  });

  // =========================================================================
  // API interaction tests
  // =========================================================================

  it('sends a POST request with correct payload', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'Response text',
          sources: [],
          citations: [],
          intent: 'general',
          safety: {},
          pipeline_meta: {},
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'What is Islam?');
    await user.click(button);

    expect(mockFetch).toHaveBeenCalledTimes(1);
    const [url, options] = mockFetch.mock.calls[0];
    expect(url).toContain('/api/v1/query');
    expect(options.method).toBe('POST');
    expect(options.headers['Content-Type']).toBe('application/json');

    const body = JSON.parse(options.body);
    expect(body.query).toBe('What is Islam?');
    expect(body.language).toBe('en');
    expect(body.max_sources).toBe(5);
  });

  it('displays the bot response after successful API call', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'The Quran emphasizes patience.',
          sources: [
            {
              source_tag: 'Q 2:153',
              corpus: 'quran',
              text_ar: '...',
              text_en: 'O you who have believed...',
              verification_status: 'exact',
              retrieval_score: 0.94,
            },
          ],
          citations: ['Q 2:153'],
          intent: 'quran_verse',
          safety: {
            hallucination_detected: false,
            flagged_spans: [],
            fatwa_boundary_triggered: false,
            disclaimer: null,
          },
          pipeline_meta: { phase: 1, llm_calls: 1, elapsed: 0.5 },
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'patience');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText('The Quran emphasizes patience.')).toBeInTheDocument();
    });
  });

  it('shows sources section when sources are returned', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'Answer with sources.',
          sources: [
            {
              source_tag: 'Q 2:153',
              corpus: 'quran',
              text_ar: '...',
              text_en: 'O you who have believed...',
              verification_status: 'exact',
              retrieval_score: 0.94,
            },
          ],
          citations: ['Q 2:153'],
          intent: 'quran_verse',
          safety: {},
          pipeline_meta: {},
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'test');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText(/Sources \(1\)/)).toBeInTheDocument();
    });
  });

  it('shows safety disclaimer when present', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'Regarding talaq...',
          sources: [],
          citations: [],
          intent: 'fiqh',
          safety: {
            hallucination_detected: false,
            flagged_spans: [],
            fatwa_boundary_triggered: true,
            disclaimer: 'For a definitive ruling, please consult a qualified scholar.',
          },
          pipeline_meta: {},
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'divorce');
    await user.click(button);

    await waitFor(() => {
      expect(
        screen.getByText(/For a definitive ruling, please consult a qualified scholar/)
      ).toBeInTheDocument();
    });
  });

  it('shows error message on API failure', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: false,
      json: () =>
        Promise.resolve({
          error: 'Pipeline processing failed',
          detail: 'Something went wrong',
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'test');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText(/Sorry, something went wrong/)).toBeInTheDocument();
    });
  });

  it('shows error message on network failure', async () => {
    const user = userEvent.setup();
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'test');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText(/Sorry, something went wrong/)).toBeInTheDocument();
    });
  });

  // =========================================================================
  // Citation formatting tests
  // =========================================================================

  it('renders Quran citations with styled spans', async () => {
    const user = userEvent.setup();
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'Allah says [Q 2:255] is the greatest verse.',
          sources: [],
          citations: ['Q 2:255'],
          intent: 'quran_verse',
          safety: {},
          pipeline_meta: {},
        }),
    });

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'ayatul kursi');
    await user.click(button);

    await waitFor(() => {
      expect(screen.getByText('[Q 2:255]')).toBeInTheDocument();
    });
  });

  // =========================================================================
  // Loading state tests
  // =========================================================================

  it('shows loading indicator while waiting for response', async () => {
    const user = userEvent.setup();
    // Create a promise that doesn't resolve immediately
    let resolvePromise;
    const promise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    mockFetch.mockReturnValueOnce(promise);

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'test');
    await user.click(button);

    expect(screen.getByText(/Searching Quran & Hadith/)).toBeInTheDocument();

    // Clean up
    resolvePromise({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'Done',
          sources: [],
          citations: [],
          intent: 'general',
          safety: {},
          pipeline_meta: {},
        }),
    });
  });

  it('disables input while loading', async () => {
    const user = userEvent.setup();
    let resolvePromise;
    const promise = new Promise((resolve) => {
      resolvePromise = resolve;
    });
    mockFetch.mockReturnValueOnce(promise);

    render(<App />);
    const input = screen.getByPlaceholderText(/Ask about the Quran or Hadith/);
    const button = screen.getByRole('button', { name: /Ask/i });

    await user.type(input, 'test');
    await user.click(button);

    expect(input).toBeDisabled();

    // Clean up
    resolvePromise({
      ok: true,
      json: () =>
        Promise.resolve({
          answer: 'Done',
          sources: [],
          citations: [],
          intent: 'general',
          safety: {},
          pipeline_meta: {},
        }),
    });
  });
});