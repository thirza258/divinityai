import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

const account = { id: 7, name: 'Amina', email: 'amina@example.com', memory_enabled: true };
const savedChat = { id: 'chat-1', title: 'Studying patience', language: 'en', updated_at: '2026-09-17T10:00:00Z' };
const answer = { answer: 'Patience and prayer. [Q 2:153]', sources: [{ source_tag: 'Q 2:153', text_en: 'Seek help through patience and prayer.', verification_status: 'exact' }], citations: ['Q 2:153'], safety: {}, pipeline_meta: {} };
let sessionUser;
let conversations;
let memories;
let queryImpl;
let fetchMock;

function json(data, status = 200) {
  const payload = structuredClone(data);
  return Promise.resolve({ ok: status < 400, status, json: () => Promise.resolve(payload) });
}

describe('Accounts, chat history, and memory', () => {
  beforeEach(() => {
    sessionUser = { ...account };
    conversations = [{ ...savedChat }];
    memories = [];
    queryImpl = () => json({ ...answer, conversation: savedChat });
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    fetchMock = vi.fn((url, options = {}) => {
      const path = new URL(url, 'http://localhost').pathname;
      const method = options.method || 'GET';
      const body = options.body ? JSON.parse(options.body) : {};
      if (path.endsWith('/health')) return json({ status: 'ok', checks: {} });
      if (path.endsWith('/auth/session')) return json({ user: sessionUser, csrf_token: 'session-csrf' });
      if (path.endsWith('/auth/login') || path.endsWith('/auth/register')) { sessionUser = { ...account }; return json({ user: sessionUser, csrf_token: 'rotated-csrf' }); }
      if (path.endsWith('/auth/logout')) { sessionUser = null; return json({ user: null, csrf_token: 'guest-csrf' }); }
      if (path.endsWith('/auth/profile')) { sessionUser = { ...sessionUser, ...body }; return json({ user: sessionUser }); }
      if (path.endsWith('/query')) return queryImpl(options);
      if (path.endsWith('/conversations')) {
        if (method === 'DELETE') { conversations = []; return json(null, 204); }
        return json({ results: conversations, count: conversations.length, next: null });
      }
      if (path.endsWith('/conversations/chat-1')) {
        if (method === 'PATCH') { conversations[0] = { ...conversations[0], ...body }; return json(conversations[0]); }
        if (method === 'DELETE') { conversations = []; return json(null, 204); }
        return json({ ...savedChat, messages: [
          { role: 'user', content: 'My previous question', response: {} },
          { role: 'assistant', content: answer.answer, response: answer },
        ] });
      }
      if (path.endsWith('/memories')) {
        if (method === 'POST') { const memory = { id: 'memory-1', ...body }; memories.push(memory); return json(memory, 201); }
        if (method === 'DELETE') { memories = []; return json(null, 204); }
        return json(memories);
      }
      if (path.endsWith('/memories/memory-1')) {
        if (method === 'DELETE') { memories = []; return json(null, 204); }
        memories[0] = { ...memories[0], ...body }; return json(memories[0]);
      }
      throw new Error(`Unexpected endpoint: ${method} ${path}`);
    });
    vi.stubGlobal('fetch', fetchMock);
  });

  afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); });

  it('creates an account with CSRF protection and clears the guest conversation', async () => {
    sessionUser = null;
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Create account' }));
    const dialog = screen.getByRole('dialog', { name: 'Create your account' });
    await user.type(within(dialog).getByLabelText('Name'), 'Amina');
    await user.type(within(dialog).getByLabelText('Email'), 'amina@example.com');
    await user.type(within(dialog).getByLabelText('Password'), 'QuietRiver!42');
    await user.click(within(dialog).getByRole('button', { name: 'Create account' }));
    expect(await screen.findByRole('button', { name: 'Sign out' })).toBeInTheDocument();
    const [, options] = fetchMock.mock.calls.find(([url]) => url.endsWith('/auth/register'));
    expect(options.credentials).toBe('include');
    expect(options.headers['X-CSRFToken']).toBe('session-csrf');
    expect(JSON.parse(options.body)).toEqual({ name: 'Amina', email: 'amina@example.com', password: 'QuietRiver!42' });
  });

  it('reopens saved messages with sources and continues the same conversation', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Open Studying patience' }));
    expect(await screen.findByText('My previous question')).toBeInTheDocument();
    expect(screen.getByText('Sources (1)')).toBeInTheDocument();
    await user.type(screen.getByRole('textbox', { name: 'Your question' }), 'Explain that verse');
    await user.click(screen.getByRole('button', { name: 'Ask' }));
    await waitFor(() => expect(screen.getByRole('button', { name: 'Ask' })).toBeDisabled());
    const [, options] = fetchMock.mock.calls.find(([url]) => url.endsWith('/query'));
    expect(JSON.parse(options.body)).toMatchObject({ conversation_id: 'chat-1', save_history: true, query: 'Explain that verse' });
    expect(options.headers['X-CSRFToken']).toBe('session-csrf');
  });

  it('starts a separate chat without deleting history', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Open Studying patience' }));
    await screen.findByText('My previous question');
    await user.click(screen.getByRole('button', { name: '+ New chat' }));
    expect(screen.queryByText('My previous question')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open Studying patience' })).toBeInTheDocument();
    await user.type(screen.getByRole('textbox', { name: 'Your question' }), 'A new question');
    await user.click(screen.getByRole('button', { name: 'Ask' }));
    const [, options] = fetchMock.mock.calls.find(([url]) => url.endsWith('/query'));
    expect(JSON.parse(options.body).conversation_id).toBeNull();
  });

  it('renames and deletes a conversation', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Rename Studying patience' }));
    const title = screen.getByRole('textbox', { name: 'Chat title' });
    await user.clear(title);
    await user.type(title, 'Evening study');
    await user.click(screen.getByRole('button', { name: 'Save title' }));
    await user.click(await screen.findByRole('button', { name: 'Delete Evening study' }));
    await waitFor(() => expect(screen.queryByRole('button', { name: 'Open Evening study' })).not.toBeInTheDocument());
    expect(window.confirm).toHaveBeenCalled();
  });

  it('adds, edits, disables and clears memories while preserving chat history', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Memory' }));
    const input = await screen.findByRole('textbox', { name: 'Add a memory' });
    await user.type(input, 'Use simple explanations.');
    await user.click(screen.getByRole('button', { name: 'Add memory' }));
    expect(await screen.findByText('Use simple explanations.')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Edit memory' }));
    const edit = screen.getByRole('textbox', { name: 'Edit memory' });
    await user.clear(edit);
    await user.type(edit, 'Use short paragraphs.');
    await user.click(screen.getByRole('button', { name: 'Save memory' }));
    expect(await screen.findByText('Use short paragraphs.')).toBeInTheDocument();
    await user.click(screen.getByRole('checkbox', { name: 'Use saved memories in answers' }));
    await waitFor(() => expect(screen.getByRole('checkbox')).not.toBeChecked());
    await user.click(screen.getByRole('button', { name: 'Clear all memories' }));
    await waitFor(() => expect(screen.queryByText('Use short paragraphs.')).not.toBeInTheDocument());
    await user.click(screen.getByRole('button', { name: 'Close memory' }));
    expect(screen.getByRole('button', { name: 'Open Studying patience' })).toBeInTheDocument();
  });

  it('clears private messages on sign out', async () => {
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Open Studying patience' }));
    await screen.findByText('My previous question');
    await user.click(screen.getByRole('button', { name: 'Sign out' }));
    expect(await screen.findByText('Guest chat · not saved')).toBeInTheDocument();
    expect(screen.queryByText('My previous question')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open Studying patience' })).not.toBeInTheDocument();
  });

  it('does not publish a late reply after the account changes in another tab', async () => {
    let finish;
    queryImpl = () => new Promise((resolve) => { finish = resolve; });
    const user = userEvent.setup();
    render(<App />);
    await screen.findByRole('button', { name: 'Sign out' });
    await user.type(screen.getByRole('textbox', { name: 'Your question' }), 'Private question');
    await user.click(screen.getByRole('button', { name: 'Ask' }));
    sessionUser = null;
    await act(async () => { window.dispatchEvent(new Event('focus')); });
    await screen.findByText('Guest chat · not saved');
    await act(async () => { finish(await json({ ...answer, conversation: savedChat })); });
    expect(screen.queryByText('Private question')).not.toBeInTheDocument();
    expect(screen.queryByText('Patience and prayer.')).not.toBeInTheDocument();
  });

  it('shows a failed sign-in without hiding the form', async () => {
    sessionUser = null;
    const original = fetchMock.getMockImplementation();
    fetchMock.mockImplementation((url, options) => url.endsWith('/auth/login') ? json({ detail: 'Email or password is incorrect.' }, 400) : original(url, options));
    const user = userEvent.setup();
    render(<App />);
    await user.click(await screen.findByRole('button', { name: 'Sign in' }));
    const dialog = screen.getByRole('dialog');
    await user.type(within(dialog).getByLabelText('Email'), 'amina@example.com');
    await user.type(within(dialog).getByLabelText('Password'), 'wrong');
    await user.click(within(dialog).getByRole('button', { name: 'Sign in' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Email or password is incorrect.');
    expect(within(dialog).getByRole('button', { name: 'Sign in' })).not.toBeDisabled();
  });
});
