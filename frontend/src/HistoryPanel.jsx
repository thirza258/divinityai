import { useEffect, useState } from "react";
import { apiRequest } from "./api";

export default function HistoryPanel({ activeId, revision, disabled, onOpen, onRename, onDelete, onClear }) {
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [items, setItems] = useState([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [version, setVersion] = useState(0);
  const [editing, setEditing] = useState(null);
  const [title, setTitle] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      setError("");
      try {
        const data = await apiRequest(`/conversations?search=${encodeURIComponent(search)}&page=${page}`, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setItems((previous) => page === 1 ? data.results : [...previous.filter((item) => !data.results.some((next) => next.id === item.id)), ...data.results]);
        setHasMore(Boolean(data.next));
      } catch (err) { if (!controller.signal.aborted) setError(err.message); }
      finally { if (!controller.signal.aborted) setLoading(false); }
    }, search ? 250 : 0);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [search, page, revision, version]);

  function refresh() { setPage(1); setVersion((value) => value + 1); }

  async function mutate(action) {
    setBusy(true);
    setError("");
    try { await action(); refresh(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  const locked = disabled || busy;
  return (
    <section aria-labelledby="history-title" className="space-y-3">
      <h2 id="history-title" className="text-lg font-bold text-islam-800">Chat history</h2>
      <label className="sr-only" htmlFor="history-search">Search chat history</label>
      <input id="history-search" type="search" className="account-input" placeholder="Search your chats…" maxLength={200} value={search} disabled={locked} onChange={(event) => { setSearch(event.target.value); setPage(1); }} />
      {error && <div role="alert" className="account-error">{error} <button type="button" className="underline" onClick={refresh}>Retry</button></div>}
      <div className="space-y-2 max-h-[55vh] overflow-y-auto chat-scroll" aria-busy={loading}>
        {items.map((item) => <div key={item.id} className={`border p-3 ${item.id === activeId ? "border-gold-600 bg-gold-50" : "border-dune-300 bg-dune-50"}`}>
          {editing === item.id ? <form onSubmit={(event) => {
            event.preventDefault();
            mutate(async () => { const updated = await apiRequest(`/conversations/${item.id}`, { method: "PATCH", body: { title } }); onRename(updated); setEditing(null); });
          }} className="space-y-2">
            <label className="account-label">Chat title<input className="account-input" value={title} onChange={(event) => setTitle(event.target.value)} maxLength={120} required autoFocus /></label>
            <div className="flex gap-2"><button type="submit" className="account-button" disabled={locked || !title.trim()}>Save title</button><button type="button" className="account-button" disabled={busy} onClick={() => setEditing(null)}>Cancel</button></div>
          </form> : <>
            <button type="button" className="text-left w-full disabled:opacity-60" disabled={locked} onClick={() => onOpen(item.id)} aria-label={`Open ${item.title}`} aria-current={item.id === activeId ? "true" : undefined}>
              <span className="block text-sm font-semibold text-islam-800 break-words line-clamp-2">{item.title}</span>
              <span className="text-xs text-dune-700">{new Date(item.updated_at).toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}</span>
            </button>
            <div className="flex gap-3 mt-2 text-xs">
              <button type="button" className="text-dune-700 underline" aria-label={`Rename ${item.title}`} disabled={locked} onClick={() => { setEditing(item.id); setTitle(item.title); }}>Rename</button>
              <button type="button" className="text-red-800 underline" aria-label={`Delete ${item.title}`} disabled={locked} onClick={() => {
                if (!window.confirm(`Delete “${item.title}”? This cannot be undone.`)) return;
                mutate(async () => { await apiRequest(`/conversations/${item.id}`, { method: "DELETE" }); onDelete(item.id); });
              }}>Delete</button>
            </div>
          </>}
        </div>)}
        {loading && <p role="status" className="text-sm text-dune-700">Loading chats…</p>}
        {!loading && !error && items.length === 0 && <p className="text-sm text-dune-700 py-3">{search ? "No chats match your search." : "Your conversations will appear here after your first question."}</p>}
      </div>
      {hasMore && <button className="account-button w-full" type="button" disabled={locked || loading} onClick={() => setPage((value) => value + 1)}>Load more chats</button>}
      {items.length > 0 && <button type="button" className="text-xs text-red-800 underline" disabled={locked} onClick={() => {
        if (!window.confirm("Delete all chat history? This cannot be undone. Your saved memories will stay.")) return;
        mutate(async () => { await apiRequest("/conversations", { method: "DELETE" }); setItems([]); onClear(); });
      }}>Clear chat history</button>}
    </section>
  );
}
