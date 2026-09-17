import { useEffect, useRef, useState } from "react";
import { apiRequest } from "./api";

export default function MemoryDialog({ user, onAccount, onClose }) {
  const dialog = useRef(null);
  const [memories, setMemories] = useState([]);
  const [content, setContent] = useState("");
  const [editing, setEditing] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [loadFailed, setLoadFailed] = useState(false);
  const [reload, setReload] = useState(0);

  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    return () => element.close();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    apiRequest("/memories", { signal: controller.signal })
      .then((data) => { setMemories(data); setLoadFailed(false); setError(""); })
      .catch((err) => { if (!controller.signal.aborted) { setError(err.message); setLoadFailed(true); } })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [reload]);

  async function mutate(action) {
    setBusy(true);
    setError("");
    try { await action(); } catch (err) { setError(err.message); } finally { setBusy(false); }
  }

  function save(event) {
    event.preventDefault();
    if (!content.trim()) return;
    mutate(async () => {
      const memory = await apiRequest(editing ? `/memories/${editing}` : "/memories", {
        method: editing ? "PATCH" : "POST", body: { content: content.trim() },
      });
      setMemories((previous) => editing ? previous.map((item) => item.id === editing ? memory : item) : [...previous, memory]);
      setEditing(null);
      setContent("");
    });
  }

  function remove(memory) {
    if (!window.confirm("Delete this memory? Future answers will no longer use it.")) return;
    mutate(async () => {
      await apiRequest(`/memories/${memory.id}`, { method: "DELETE" });
      setMemories((previous) => previous.filter((item) => item.id !== memory.id));
      if (editing === memory.id) { setEditing(null); setContent(""); }
    });
  }

  return (
    <dialog ref={dialog} className="account-dialog" aria-labelledby="memory-title" onCancel={(event) => { event.preventDefault(); if (!busy) onClose(); }}>
      <div className="flex justify-between gap-4 mb-4">
        <div><h2 id="memory-title" className="text-xl font-bold text-islam-800">Saved memory</h2>
          <p className="text-sm text-dune-700 mt-1">Choose what DivinityAI remembers across your chats.</p></div>
        <button type="button" className="account-button" aria-label="Close memory" disabled={busy} onClick={onClose}>×</button>
      </div>
      <label className="flex items-center gap-3 border border-dune-300 bg-dune-100 p-3 text-sm text-islam-800">
        <input type="checkbox" checked={user.memory_enabled} disabled={busy} onChange={(event) => {
          const enabled = event.target.checked;
          mutate(async () => { const data = await apiRequest("/auth/profile", { method: "PATCH", body: { memory_enabled: enabled } }); onAccount(data.user); });
        }} />
        Use saved memories in answers
      </label>
      <p className="text-xs text-dune-700 mt-3">Only notes you add here are saved as memories. Turning memory off keeps your notes and chat history. Memories guide explanations; Quran and Hadith remain the sources.</p>
      {error && <p role="alert" className="account-error mt-3">{error}</p>}
      {loadFailed && <button type="button" className="account-button mt-2" onClick={() => { setLoading(true); setReload((value) => value + 1); }}>Retry loading memories</button>}
      {loading ? <p role="status" className="my-5 text-sm">Loading memories…</p> : !loadFailed && <>
        <div className="space-y-2 my-4 max-h-64 overflow-y-auto">
          {memories.length === 0 && <p className="text-sm text-dune-700 py-3">No memories yet. Add a preference, such as “I prefer simple explanations.”</p>}
          {memories.map((memory) => <div key={memory.id} className="border border-dune-300 p-3 bg-white/50">
            <p className="text-sm whitespace-pre-wrap break-words" dir="auto">{memory.content}</p>
            <div className="flex gap-3 mt-2 text-xs">
              <button type="button" className="text-islam-700 underline" disabled={busy} onClick={() => { setEditing(memory.id); setContent(memory.content); }}>Edit memory</button>
              <button type="button" className="text-red-800 underline" disabled={busy} onClick={() => remove(memory)}>Delete memory</button>
            </div>
          </div>)}
        </div>
        <form onSubmit={save} className="space-y-2">
          <label className="account-label">{editing ? "Edit memory" : "Add a memory"}
            <textarea className="account-input resize-y min-h-20" value={content} maxLength={500} required disabled={busy} onChange={(event) => setContent(event.target.value)} placeholder="For example: I’m new to studying the Quran." />
          </label>
          <div className="flex justify-between items-center gap-2">
            <span className="text-xs text-dune-700">{memories.length}/20 memories · {content.length}/500 characters</span>
            <div className="flex gap-2">
              {editing && <button type="button" className="account-button" disabled={busy} onClick={() => { setEditing(null); setContent(""); }}>Cancel edit</button>}
              <button type="submit" className="account-button account-primary" disabled={busy || !content.trim() || (!editing && memories.length >= 20)}>{editing ? "Save memory" : "Add memory"}</button>
            </div>
          </div>
        </form>
        {memories.length > 0 && <button type="button" className="text-sm text-red-800 underline mt-5" disabled={busy} onClick={() => {
          if (!window.confirm("Delete all saved memories? This cannot be undone. Your chats will stay in history.")) return;
          mutate(async () => { await apiRequest("/memories", { method: "DELETE" }); setMemories([]); setEditing(null); setContent(""); });
        }}>Clear all memories</button>}
      </>}
    </dialog>
  );
}
