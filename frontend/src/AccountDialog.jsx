import { useEffect, useRef, useState } from "react";
import { apiRequest } from "./api";

export default function AccountDialog({ user, initialMode, onClose, onAccount }) {
  const dialog = useRef(null);
  const [mode, setMode] = useState(initialMode);
  const [name, setName] = useState(user?.name || "");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const title = user ? "Your account" : mode === "register" ? "Create your account" : "Welcome back";

  useEffect(() => {
    const element = dialog.current;
    element.showModal();
    return () => element.close();
  }, []);

  async function submit(event) {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      const data = await apiRequest(user ? "/auth/profile" : `/auth/${mode}`, {
        method: user ? "PATCH" : "POST",
        body: user ? { name } : { email, password, ...(mode === "register" ? { name } : {}) },
      });
      onAccount(data.user);
      onClose();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <dialog ref={dialog} className="account-dialog" aria-labelledby="account-title" onCancel={(event) => { event.preventDefault(); if (!busy) onClose(); }}>
      <div className="flex items-start justify-between gap-4 mb-5">
        <div>
          <h2 id="account-title" className="text-xl font-bold text-islam-800">{title}</h2>
          <p className="text-sm text-dune-700 mt-1">{user ? user.email : "Save your chats and make room for what matters to you."}</p>
        </div>
        <button type="button" className="account-button" aria-label="Close account" onClick={onClose} disabled={busy}>×</button>
      </div>
      <form onSubmit={submit} className="space-y-4">
        {(user || mode === "register") && <label className="account-label">Name
          <input className="account-input" value={name} onChange={(event) => setName(event.target.value)} autoComplete="name" maxLength={150} required autoFocus />
        </label>}
        {!user && <>
          <label className="account-label">Email
            <input className="account-input" type="email" value={email} onChange={(event) => setEmail(event.target.value)} autoComplete="email" maxLength={150} required autoFocus={mode === "login"} />
          </label>
          <label className="account-label">Password
            <input className="account-input" type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete={mode === "register" ? "new-password" : "current-password"} minLength={mode === "register" ? 8 : undefined} maxLength={128} required aria-describedby={mode === "register" ? "password-help" : undefined} />
          </label>
          {mode === "register" && <p id="password-help" className="text-xs text-dune-700">Use at least 8 characters. Avoid common passwords or your personal details.</p>}
        </>}
        {error && <p role="alert" className="account-error">{error}</p>}
        <button className="account-button account-primary w-full" disabled={busy} type="submit">{busy ? "Saving…" : user ? "Save account" : mode === "register" ? "Create account" : "Sign in"}</button>
      </form>
      {!user && <button type="button" className="mt-4 text-sm text-islam-700 underline underline-offset-4" disabled={busy} onClick={() => { setMode(mode === "register" ? "login" : "register"); setError(""); setPassword(""); }}>
        {mode === "register" ? "Already have an account? Sign in" : "New here? Create an account"}
      </button>}
    </dialog>
  );
}
