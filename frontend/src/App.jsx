import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";
const APP_TITLE = import.meta.env.VITE_APP_TITLE || "DivinityAI";
const APP_SUBTITLE = import.meta.env.VITE_APP_SUBTITLE || "Quran & Hadith QA";

/* ------------------------------------------------------------------ */
/* SVG decorative elements                                            */
/* ------------------------------------------------------------------ */

function Star() {
  return (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" className="text-gold-500">
      <path d="M12 2l2.4 7.2h7.6l-6 4.8 2.4 7.2-6-4.8-6 4.8 2.4-7.2-6-4.8h7.6z" />
    </svg>
  );
}

function Crescent() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" className="text-gold-500">
      <path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10c1.5 0 3-.3 4.3-.9-2.5-1.5-4.3-4.3-4.3-7.6s1.8-6.1 4.3-7.6C15 2.3 13.5 2 12 2z" />
    </svg>
  );
}

function Bismillah() {
  return (
    <span className="arabic text-2xl text-gold-500 select-none">
      ﷽
    </span>
  );
}

/* ------------------------------------------------------------------ */
/* Citation helper                                                     */
/* ------------------------------------------------------------------ */

function formatMessage(text) {
  const parts = text.split(/(\[Q\s+\d+:\d+\]|\[C\s+[^\]]+\])/g);
  return parts.map((part, i) => {
    if (part.match(/^\[(Q|C)\s/)) {
      return (
        <span
          key={i}
          className="inline-block bg-gold-100 text-gold-800 px-1.5 py-0.5 rounded text-xs font-semibold mx-0.5 border border-gold-300"
        >
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [language, setLanguage] = useState("en");
  const chatEnd = useRef(null);

  const scrollToBottom = useCallback(() => {
    chatEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, scrollToBottom]);

  const sendQuery = async (e) => {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    const userMsg = { role: "user", content: query };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/v1/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, language, max_sources: 5 }),
      });
      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || data.error || "Request failed");
      }

      const botMsg = {
        role: "assistant",
        content: data.answer || "No answer received.",
        sources: data.sources || [],
        citations: data.citations || [],
        intent: data.intent || "general",
        safety: data.safety || {},
        meta: data.pipeline_meta || {},
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Sorry, something went wrong: ${err.message}`,
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-dune-100">
      {/* ============================================================ */}
      {/* TOP BANNER — Ornate 2000s religious style                     */}
      {/* ============================================================ */}
      <header className="banner-pattern text-white relative overflow-hidden">
        {/* Decorative border */}
        <div className="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-gold-500 to-transparent" />
        <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-transparent via-gold-500 to-transparent" />

        <div className="max-w-4xl mx-auto px-4 py-6 text-center relative z-10">
          {/* Top row: stars */}
          <div className="flex items-center justify-center gap-4 mb-2">
            <Star />
            <Crescent />
            <span className="text-gold-300 text-sm tracking-[0.3em] font-semibold">
              بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ
            </span>
            <Crescent />
            <Star />
          </div>

          {/* Title */}
          <h1 className="text-4xl md:text-5xl font-bold tracking-wide mb-1 font-[Georgia] drop-shadow-lg">
            {APP_TITLE}
          </h1>
          <p className="text-gold-300 text-sm tracking-[0.15em] font-[Georgia]">
            {APP_SUBTITLE}
          </p>

          {/* Bottom row: Bismillah */}
          <div className="mt-3">
            <Bismillah />
          </div>
        </div>

        {/* Decorative side banners */}
        <div className="absolute left-0 top-0 bottom-0 w-2 bg-gradient-to-b from-gold-500/20 via-gold-500/40 to-gold-500/20" />
        <div className="absolute right-0 top-0 bottom-0 w-2 bg-gradient-to-b from-gold-500/20 via-gold-500/40 to-gold-500/20" />
      </header>

      {/* ============================================================ */}
      {/* SUB-BANNER — Info bar                                         */}
      {/* ============================================================ */}
      <div className="bg-gradient-to-r from-islam-700 via-islam-600 to-islam-700 text-gold-100 px-4 py-2 text-center text-xs tracking-wide border-y border-gold-600/50">
        <div className="max-w-4xl mx-auto flex items-center justify-center gap-6 flex-wrap">
          <span className="flex items-center gap-1">
            <Star /> Sources: Quran &amp; Authentic Hadith
          </span>
          <span className="hidden sm:inline">•</span>
          <span>Not a fatwa-issuing system</span>
          <span className="hidden sm:inline">•</span>
          <span className="flex items-center gap-1">
            AR / EN / ID <Crescent />
          </span>
        </div>
      </div>

      {/* ============================================================ */}
      {/* CHAT AREA                                                     */}
      {/* ============================================================ */}
      <main className="flex-1 max-w-4xl mx-auto w-full px-4 py-4 flex flex-col">
        {/* Chat container */}
        <div className="flex-1 geometric-border bg-[#fdf8f0] rounded-none overflow-hidden flex flex-col">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 md:p-6 space-y-4 chat-scroll min-h-[400px] max-h-[60vh]">
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-full text-center text-dune-500 py-12">
                <div className="mb-4">
                  <Bismillah />
                </div>
                <div className="star-divider mb-4 w-48">
                  <Star />
                </div>
                <p className="text-lg font-[Georgia] text-dune-700 mb-2">
                  Welcome to {APP_TITLE}
                </p>
                <p className="text-sm max-w-md">
                  Ask any question about the Quran or Hadith.
                  Every answer is grounded in authentic sources
                  with verifiable citations.
                </p>
                <div className="mt-6 flex flex-wrap gap-2 justify-center">
                  {[
                    "What does the Quran say about patience?",
                    "Hadith about seeking knowledge",
                    "What is the ruling on zakat?",
                  ].map((q) => (
                    <button
                      key={q}
                      type="button"
                      className="text-xs px-3 py-1.5 bg-dune-200 text-dune-700 border border-dune-300 hover:bg-dune-300 transition-colors"
                      onClick={() => setInput(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div
                  className={`max-w-[85%] md:max-w-[75%] px-4 py-3 ${
                    msg.role === "user"
                      ? "bg-islam-600 text-white rounded-bl-lg rounded-tl-lg"
                      : msg.error
                        ? "bg-red-50 text-red-800 border border-red-300"
                        : "bg-dune-50 border border-dune-300 text-dune-900"
                  }`}
                >
                  {/* Role label */}
                  <div className="text-xs font-semibold mb-1 opacity-70 tracking-wide">
                    {msg.role === "user" ? "You" : APP_TITLE}
                  </div>

                  {/* Content */}
                  <div className="text-sm leading-relaxed whitespace-pre-wrap">
                    {msg.role === "assistant" && !msg.error
                      ? formatMessage(msg.content)
                      : msg.content}
                  </div>

                  {/* Sources */}
                  {msg.sources?.length > 0 && (
                    <details className="mt-3 text-xs">
                      <summary className="cursor-pointer text-dune-500 font-semibold tracking-wide hover:text-dune-700">
                        Sources ({msg.sources.length})
                      </summary>
                      <div className="mt-2 space-y-2 max-h-48 overflow-y-auto">
                        {msg.sources.map((s, j) => (
                          <div
                            key={j}
                            className="bg-dune-100 p-2 border border-dune-200"
                          >
                            <div className="text-gold-700 font-semibold">
                              {s.source_tag}{" "}
                              <span className="text-dune-400">
                                ({s.verification_status})
                              </span>
                            </div>
                            {s.text_en && (
                              <div className="text-dune-600 mt-1">
                                {s.text_en.slice(0, 200)}
                                {s.text_en.length > 200 && "…"}
                              </div>
                            )}
                            {s.text_ar && (
                              <div className="arabic text-dune-700 mt-1 text-base">
                                {s.text_ar.slice(0, 200)}
                                {s.text_ar.length > 200 && "…"}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </details>
                  )}

                  {/* Safety disclaimer */}
                  {msg.safety?.disclaimer && (
                    <div className="mt-2 text-xs bg-gold-50 border border-gold-300 text-gold-800 p-2 italic">
                      {msg.safety.disclaimer}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Loading indicator */}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-dune-50 border border-dune-300 px-4 py-3">
                  <div className="text-xs font-semibold mb-1 opacity-70 tracking-wide">
                    {APP_TITLE}
                  </div>
                  <div className="flex items-center gap-2 text-dune-500 text-sm">
                    <span className="animate-pulse-gold">Searching Quran & Hadith</span>
                    <span className="flex gap-1">
                      <span className="animate-pulse-gold inline-block w-1.5 h-1.5 bg-gold-500 rounded-full" style={{ animationDelay: "0s" }} />
                      <span className="animate-pulse-gold inline-block w-1.5 h-1.5 bg-gold-500 rounded-full" style={{ animationDelay: "0.3s" }} />
                      <span className="animate-pulse-gold inline-block w-1.5 h-1.5 bg-gold-500 rounded-full" style={{ animationDelay: "0.6s" }} />
                    </span>
                  </div>
                </div>
              </div>
            )}

            <div ref={chatEnd} />
          </div>

          {/* ======================================================== */}
          {/* INPUT AREA                                                */}
          {/* ======================================================== */}
          <div className="border-t-2 border-gold-500 bg-gradient-to-r from-islam-800 via-islam-700 to-islam-800 px-4 py-3">
            <form onSubmit={sendQuery} className="flex gap-2 max-w-3xl mx-auto">
              {/* Language selector */}
              <select
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                className="bg-islam-900 text-gold-200 border border-gold-600 px-2 py-2 text-xs font-semibold tracking-wide focus:outline-none focus:border-gold-400"
              >
                <option value="en">EN</option>
                <option value="ar">AR</option>
                <option value="id">ID</option>
              </select>

              {/* Input */}
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about the Quran or Hadith..."
                className="flex-1 bg-[#fdf8f0] border border-gold-500 px-4 py-2 text-sm text-dune-900 placeholder-dune-400 focus:outline-none focus:ring-2 focus:ring-gold-500 font-[Georgia]"
                disabled={loading}
                dir="auto"
              />

              {/* Submit */}
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="bg-gold-500 hover:bg-gold-600 disabled:opacity-50 disabled:cursor-not-allowed text-islam-900 px-6 py-2 text-sm font-bold tracking-wide transition-colors border border-gold-600"
              >
                Ask
              </button>
            </form>
          </div>
        </div>
      </main>

      {/* ============================================================ */}
      {/* FOOTER — Ornate                                                */}
      {/* ============================================================ */}
      <footer className="banner-pattern text-gold-200 text-center py-4 text-xs tracking-wide border-t-2 border-gold-600/50 relative">
        <div className="absolute inset-x-0 top-0 h-0.5 bg-gradient-to-r from-transparent via-gold-500 to-transparent" />
        <div className="flex items-center justify-center gap-3 mb-1">
          <Star />
          <span className="tracking-[0.2em] font-semibold">
            {APP_TITLE}
          </span>
          <Star />
        </div>
        <p className="text-gold-400/70">
          Grounded in the Quran &bull; Authenticated Hadith &bull; Verifiable Citations
        </p>
        <p className="text-gold-400/50 mt-1">
          Not a fatwa-issuing system. For definitive rulings, consult a qualified scholar.
        </p>
      </footer>
    </div>
  );
}