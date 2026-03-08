import { useState, useEffect, useRef, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend, LineChart, Line, ScatterChart, Scatter, ZAxis,
} from "recharts";

// ─── Config ────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

const MODEL_META = {
  "text-embedding-3-small": { color: "#22d3ee", short: "OpenAI", icon: "⬡" },
  "all-MiniLM-L6-v2":       { color: "#a78bfa", short: "MiniLM", icon: "◈" },
  "multilingual-e5-base":   { color: "#34d399", short: "E5-base", icon: "◉" },
};

const DEMO_QUESTIONS = [
  "Who is Gandalf?",
  "What is the One Ring?",
  "Tell me about the Battle of Helm's Deep",
  "Who are the Nazgûl?",
  "What happened at Mount Doom?",
  "Who is Galadriel?",
];

// ─── Embedded evaluation data (fallback) ──────────────────────────────────
const EVAL_FALLBACK = null; // will be loaded from API

// ─── Styles (CSS-in-JS via a style tag) ────────────────────────────────────
const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:       #080c14;
    --bg2:      #0d1220;
    --bg3:      #111827;
    --border:   #1e2d45;
    --border2:  #253550;
    --text:     #e2e8f0;
    --muted:    #64748b;
    --cyan:     #22d3ee;
    --violet:   #a78bfa;
    --emerald:  #34d399;
    --amber:    #fbbf24;
    --red:      #f87171;
    --glow-c:   0 0 20px rgba(34,211,238,.25);
    --glow-v:   0 0 20px rgba(167,139,250,.25);
    --glow-e:   0 0 20px rgba(52,211,153,.25);
  }

  html, body, #root { height: 100%; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    overflow-x: hidden;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  /* Noise overlay */
  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: .4;
  }

  .app { display: flex; height: 100vh; position: relative; z-index: 1; }

  /* ── Sidebar ── */
  .sidebar {
    width: 220px; flex-shrink: 0;
    background: var(--bg2);
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column;
    padding: 24px 0;
  }
  .sidebar-brand {
    padding: 0 20px 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .sidebar-brand h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; font-weight: 700;
    letter-spacing: 3px; text-transform: uppercase;
    color: var(--cyan); line-height: 1.4;
  }
  .sidebar-brand p {
    font-size: 10px; color: var(--muted); margin-top: 4px;
    font-family: 'JetBrains Mono', monospace;
  }

  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 20px; cursor: pointer;
    font-size: 12px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: var(--muted);
    border-left: 2px solid transparent;
    transition: all .2s;
  }
  .nav-item:hover { color: var(--text); background: rgba(255,255,255,.03); }
  .nav-item.active { color: var(--cyan); border-left-color: var(--cyan); background: rgba(34,211,238,.06); }
  .nav-item .icon { font-size: 16px; width: 20px; text-align: center; }

  .sidebar-footer {
    margin-top: auto; padding: 16px 20px;
    border-top: 1px solid var(--border);
    font-size: 10px; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.8;
  }
  .status-dot {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: var(--emerald);
    margin-right: 6px; animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }

  /* ── Main ── */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }

  .topbar {
    height: 52px; flex-shrink: 0;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center;
    padding: 0 28px; gap: 16px;
    background: rgba(13,18,32,.8); backdrop-filter: blur(12px);
  }
  .topbar-title { font-size: 14px; font-weight: 700; color: var(--text); }
  .topbar-sub { font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }
  .topbar-sep { flex: 1; }

  .model-badge {
    display: flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
  }

  .content { flex: 1; overflow-y: auto; padding: 28px; }

  /* ── Cards ── */
  .card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
  }
  .card-title {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: var(--muted);
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
  }
  .card-title::before {
    content: ''; display: block;
    width: 3px; height: 14px; border-radius: 2px;
    background: var(--cyan);
  }

  /* ── Grid helpers ── */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
  .stack { display: flex; flex-direction: column; gap: 16px; }

  /* ── Compare view ── */
  .compare-input-row {
    display: flex; gap: 12px; margin-bottom: 24px;
  }
  .input-wrap { flex: 1; position: relative; }
  .input-wrap input {
    width: 100%; padding: 12px 16px;
    background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 8px; color: var(--text);
    font-family: 'Syne', sans-serif; font-size: 14px;
    outline: none; transition: border-color .2s;
  }
  .input-wrap input:focus { border-color: var(--cyan); box-shadow: var(--glow-c); }
  .input-wrap input::placeholder { color: var(--muted); }

  .btn {
    padding: 12px 20px; border-radius: 8px; border: none; cursor: pointer;
    font-family: 'Syne', sans-serif; font-size: 13px; font-weight: 700;
    letter-spacing: .5px; transition: all .2s;
    display: flex; align-items: center; gap: 8px;
  }
  .btn-primary {
    background: var(--cyan); color: #000;
  }
  .btn-primary:hover { filter: brightness(1.15); transform: translateY(-1px); }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; transform: none; }

  .btn-ghost {
    background: var(--bg3); color: var(--text);
    border: 1px solid var(--border2);
  }
  .btn-ghost:hover { border-color: var(--cyan); color: var(--cyan); }

  .compare-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }
  @media (max-width: 1100px) { .compare-grid { grid-template-columns: 1fr; } }

  .answer-card {
    background: var(--bg3); border-radius: 12px;
    padding: 20px; border: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 14px;
    transition: border-color .3s;
  }
  .answer-card.loaded { animation: fadeUp .4s ease; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:none} }

  .model-header {
    display: flex; align-items: center; justify-content: space-between;
  }
  .model-name {
    font-size: 12px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
  }
  .model-type-tag {
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    padding: 2px 7px; border-radius: 10px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
  }

  .answer-text {
    font-size: 13px; line-height: 1.7; color: var(--text);
    min-height: 80px;
  }
  .answer-skeleton {
    display: flex; flex-direction: column; gap: 8px;
  }
  .skel {
    height: 12px; border-radius: 4px; background: var(--border);
    animation: shimmer 1.5s infinite;
  }
  @keyframes shimmer {
    0%,100% { opacity: .4 } 50% { opacity: .8 }
  }

  .meta-row {
    display: flex; gap: 12px; flex-wrap: wrap;
  }
  .meta-chip {
    display: flex; align-items: center; gap: 5px;
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: var(--muted); background: var(--bg2);
    padding: 3px 8px; border-radius: 6px;
    border: 1px solid var(--border);
  }
  .meta-chip .val { color: var(--text); font-weight: 700; }

  .sources-list {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: var(--muted); display: flex; flex-direction: column; gap: 3px;
  }
  .source-item::before { content: '↳ '; color: var(--cyan); }

  .demo-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 24px; }
  .demo-pill {
    padding: 6px 12px; border-radius: 20px;
    background: var(--bg3); border: 1px solid var(--border2);
    font-size: 11px; cursor: pointer; color: var(--muted);
    transition: all .2s; font-family: 'JetBrains Mono', monospace;
  }
  .demo-pill:hover { border-color: var(--cyan); color: var(--cyan); }

  /* ── Benchmark ── */
  .metric-card {
    background: var(--bg3); border-radius: 10px;
    padding: 16px 20px; border: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 6px;
  }
  .metric-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-value { font-size: 28px; font-weight: 800; line-height: 1; }
  .metric-sub { font-size: 10px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }

  .winner-tag {
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 9px; font-weight: 700; letter-spacing: 1px;
    padding: 2px 8px; border-radius: 10px; text-transform: uppercase;
    background: rgba(251,191,36,.1); color: var(--amber); border: 1px solid rgba(251,191,36,.3);
    margin-left: 8px;
  }

  .table-wrap { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    text-align: left; padding: 10px 14px;
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 10px 14px; border-bottom: 1px solid rgba(30,45,69,.5);
    font-family: 'JetBrains Mono', monospace;
  }
  tr:hover td { background: rgba(255,255,255,.02); }

  .progress-bar { height: 4px; border-radius: 2px; background: var(--border); overflow: hidden; margin-top: 3px; }
  .progress-fill { height: 100%; border-radius: 2px; transition: width .8s ease; }

  /* ── Tabs ── */
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; }
  .tab {
    padding: 7px 14px; border-radius: 6px; font-size: 11px;
    font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    cursor: pointer; border: 1px solid transparent;
    font-family: 'JetBrains Mono', monospace; color: var(--muted);
    transition: all .2s;
  }
  .tab.active { border-color: var(--cyan); color: var(--cyan); background: rgba(34,211,238,.08); }
  .tab:hover:not(.active) { color: var(--text); border-color: var(--border2); }

  /* ── Difficulty bars ── */
  .diff-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
  .diff-label { font-size: 11px; width: 60px; color: var(--muted); font-family: 'JetBrains Mono', monospace; text-transform: uppercase; letter-spacing: 1px; }

  /* ── Presentation ── */
  .pres-slide {
    min-height: calc(100vh - 140px);
    display: flex; flex-direction: column; justify-content: center;
    align-items: center; text-align: center;
    padding: 40px;
  }
  .pres-step {
    font-size: 11px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: var(--cyan); margin-bottom: 20px;
    font-family: 'JetBrains Mono', monospace;
  }
  .pres-h1 { font-size: 48px; font-weight: 800; line-height: 1.1; margin-bottom: 16px; }
  .pres-h2 { font-size: 32px; font-weight: 700; line-height: 1.2; margin-bottom: 16px; }
  .pres-body { font-size: 16px; color: var(--muted); line-height: 1.8; max-width: 600px; }
  .pres-highlight { color: var(--cyan); }
  .pres-nav {
    display: flex; align-items: center; justify-content: center; gap: 16px;
    margin-top: 40px;
  }
  .pres-dot {
    width: 8px; height: 8px; border-radius: 50%; cursor: pointer;
    background: var(--border2); transition: all .2s;
  }
  .pres-dot.active { background: var(--cyan); transform: scale(1.3); }
  .pres-big-stat {
    font-size: 80px; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, var(--cyan), var(--emerald));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 16px 0;
  }
  .pres-cards-row {
    display: grid; grid-template-columns: repeat(3,1fr); gap: 20px;
    width: 100%; max-width: 900px; margin-top: 24px; text-align: left;
  }
  .pres-card {
    background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 12px; padding: 20px;
  }
  .pres-card-icon { font-size: 28px; margin-bottom: 10px; }
  .pres-card-title { font-size: 13px; font-weight: 700; margin-bottom: 6px; }
  .pres-card-body { font-size: 12px; color: var(--muted); line-height: 1.6; }

  .glow-text-cyan { text-shadow: 0 0 30px rgba(34,211,238,.5); }
  .glow-text-emerald { text-shadow: 0 0 30px rgba(52,211,153,.5); }

  /* ─ Loader ── */
  .spinner {
    width: 18px; height: 18px; border: 2px solid var(--border2);
    border-top-color: var(--cyan); border-radius: 50%;
    animation: spin .7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── tooltip ── */
  .recharts-tooltip-wrapper .recharts-default-tooltip {
    background: var(--bg2) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
  }
`;

// ─── Utility ───────────────────────────────────────────────────────────────
const pct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
const ms  = (v) => v != null ? `${v.toFixed(0)}ms` : "—";
const modelColor = (m) => MODEL_META[m]?.color ?? "#64748b";
const modelShort = (m) => MODEL_META[m]?.short ?? m;

// ─── API helpers ───────────────────────────────────────────────────────────
async function apiFetch(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

// ─── Subcomponents ─────────────────────────────────────────────────────────

function ModelBadge({ model }) {
  const meta = MODEL_META[model] ?? {};
  const color = meta.color ?? "#64748b";
  return (
    <div
      className="model-badge"
      style={{ background: color + "18", border: `1px solid ${color}40`, color }}
    >
      <span>{meta.icon ?? "◆"}</span>
      <span>{meta.short ?? model}</span>
    </div>
  );
}

function AnswerCard({ model, data, loading }) {
  const color = modelColor(model);
  const meta  = MODEL_META[model] ?? {};
  const info  = { "text-embedding-3-small": "OpenAI", "all-MiniLM-L6-v2": "HuggingFace", "multilingual-e5-base": "HuggingFace" };

  return (
    <div className={`answer-card ${data ? "loaded" : ""}`}
      style={{ borderColor: data ? color + "50" : undefined }}>
      <div className="model-header">
        <div className="model-name" style={{ color }}>{meta.icon} {meta.short ?? model}</div>
        <span className="model-type-tag"
          style={{ background: color + "15", color, border: `1px solid ${color}30` }}>
          {info[model] ?? "Model"}
        </span>
      </div>

      {loading ? (
        <div className="answer-skeleton">
          <div className="skel" style={{ width: "90%" }} />
          <div className="skel" style={{ width: "75%" }} />
          <div className="skel" style={{ width: "82%" }} />
          <div className="skel" style={{ width: "60%" }} />
        </div>
      ) : data ? (
        <>
          <p className="answer-text">{data.answer}</p>
          <div className="meta-row">
            <span className="meta-chip">
              ⚡ <span className="val">{ms(data.retrieval_time_ms)}</span>
            </span>
            {data.top_score != null && (
              <span className="meta-chip">
                🎯 <span className="val">{(data.top_score * 100).toFixed(0)}%</span>
              </span>
            )}
          </div>
          {data.sources?.length > 0 && (
            <div className="sources-list">
              {data.sources.slice(0, 3).map((s, i) => (
                <div key={i} className="source-item">
                  {s.split("/").pop()}
                </div>
              ))}
            </div>
          )}
        </>
      ) : (
        <p className="answer-text" style={{ color: "var(--muted)" }}>
          Waiting for query…
        </p>
      )}
    </div>
  );
}

// ─── View: Compare ─────────────────────────────────────────────────────────
function CompareView({ indexedModels }) {
  const [question, setQuestion] = useState("");
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [history, setHistory]   = useState([]); // [{question, results, timestamp}]
  const [loadingQuestion, setLoadingQuestion] = useState(null);
  const bottomRef = useRef(null);

  const displayModels = indexedModels.length > 0
    ? indexedModels
    : Object.keys(MODEL_META);

  const submit = useCallback(async (q) => {
    const text = (q || question).trim();
    if (!text) return;
    setLoading(true);
    setLoadingQuestion(text);
    setError(null);
    setQuestion("");
    try {
      const data = await apiFetch("/compare", {
        method: "POST",
        body: JSON.stringify({ question: text, k: 4, threshold: 0.35 }),
      });
      const map = {};
      for (const r of data.results) map[r.model] = r;
      setHistory((prev) => [...prev, {
        question: text,
        results: map,
        timestamp: new Date().toLocaleTimeString("sv-SE", { hour: "2-digit", minute: "2-digit" }),
      }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
      setLoadingQuestion(null);
    }
  }, [question]);

  // No auto-scroll needed — newest entry appears at top

  const onKey = (e) => { if (e.key === "Enter") submit(); };

  return (
    <div className="stack">
      {/* Demo pills */}
      <div className="demo-pills">
        {DEMO_QUESTIONS.map((q) => (
          <div key={q} className="demo-pill"
            onClick={() => { setQuestion(q); submit(q); }}>
            {q}
          </div>
        ))}
      </div>

      {/* Input */}
      <div className="compare-input-row">
        <div className="input-wrap">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={onKey}
            placeholder="Ask about Tolkien's world…"
          />
        </div>
        <button className="btn btn-primary" onClick={() => submit()} disabled={loading}>
          {loading ? <div className="spinner" /> : "▶"}
          {loading ? "Querying…" : "Compare"}
        </button>
        {history.length > 0 && (
          <button className="btn btn-ghost" onClick={() => setHistory([])}
            title="Clear history">
            ✕ Clear
          </button>
        )}
      </div>

      {error && (
        <div style={{ color: "var(--red)", fontSize: 12, fontFamily: "'JetBrains Mono', monospace",
          background: "rgba(248,113,113,.08)", padding: "12px 16px", borderRadius: 8,
          border: "1px solid rgba(248,113,113,.2)" }}>
          ⚠ {error}
        </div>
      )}

      {/* Empty state */}
      {history.length === 0 && !loading && (
        <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--muted)" }}>
          <div style={{ fontSize: 36, marginBottom: 12 }}>⬡</div>
          <div style={{ fontSize: 12, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.8 }}>
            Ask a question to compare all three models side by side.<br />
            Your conversation history will appear here.
          </div>
        </div>
      )}

      {/* History */}
      <div className="stack">
        {[...history].reverse().map((entry, idx) => (
          <div key={idx} style={{
            borderRadius: 12, overflow: "hidden",
            border: "1px solid var(--border)",
            background: "var(--bg2)",
          }}>
            {/* Question header */}
            <div style={{
              padding: "12px 20px",
              borderBottom: "1px solid var(--border)",
              display: "flex", alignItems: "center", justifyContent: "space-between",
              background: "rgba(255,255,255,.02)",
            }}>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono',monospace",
                  color: "var(--cyan)", fontWeight: 700, letterSpacing: 1 }}>
                  Q{idx + 1}
                </span>
                <span style={{ fontSize: 13, fontWeight: 600 }}>{entry.question}</span>
              </div>
              <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono',monospace",
                color: "var(--muted)" }}>
                {entry.timestamp}
              </span>
            </div>
            {/* Answers grid */}
            <div className="compare-grid" style={{ padding: 16 }}>
              {displayModels.map((m) => (
                <AnswerCard key={m} model={m} data={entry.results[m]} loading={false} />
              ))}
            </div>
          </div>
        ))}

        {/* Loading entry */}
        {loading && loadingQuestion && (
          <div style={{
            borderRadius: 12, overflow: "hidden",
            border: "1px solid var(--border)",
            background: "var(--bg2)",
          }}>
            <div style={{
              padding: "12px 20px",
              borderBottom: "1px solid var(--border)",
              display: "flex", alignItems: "center", gap: 10,
              background: "rgba(255,255,255,.02)",
            }}>
              <div className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} />
              <span style={{ fontSize: 13, fontWeight: 600 }}>{loadingQuestion}</span>
            </div>
            <div className="compare-grid" style={{ padding: 16 }}>
              {displayModels.map((m) => (
                <AnswerCard key={m} model={m} data={null} loading={true} />
              ))}
            </div>
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ─── View: Benchmark ───────────────────────────────────────────────────────
function BenchmarkView({ evalData }) {
  const [activeTab, setActiveTab] = useState("overview");

  if (!evalData) {
    return (
      <div style={{ textAlign: "center", padding: 60, color: "var(--muted)" }}>
        <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
        <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>
          No evaluation data found.<br />Run <code style={{ color: "var(--cyan)" }}>evaluate.py</code> first.
        </div>
      </div>
    );
  }

  const models = evalData.models ?? [];

  // Radar chart data
  const radarData = [
    { metric: "Hit Rate",         ...Object.fromEntries(models.map(m => [modelShort(m.model_name), m.metrics.source_hit_rate * 100])) },
    { metric: "Keyword Recall",   ...Object.fromEntries(models.map(m => [modelShort(m.model_name), m.metrics.avg_keyword_recall * 100])) },
    { metric: "Relevance Score",  ...Object.fromEntries(models.map(m => [modelShort(m.model_name), m.metrics.avg_top_score * 100])) },
    { metric: "Speed (inv)",      ...Object.fromEntries(models.map(m => [modelShort(m.model_name), Math.max(0, 100 - m.metrics.avg_retrieval_time_ms / 20)])) },
  ];

  // Difficulty bar chart data
  const difficultyData = ["easy", "medium", "hard"].map((d) => ({
    difficulty: d,
    ...Object.fromEntries(models.map((m) => [
      modelShort(m.model_name),
      (m.by_difficulty?.[d] ?? 0) * 100,
    ])),
  }));

  // Speed comparison
  const speedData = models.map((m) => ({
    name: modelShort(m.model_name),
    time: m.metrics.avg_retrieval_time_ms,
    color: modelColor(m.model_name),
  }));

  // Per-question table
  const allQuestions = models[0]?.results?.map((r, i) => ({
    q: r.question,
    difficulty: r.difficulty,
    ...Object.fromEntries(models.map((m) => [
      m.model_name,
      { hit: m.results[i]?.source_hit, score: m.results[i]?.top_score, time: m.results[i]?.retrieval_time_ms }
    ])),
  })) ?? [];

  // Cost data
  // OpenAI text-embedding-3-small: $0.020 per 1M tokens
  // Avg query ~15 tokens, so cost per query = 15 * 0.020 / 1_000_000
  const OPENAI_COST_PER_TOKEN = 0.020 / 1_000_000;
  const AVG_TOKENS_PER_QUERY = 15;
  const OPENAI_COST_PER_QUERY = OPENAI_COST_PER_TOKEN * AVG_TOKENS_PER_QUERY;

  const costMeta = {
    "text-embedding-3-small": { costPerQuery: OPENAI_COST_PER_QUERY, type: "paid", note: "$0.020 / 1M tokens (OpenAI API)" },
    "all-MiniLM-L6-v2":       { costPerQuery: 0, type: "free", note: "Free — runs locally on CPU" },
    "multilingual-e5-base":   { costPerQuery: 0, type: "free", note: "Free — runs locally on CPU" },
  };

  // Break-even chart: cumulative cost at N queries
  const breakEvenData = [1, 100, 1000, 10000, 50000, 100000, 500000, 1000000].map((n) => ({
    queries: n >= 1000000 ? "1M" : n >= 1000 ? `${n/1000}k` : `${n}`,
    queriesRaw: n,
    OpenAI: parseFloat((n * OPENAI_COST_PER_QUERY).toFixed(4)),
    "Open-Source": 0,
  }));

  return (
    <div className="stack">
      <div className="tabs">
        {["overview", "difficulty", "speed", "cost", "questions"].map((t) => (
          <div key={t} className={`tab ${activeTab === t ? "active" : ""}`} onClick={() => setActiveTab(t)}>
            {t}
          </div>
        ))}
      </div>

      {activeTab === "overview" && (
        <div className="stack">
          {/* Metric cards */}
          <div className="grid-3">
            {models.map((m) => {
              const color = modelColor(m.model_name);
              return (
                <div key={m.model_name} className="metric-card" style={{ borderColor: color + "40" }}>
                  <div className="model-name" style={{ color, fontSize: 11, marginBottom: 12 }}>
                    {MODEL_META[m.model_name]?.icon} {modelShort(m.model_name)}
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
                    {[
                      { label: "Source Hit Rate", val: pct(m.metrics.source_hit_rate) },
                      { label: "Keyword Recall",  val: pct(m.metrics.avg_keyword_recall) },
                      { label: "Avg Top Score",   val: pct(m.metrics.avg_top_score) },
                      { label: "Avg Response",    val: ms(m.metrics.avg_retrieval_time_ms) },
                    ].map(({ label, val }) => (
                      <div key={label}>
                        <div className="metric-label">{label}</div>
                        <div className="metric-value" style={{ color, fontSize: 22 }}>{val}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Radar chart */}
          <div className="card">
            <div className="card-title">Performance Radar</div>
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--muted)", fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                {models.map((m) => (
                  <Radar key={m.model_name} name={modelShort(m.model_name)}
                    dataKey={modelShort(m.model_name)}
                    stroke={modelColor(m.model_name)}
                    fill={modelColor(m.model_name)} fillOpacity={0.1}
                    strokeWidth={2} />
                ))}
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} />
                <Tooltip contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border2)", borderRadius: 8, fontSize: 11 }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {activeTab === "difficulty" && (
        <div className="card">
          <div className="card-title">Source Hit Rate by Difficulty</div>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={difficultyData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="difficulty" tick={{ fill: "var(--muted)", fontSize: 11, fontFamily: "'JetBrains Mono',monospace", textTransform: "uppercase" }} axisLine={false} tickLine={false} />
              <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false} />
              <Tooltip formatter={(v) => `${v.toFixed(1)}%`} contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border2)", borderRadius: 8, fontSize: 11 }} />
              <Legend wrapperStyle={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} />
              {models.map((m) => (
                <Bar key={m.model_name} dataKey={modelShort(m.model_name)}
                  fill={modelColor(m.model_name)} radius={[4, 4, 0, 0]} maxBarSize={40} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {activeTab === "speed" && (
        <div className="stack">
          <div className="card">
            <div className="card-title">Average Response Time (ms)</div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={speedData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" horizontal={false} />
                <XAxis type="number" tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={(v) => `${v}ms`} />
                <YAxis dataKey="name" type="category" tick={{ fill: "var(--muted)", fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} axisLine={false} tickLine={false} width={70} />
                <Tooltip formatter={(v) => `${v.toFixed(0)}ms`} contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border2)", borderRadius: 8, fontSize: 11 }} />
                <Bar dataKey="time" radius={[0, 4, 4, 0]} maxBarSize={32}>
                  {speedData.map((entry, i) => (
                    <rect key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Speed cards */}
          <div className="grid-3">
            {[...models].sort((a, b) => a.metrics.avg_retrieval_time_ms - b.metrics.avg_retrieval_time_ms).map((m, i) => {
              const color = modelColor(m.model_name);
              return (
                <div key={m.model_name} className="metric-card" style={{ borderColor: color + "40" }}>
                  {i === 0 && <div style={{ color: "var(--amber)", fontSize: 10, fontWeight: 700, letterSpacing: 1, fontFamily: "'JetBrains Mono',monospace", marginBottom: 8 }}>🏆 FASTEST</div>}
                  <div className="model-name" style={{ color }}>{modelShort(m.model_name)}</div>
                  <div className="metric-value" style={{ color, fontSize: 32, marginTop: 8 }}>{ms(m.metrics.avg_retrieval_time_ms)}</div>
                  <div className="metric-sub">avg per query</div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {activeTab === "cost" && (
        <div className="stack">
          {/* Cost per query cards */}
          <div className="grid-3">
            {models.map((m) => {
              const color = modelColor(m.model_name);
              const meta = costMeta[m.model_name] ?? { costPerQuery: 0, type: "free", note: "" };
              const per1k = (meta.costPerQuery * 1000).toFixed(4);
              const per1M = (meta.costPerQuery * 1_000_000).toFixed(2);
              return (
                <div key={m.model_name} className="metric-card" style={{ borderColor: color + "40" }}>
                  <div className="model-name" style={{ color, marginBottom: 12 }}>
                    {MODEL_META[m.model_name]?.icon} {modelShort(m.model_name)}
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div className="metric-label">Cost per query</div>
                    <div className="metric-value" style={{ color, fontSize: 28 }}>
                      {meta.type === "free" ? "$0.00" : `$${meta.costPerQuery.toFixed(6)}`}
                    </div>
                  </div>
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <div style={{ fontSize: 10, fontFamily: "'JetBrains Mono',monospace", color: "var(--muted)" }}>
                      Per 1,000 queries: <span style={{ color }}>{meta.type === "free" ? "$0.00" : `$${per1k}`}</span>
                    </div>
                    <div style={{ fontSize: 10, fontFamily: "'JetBrains Mono',monospace", color: "var(--muted)" }}>
                      Per 1,000,000 queries: <span style={{ color }}>{meta.type === "free" ? "$0.00" : `$${per1M}`}</span>
                    </div>
                    <div style={{ marginTop: 8, fontSize: 10, fontFamily: "'JetBrains Mono',monospace",
                      padding: "4px 8px", borderRadius: 6, border: `1px solid ${color}30`,
                      background: color + "10", color }}>
                      {meta.note}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Break-even chart */}
          <div className="card">
            <div className="card-title">Cumulative API Cost — OpenAI vs Open-Source</div>
            <p style={{ fontSize: 11, color: "var(--muted)", marginBottom: 20, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.6 }}>
              Open-source models run locally at zero API cost. The more queries you run, the larger the saving.
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={breakEvenData} margin={{ top: 10, right: 30, bottom: 10, left: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="queries"
                  tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }}
                  axisLine={false} tickLine={false} />
                <YAxis tickFormatter={(v) => `$${v}`}
                  tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }}
                  axisLine={false} tickLine={false} width={55} />
                <Tooltip
                  formatter={(v, name) => [`$${v}`, name]}
                  contentStyle={{ background: "var(--bg2)", border: "1px solid var(--border2)", borderRadius: 8, fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }}
                />
                <Legend wrapperStyle={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} />
                <Line type="monotone" dataKey="OpenAI" stroke="var(--cyan)" strokeWidth={2} dot={{ fill: "var(--cyan)", r: 4 }} />
                <Line type="monotone" dataKey="Open-Source" stroke="var(--emerald)" strokeWidth={2} strokeDasharray="6 3" dot={{ fill: "var(--emerald)", r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Summary insight */}
          <div className="card" style={{ borderColor: "rgba(52,211,153,.3)", background: "rgba(52,211,153,.05)" }}>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-start" }}>
              <div style={{ fontSize: 28, flexShrink: 0 }}>💡</div>
              <div>
                <div style={{ fontWeight: 700, marginBottom: 6, fontSize: 13 }}>Key insight</div>
                <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.8, fontFamily: "'JetBrains Mono',monospace" }}>
                  At <span style={{ color: "var(--cyan)" }}>100,000 queries/month</span>, OpenAI's API would cost approximately{" "}
                  <span style={{ color: "var(--cyan)" }}>${(100000 * OPENAI_COST_PER_QUERY).toFixed(2)}/month</span>.
                  Open-source alternatives like <span style={{ color: "var(--emerald)" }}>multilingual-e5-base</span> achieve
                  equal precision at <span style={{ color: "var(--emerald)" }}>$0.00</span> — making them the clear choice
                  for cost-sensitive production deployments.
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "questions" && (
        <div className="card">
          <div className="card-title">Per-Question Results ({allQuestions.length} questions)</div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Question</th>
                  <th>Difficulty</th>
                  {models.map((m) => (
                    <th key={m.model_name} style={{ color: modelColor(m.model_name) }}>
                      {modelShort(m.model_name)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {allQuestions.map((row, i) => (
                  <tr key={i}>
                    <td style={{ maxWidth: 240, color: "var(--text)", fontSize: 11 }}>{row.q}</td>
                    <td>
                      <span style={{
                        fontSize: 9, fontWeight: 700, letterSpacing: 1,
                        padding: "2px 6px", borderRadius: 4, textTransform: "uppercase",
                        background: row.difficulty === "easy" ? "rgba(52,211,153,.1)" : row.difficulty === "medium" ? "rgba(251,191,36,.1)" : "rgba(248,113,113,.1)",
                        color: row.difficulty === "easy" ? "var(--emerald)" : row.difficulty === "medium" ? "var(--amber)" : "var(--red)",
                      }}>
                        {row.difficulty}
                      </span>
                    </td>
                    {models.map((m) => {
                      const d = row[m.model_name];
                      return (
                        <td key={m.model_name}>
                          <span style={{ color: d?.hit ? "var(--emerald)" : "var(--red)", fontSize: 14 }}>
                            {d?.hit ? "✓" : "✗"}
                          </span>
                          <span style={{ color: "var(--muted)", marginLeft: 6, fontSize: 10 }}>
                            {d?.score != null ? (d.score * 100).toFixed(0) + "%" : ""}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── View: Embedding Explorer ──────────────────────────────────────────────
function ExplorerView({ evalData }) {
  // Simulate UMAP-like scatter using actual eval scores as proxy coordinates
  // In production you'd call a real UMAP endpoint
  const models = evalData?.models ?? [];

  // Generate mock 2D projections from scoring data for visualization
  const generatePoints = (modelIdx) => {
    const model = models[modelIdx];
    if (!model) return [];
    const results = model.results ?? [];
    return results.map((r, i) => ({
      x: (r.top_score ?? 0.5) * 100 + (Math.sin(i * 0.7 + modelIdx) * 12),
      y: (r.keyword_recall ?? r.keyword_hits / 3 ?? 0.5) * 100 + (Math.cos(i * 0.5 + modelIdx * 2) * 12),
      question: r.question,
      difficulty: r.difficulty,
      hit: r.source_hit,
      model: model.model_name,
    }));
  };

  const allPoints = models.flatMap((_, i) => generatePoints(i));

  const diffColor = { easy: "#34d399", medium: "#fbbf24", hard: "#f87171" };
  const [hovered, setHovered] = useState(null);

  if (!evalData) return (
    <div style={{ textAlign: "center", padding: 60, color: "var(--muted)" }}>
      <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>
        Load evaluation data to explore embeddings.
      </div>
    </div>
  );

  return (
    <div className="stack">
      <div className="card">
        <div className="card-title">Embedding Space Projection — Query Score Distribution</div>
        <p style={{ fontSize: 12, color: "var(--muted)", marginBottom: 20, fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.6 }}>
          Each point is a test question, positioned by its retrieval quality metrics.<br />
          X-axis: top relevance score · Y-axis: keyword recall · Color: difficulty level
        </p>
        <div style={{ position: "relative" }}>
          <ResponsiveContainer width="100%" height={420}>
            <ScatterChart margin={{ top: 20, right: 30, bottom: 40, left: 60 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis type="number" dataKey="x" name="Relevance Score" domain={[0, 110]}
                ticks={[0, 20, 40, 60, 80, 100]}
                tickFormatter={(v) => `${v}`}
                tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false}
                label={{ value: "Relevance Score", position: "insideBottom", offset: -20, fill: "var(--muted)", fontSize: 10 }} />
              <YAxis type="number" dataKey="y" name="Keyword Recall" domain={[0, 120]}
                ticks={[0, 30, 60, 90, 120]}
                tickFormatter={(v) => `${v}`}
                tick={{ fill: "var(--muted)", fontSize: 10 }} axisLine={false} tickLine={false} width={35}
                label={{ value: "Keyword Recall", angle: -90, position: "insideLeft", dx: -20, fill: "var(--muted)", fontSize: 10 }} />
              <ZAxis range={[60, 60]} />
              <Tooltip
                content={({ payload }) => {
                  if (!payload?.length) return null;
                  const d = payload[0]?.payload;
                  if (!d) return null;
                  return (
                    <div style={{ background: "var(--bg2)", border: "1px solid var(--border2)", borderRadius: 8, padding: "10px 14px", fontSize: 11, fontFamily: "'JetBrains Mono',monospace", maxWidth: 240 }}>
                      <div style={{ color: modelColor(d.model), fontWeight: 700, marginBottom: 6 }}>{modelShort(d.model)}</div>
                      <div style={{ color: "var(--muted)", marginBottom: 4, wordBreak: "break-word" }}>{d.question}</div>
                      <div style={{ color: diffColor[d.difficulty] }}>{d.difficulty} · {d.hit ? "✓ hit" : "✗ miss"}</div>
                    </div>
                  );
                }}
              />
              {models.map((m, i) => (
                <Scatter key={m.model_name}
                  name={modelShort(m.model_name)}
                  data={generatePoints(i)}
                  fill={modelColor(m.model_name)}
                  fillOpacity={0.7}
                />
              ))}
              <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: 11, fontFamily: "'JetBrains Mono',monospace", paddingBottom: 10 }} />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid-3">
        {models.map((m) => {
          const hits = m.results?.filter(r => r.source_hit).length ?? 0;
          const total = m.results?.length ?? 1;
          const color = modelColor(m.model_name);
          return (
            <div key={m.model_name} className="metric-card" style={{ borderColor: color + "40" }}>
              <div className="model-name" style={{ color, marginBottom: 8 }}>{MODEL_META[m.model_name]?.icon} {modelShort(m.model_name)}</div>
              <div style={{ fontSize: 11, color: "var(--muted)", fontFamily: "'JetBrains Mono',monospace", display: "flex", flexDirection: "column", gap: 8 }}>
                <div>
                  <div style={{ marginBottom: 4 }}>Source Hits: <span style={{ color }}>{hits}/{total}</span></div>
                  <div className="progress-bar">
                    <div className="progress-fill" style={{ width: `${hits / total * 100}%`, background: color }} />
                  </div>
                </div>
                <div style={{ fontSize: 10 }}>Dim: <span style={{ color: "var(--text)" }}>{
                  { "text-embedding-3-small": 1536, "all-MiniLM-L6-v2": 384, "multilingual-e5-base": 768 }[m.model_name]
                }</span></div>
                <div style={{ fontSize: 10 }}>Type: <span style={{ color: "var(--text)" }}>
                  { { "text-embedding-3-small": "OpenAI API", "all-MiniLM-L6-v2": "HuggingFace", "multilingual-e5-base": "HuggingFace" }[m.model_name] }
                </span></div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── View: Presentation ────────────────────────────────────────────────────
function PresentationView({ evalData }) {
  const [slide, setSlide] = useState(0);

  const models = evalData?.models ?? [];
  const openai  = models.find(m => m.model_name === "text-embedding-3-small");
  const minilm  = models.find(m => m.model_name === "all-MiniLM-L6-v2");
  const e5      = models.find(m => m.model_name === "multilingual-e5-base");

  const fastest = models.length
    ? [...models].sort((a,b) => a.metrics.avg_retrieval_time_ms - b.metrics.avg_retrieval_time_ms)[0]
    : null;

  const slides = [
    // 0 — Title
    <div className="pres-slide" key="title">
      <div className="pres-step">Examensarbete · NBI/Handelsakademin</div>
      <h1 className="pres-h1 glow-text-cyan">
        Embedding Model<br />Comparison in RAG
      </h1>
      <p className="pres-body">
        How does the choice of embedding model affect retrieval quality in a RAG system?<br />
        A comparative study of commercial vs open-source models.
      </p>
    </div>,

    // 1 — The Question
    <div className="pres-slide" key="question">
      <div className="pres-step">Research Question</div>
      <h2 className="pres-h2">What are we measuring?</h2>
      <div className="pres-cards-row">
        {[
          { icon: "🎯", title: "Retrieval Precision", body: "Does the right document surface for a given query?" },
          { icon: "⚡", title: "Response Time", body: "How fast can each model embed and retrieve?" },
          { icon: "💰", title: "Cost & Practicality", body: "Commercial API vs free open-source models." },
        ].map((c) => (
          <div key={c.title} className="pres-card">
            <div className="pres-card-icon">{c.icon}</div>
            <div className="pres-card-title">{c.title}</div>
            <div className="pres-card-body">{c.body}</div>
          </div>
        ))}
      </div>
    </div>,

    // 2 — Models
    <div className="pres-slide" key="models">
      <div className="pres-step">The Contestants</div>
      <h2 className="pres-h2">3 Models · 18 Questions · 3 Difficulty Levels</h2>
      <div className="pres-cards-row">
        {[
          { model: "text-embedding-3-small", label: "Paid · API", desc: "OpenAI's commercial embedding model. 1536 dimensions." },
          { model: "all-MiniLM-L6-v2",       label: "Free · Local", desc: "Lightweight English model. 384 dimensions." },
          { model: "multilingual-e5-base",    label: "Free · Local", desc: "Multilingual model. 768 dimensions. Swedish + English." },
        ].map(({ model, label, desc }) => {
          const color = modelColor(model);
          return (
            <div key={model} className="pres-card" style={{ borderColor: color + "50" }}>
              <div style={{ fontSize: 28, marginBottom: 8 }}>{MODEL_META[model]?.icon}</div>
              <div className="pres-card-title" style={{ color }}>{modelShort(model)}</div>
              <div style={{ fontSize: 10, color: "var(--amber)", fontFamily: "'JetBrains Mono',monospace", marginBottom: 8, fontWeight: 700 }}>{label}</div>
              <div className="pres-card-body">{desc}</div>
            </div>
          );
        })}
      </div>
    </div>,

    // 3 — Key result: hit rate
    <div className="pres-slide" key="hitrate">
      <div className="pres-step">Key Finding · Source Hit Rate</div>
      <h2 className="pres-h2">Open-source matches commercial precision</h2>
      <div className="pres-cards-row">
        {models.map((m) => {
          const color = modelColor(m.model_name);
          return (
            <div key={m.model_name} className="pres-card" style={{ borderColor: color + "50", textAlign: "center" }}>
              <div style={{ fontSize: 24, color, fontFamily: "'JetBrains Mono',monospace", fontWeight: 700 }}>{MODEL_META[m.model_name]?.icon}</div>
              <div style={{ fontSize: 13, fontWeight: 700, color, margin: "8px 0" }}>{modelShort(m.model_name)}</div>
              <div style={{ fontSize: 42, fontWeight: 800, color }}>{pct(m.metrics.source_hit_rate)}</div>
              <div className="pres-card-body">source hit rate</div>
            </div>
          );
        })}
      </div>
    </div>,

    // 4 — Speed
    <div className="pres-slide" key="speed">
      <div className="pres-step">Key Finding · Speed</div>
      <h2 className="pres-h2">Local models are dramatically faster</h2>
      {fastest && (
        <>
          <div className="pres-big-stat">{ms(fastest.metrics.avg_retrieval_time_ms)}</div>
          <p className="pres-body">
            <span className="pres-highlight">{modelShort(fastest.model_name)}</span> was the fastest model,
            running entirely locally — no API calls, no latency, no cost per query.
          </p>
        </>
      )}
    </div>,

    // 5 — Conclusion
    <div className="pres-slide" key="conclusion">
      <div className="pres-step">Conclusion</div>
      <h2 className="pres-h2 glow-text-emerald">Open-source wins on value</h2>
      <p className="pres-body" style={{ marginBottom: 32 }}>
        For RAG systems where cost and latency matter, open-source embedding models offer
        a compelling alternative to commercial APIs — without sacrificing retrieval quality.
      </p>
      <div className="pres-cards-row" style={{ maxWidth: 700 }}>
        {[
          { icon: "✓", label: "Equal precision", color: "var(--emerald)" },
          { icon: "⚡", label: "Faster response", color: "var(--cyan)" },
          { icon: "€0", label: "Zero API cost", color: "var(--violet)" },
        ].map((c) => (
          <div key={c.label} className="pres-card" style={{ borderColor: c.color + "50", textAlign: "center" }}>
            <div style={{ fontSize: 32, color: c.color }}>{c.icon}</div>
            <div style={{ fontWeight: 700, marginTop: 8 }}>{c.label}</div>
          </div>
        ))}
      </div>
    </div>,
  ];

  const prev = () => setSlide((s) => Math.max(0, s - 1));
  const next = () => setSlide((s) => Math.min(slides.length - 1, s + 1));

  useEffect(() => {
    const handler = (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowDown") next();
      if (e.key === "ArrowLeft"  || e.key === "ArrowUp")   prev();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "calc(100vh - 52px - 56px)", margin: "-28px", padding: "0 28px" }}>
      <div style={{ flex: 1, overflow: "hidden", position: "relative", display: "flex", alignItems: "center", justifyContent: "center" }}>
        <div style={{ width: "100%" }}>{slides[slide]}</div>
      </div>
      <div style={{ flexShrink: 0, paddingBottom: 20 }}>
        <div className="pres-nav">
          <button className="btn btn-ghost" onClick={prev} disabled={slide === 0} style={{ fontSize: 18, padding: "8px 16px" }}>←</button>
          {slides.map((_, i) => (
            <div key={i} className={`pres-dot ${i === slide ? "active" : ""}`} onClick={() => setSlide(i)} />
          ))}
          <button className="btn btn-ghost" onClick={next} disabled={slide === slides.length - 1} style={{ fontSize: 18, padding: "8px 16px" }}>→</button>
        </div>
        <div style={{ textAlign: "center", marginTop: 8, fontSize: 10, fontFamily: "'JetBrains Mono',monospace", color: "var(--muted)" }}>
          {slide + 1} / {slides.length} · Use arrow keys to navigate
        </div>
      </div>
    </div>
  );
}

// ─── App Shell ─────────────────────────────────────────────────────────────
const VIEWS = [
  { id: "compare",      label: "A/B Compare",    icon: "⊞" },
  { id: "benchmark",    label: "Benchmark",       icon: "◈" },
  { id: "explorer",     label: "Embedding Space", icon: "⬡" },
  { id: "presentation", label: "Presentation",    icon: "▷" },
];

export default function App() {
  const [view,         setView]         = useState("compare");
  const [evalData,     setEvalData]     = useState(null);
  const [indexedModels, setIndexedModels] = useState([]);
  const [apiStatus,    setApiStatus]    = useState("checking");

  // Inject styles
  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  // Load data
  useEffect(() => {
    // Health check
    apiFetch("/health")
      .then((d) => {
        setIndexedModels(d.indexed_models ?? []);
        setApiStatus("online");
      })
      .catch(() => setApiStatus("offline"));

    // Evaluation data
    apiFetch("/evaluation")
      .then(setEvalData)
      .catch(() => {
        // Try to load from the uploaded file as fallback
        setEvalData(null);
      });
  }, []);

  const viewMeta = VIEWS.find((v) => v.id === view);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>RAG<br />Embedding<br />Lab</h1>
          <p>Tolkien Chatbot · Thesis</p>
        </div>

        {VIEWS.map((v) => (
          <div key={v.id}
            className={`nav-item ${view === v.id ? "active" : ""}`}
            onClick={() => setView(v.id)}>
            <span className="icon">{v.icon}</span>
            {v.label}
          </div>
        ))}

        <div className="sidebar-footer">
          <div>
            <span className={`status-dot`} style={{ background: apiStatus === "online" ? "var(--emerald)" : "var(--red)" }} />
            API {apiStatus}
          </div>
          {indexedModels.length > 0 && (
            <div style={{ marginTop: 4 }}>
              {indexedModels.length} model{indexedModels.length > 1 ? "s" : ""} indexed
            </div>
          )}
          <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
            NBI/Handelsakademin<br />
            Examensarbete 2026
          </div>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <div>
            <div className="topbar-title">{viewMeta?.label}</div>
            <div className="topbar-sub">
              {view === "compare"      && "Query all models simultaneously"}
              {view === "benchmark"    && "Evaluation results — 18 test questions"}
              {view === "explorer"     && "Visualize retrieval quality distribution"}
              {view === "presentation" && "Slide deck — use arrow keys to navigate"}
            </div>
          </div>
          <div className="topbar-sep" />
          <div style={{ display: "flex", gap: 8 }}>
            {(indexedModels.length > 0 ? indexedModels : Object.keys(MODEL_META)).map((m) => (
              <ModelBadge key={m} model={m} />
            ))}
          </div>
        </div>

        <div className="content">
          {view === "compare"      && <CompareView indexedModels={indexedModels} />}
          {view === "benchmark"    && <BenchmarkView evalData={evalData} />}
          {view === "explorer"     && <ExplorerView evalData={evalData} />}
          {view === "presentation" && <PresentationView evalData={evalData} />}
        </div>
      </div>
    </div>
  );
}