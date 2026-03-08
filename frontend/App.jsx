import { useState, useEffect, useRef, useCallback } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from "recharts";

// ─── Config ────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

const MODEL_META = {
  "text-embedding-3-small": { color: "#22d3ee", short: "OpenAI",  type: "OpenAI API" },
  "all-MiniLM-L6-v2":       { color: "#a78bfa", short: "MiniLM",  type: "HuggingFace" },
  "multilingual-e5-base":   { color: "#34d399", short: "E5-base", type: "HuggingFace" },
};

const DEMO_QUESTIONS = [
  "Who is Gandalf?",
  "What is the One Ring?",
  "Who are the Nazgûl?",
  "What happened at Mount Doom?",
];

const CSS = `
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Syne:wght@400;600;700;800&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:      #080c14;
    --bg2:     #0d1220;
    --bg3:     #111827;
    --border:  #1e2d45;
    --border2: #253550;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --cyan:    #22d3ee;
    --violet:  #a78bfa;
    --emerald: #34d399;
  }

  html, body, #root { height: 100%; }
  body { background: var(--bg); color: var(--text); font-family: 'Syne', sans-serif; }

  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

  .app { display: flex; height: 100vh; }

  /* Sidebar */
  .sidebar {
    width: 200px; flex-shrink: 0;
    background: var(--bg2); border-right: 1px solid var(--border);
    display: flex; flex-direction: column; padding: 24px 0;
  }
  .sidebar-brand { padding: 0 18px 20px; border-bottom: 1px solid var(--border); margin-bottom: 12px; }
  .sidebar-brand h1 {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 700; letter-spacing: 3px;
    text-transform: uppercase; color: var(--cyan); line-height: 1.5;
  }
  .sidebar-brand p { font-size: 10px; color: var(--muted); margin-top: 3px; font-family: 'JetBrains Mono', monospace; }

  .nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 18px; cursor: pointer;
    font-size: 11px; font-weight: 600; letter-spacing: 1px;
    text-transform: uppercase; color: var(--muted);
    border-left: 2px solid transparent; transition: all .2s;
  }
  .nav-item:hover { color: var(--text); }
  .nav-item.active { color: var(--cyan); border-left-color: var(--cyan); background: rgba(34,211,238,.05); }

  .sidebar-footer {
    margin-top: auto; padding: 16px 18px;
    border-top: 1px solid var(--border);
    font-size: 10px; color: var(--muted);
    font-family: 'JetBrains Mono', monospace; line-height: 1.8;
  }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 5px; }

  /* Main */
  .main { flex: 1; display: flex; flex-direction: column; overflow: hidden; }
  .topbar {
    height: 50px; flex-shrink: 0;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; padding: 0 24px; gap: 12px;
    background: rgba(13,18,32,.9);
  }
  .topbar-title { font-size: 13px; font-weight: 700; }
  .topbar-sub { font-size: 11px; color: var(--muted); font-family: 'JetBrains Mono', monospace; }
  .topbar-sep { flex: 1; }

  .model-badge {
    display: flex; align-items: center; gap: 5px;
    padding: 3px 9px; border-radius: 20px;
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
  }

  .content { flex: 1; overflow-y: auto; padding: 24px; }
  .stack { display: flex; flex-direction: column; gap: 16px; }
  .grid-3 { display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; }

  /* Cards */
  .card { background: var(--bg2); border: 1px solid var(--border); border-radius: 10px; padding: 18px; }
  .card-title {
    font-size: 10px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: var(--muted); margin-bottom: 14px;
    display: flex; align-items: center; gap: 8px;
  }
  .card-title::before { content: ''; display: block; width: 3px; height: 12px; border-radius: 2px; background: var(--cyan); }

  /* Compare */
  .demo-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
  .demo-pill {
    padding: 5px 11px; border-radius: 20px;
    background: var(--bg3); border: 1px solid var(--border2);
    font-size: 11px; cursor: pointer; color: var(--muted);
    transition: all .2s; font-family: 'JetBrains Mono', monospace;
  }
  .demo-pill:hover { border-color: var(--cyan); color: var(--cyan); }

  .input-row { display: flex; gap: 10px; margin-bottom: 20px; }
  .input-row input {
    flex: 1; padding: 11px 14px;
    background: var(--bg3); border: 1px solid var(--border2);
    border-radius: 8px; color: var(--text);
    font-family: 'Syne', sans-serif; font-size: 13px; outline: none;
    transition: border-color .2s;
  }
  .input-row input:focus { border-color: var(--cyan); }
  .input-row input::placeholder { color: var(--muted); }

  .btn {
    padding: 11px 18px; border-radius: 8px; border: none; cursor: pointer;
    font-family: 'Syne', sans-serif; font-size: 12px; font-weight: 700;
    transition: all .2s; display: flex; align-items: center; gap: 7px;
  }
  .btn-primary { background: var(--cyan); color: #000; }
  .btn-primary:hover { filter: brightness(1.1); }
  .btn-primary:disabled { opacity: .4; cursor: not-allowed; }
  .btn-ghost {
    background: var(--bg3); color: var(--muted);
    border: 1px solid var(--border2);
  }
  .btn-ghost:hover { color: var(--text); border-color: var(--border2); }

  .compare-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 14px; }

  .answer-card {
    background: var(--bg3); border-radius: 10px;
    padding: 16px; border: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 12px;
  }
  .answer-card.loaded { animation: fadeUp .35s ease; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:none} }

  .model-header { display: flex; align-items: center; justify-content: space-between; }
  .model-name-label {
    font-size: 11px; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; font-family: 'JetBrains Mono', monospace;
  }
  .type-tag {
    font-size: 9px; font-weight: 700; padding: 2px 6px;
    border-radius: 8px; text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace; letter-spacing: .5px;
  }

  .answer-text { font-size: 12px; line-height: 1.7; min-height: 70px; }
  .skel { height: 11px; border-radius: 3px; background: var(--border); animation: shimmer 1.5s infinite; margin-bottom: 7px; }
  @keyframes shimmer { 0%,100%{opacity:.4} 50%{opacity:.8} }

  .meta-row { display: flex; gap: 8px; flex-wrap: wrap; }
  .meta-chip {
    font-size: 10px; font-family: 'JetBrains Mono', monospace;
    color: var(--muted); background: var(--bg2);
    padding: 2px 7px; border-radius: 5px; border: 1px solid var(--border);
  }
  .meta-chip b { color: var(--text); }

  .sources-list { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--muted); }
  .sources-list div::before { content: '↳ '; color: var(--cyan); }

  /* History */
  .history-entry {
    border-radius: 10px; overflow: hidden;
    border: 1px solid var(--border); background: var(--bg2);
  }
  .history-header {
    padding: 10px 16px; border-bottom: 1px solid var(--border);
    display: flex; align-items: center; justify-content: space-between;
    background: rgba(255,255,255,.02);
  }
  .history-q { display: flex; align-items: center; gap: 10px; }
  .q-num {
    font-size: 10px; font-weight: 700; letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace; color: var(--cyan);
  }
  .q-text { font-size: 13px; font-weight: 600; }
  .q-time { font-size: 10px; font-family: 'JetBrains Mono', monospace; color: var(--muted); }

  /* Benchmark */
  .tabs { display: flex; gap: 4px; margin-bottom: 18px; }
  .tab {
    padding: 6px 13px; border-radius: 6px; font-size: 10px;
    font-weight: 700; letter-spacing: 1px; text-transform: uppercase;
    cursor: pointer; border: 1px solid transparent;
    font-family: 'JetBrains Mono', monospace; color: var(--muted);
    transition: all .2s;
  }
  .tab.active { border-color: var(--cyan); color: var(--cyan); background: rgba(34,211,238,.07); }
  .tab:hover:not(.active) { color: var(--text); border-color: var(--border2); }

  .metric-card {
    background: var(--bg3); border-radius: 8px;
    padding: 14px 16px; border: 1px solid var(--border);
  }
  .metric-label { font-size: 9px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: var(--muted); font-family: 'JetBrains Mono', monospace; margin-bottom: 4px; }
  .metric-value { font-size: 24px; font-weight: 800; line-height: 1; }

  .spinner { width: 15px; height: 15px; border: 2px solid var(--border2); border-top-color: var(--cyan); border-radius: 50%; animation: spin .7s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .empty-state { text-align: center; padding: 50px 20px; color: var(--muted); }
  .empty-state .icon { font-size: 32px; margin-bottom: 10px; }
  .empty-state p { font-size: 12px; font-family: 'JetBrains Mono', monospace; line-height: 1.8; }

  .progress-bar { height: 3px; border-radius: 2px; background: var(--border); overflow: hidden; margin-top: 4px; }
  .progress-fill { height: 100%; border-radius: 2px; }
`;

const pct = (v) => v != null ? `${(v * 100).toFixed(1)}%` : "—";
const ms  = (v) => v != null ? `${v.toFixed(0)}ms` : "—";
const modelColor = (m) => MODEL_META[m]?.color ?? "#64748b";
const modelShort = (m) => MODEL_META[m]?.short ?? m;

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

// ─── Answer Card ───────────────────────────────────────────────────────────
function AnswerCard({ model, data, loading }) {
  const color = modelColor(model);
  const meta  = MODEL_META[model] ?? {};
  return (
    <div className={`answer-card ${data ? "loaded" : ""}`}
      style={{ borderColor: data ? color + "50" : undefined }}>
      <div className="model-header">
        <div className="model-name-label" style={{ color }}>{meta.short}</div>
        <span className="type-tag"
          style={{ background: color + "15", color, border: `1px solid ${color}30` }}>
          {meta.type}
        </span>
      </div>
      {loading ? (
        <>
          <div className="skel" style={{ width: "88%" }} />
          <div className="skel" style={{ width: "73%" }} />
          <div className="skel" style={{ width: "80%" }} />
        </>
      ) : data ? (
        <>
          <p className="answer-text">{data.answer}</p>
          <div className="meta-row">
            <span className="meta-chip">⚡ <b>{ms(data.retrieval_time_ms)}</b></span>
            {data.top_score != null && <span className="meta-chip">🎯 <b>{(data.top_score*100).toFixed(0)}%</b></span>}
          </div>
          {data.sources?.length > 0 && (
            <div className="sources-list">
              {data.sources.slice(0,3).map((s,i) => <div key={i}>{s.split("/").pop()}</div>)}
            </div>
          )}
        </>
      ) : (
        <p className="answer-text" style={{ color: "var(--muted)", fontSize: 12 }}>Waiting…</p>
      )}
    </div>
  );
}

// ─── Compare View ──────────────────────────────────────────────────────────
function CompareView({ indexedModels }) {
  const [question, setQuestion]       = useState("");
  const [loading, setLoading]         = useState(false);
  const [error, setError]             = useState(null);
  const [history, setHistory]         = useState([]);
  const [loadingQuestion, setLoadingQuestion] = useState(null);

  const displayModels = indexedModels.length > 0 ? indexedModels : Object.keys(MODEL_META);

  const submit = useCallback(async (q) => {
    const text = (q || question).trim();
    if (!text) return;
    setLoading(true); setLoadingQuestion(text); setError(null); setQuestion("");
    try {
      const data = await apiFetch("/compare", {
        method: "POST",
        body: JSON.stringify({ question: text, k: 4, threshold: 0.35 }),
      });
      const map = {};
      for (const r of data.results) map[r.model] = r;
      setHistory((prev) => [{ question: text, results: map, timestamp: new Date().toLocaleTimeString("sv-SE", { hour: "2-digit", minute: "2-digit" }) }, ...prev]);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false); setLoadingQuestion(null);
    }
  }, [question]);

  const onKey = (e) => { if (e.key === "Enter") submit(); };

  return (
    <div className="stack">
      <div className="demo-pills">
        {DEMO_QUESTIONS.map((q) => (
          <div key={q} className="demo-pill" onClick={() => { setQuestion(q); submit(q); }}>{q}</div>
        ))}
      </div>
      <div className="input-row">
        <input value={question} onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKey} placeholder="Ask about Tolkien's world…" />
        <button className="btn btn-primary" onClick={() => submit()} disabled={loading}>
          {loading ? <div className="spinner" /> : "▶"}
          {loading ? "Querying…" : "Compare"}
        </button>
        {history.length > 0 && (
          <button className="btn btn-ghost" onClick={() => setHistory([])}>✕ Clear</button>
        )}
      </div>
      {error && <div style={{ color: "var(--red)", fontSize: 12, fontFamily: "'JetBrains Mono',monospace", background: "rgba(248,113,113,.08)", padding: "10px 14px", borderRadius: 8, border: "1px solid rgba(248,113,113,.2)" }}>⚠ {error}</div>}

      {history.length === 0 && !loading && (
        <div className="empty-state">
          <div className="icon">⬡</div>
          <p>Ask a question to compare all three models side by side.<br />Your history will appear here.</p>
        </div>
      )}

      <div className="stack">
        {loading && loadingQuestion && (
          <div className="history-entry">
            <div className="history-header">
              <div className="history-q">
                <div className="spinner" style={{ width: 12, height: 12, borderWidth: 1.5 }} />
                <span className="q-text">{loadingQuestion}</span>
              </div>
            </div>
            <div className="compare-grid" style={{ padding: 14 }}>
              {displayModels.map((m) => <AnswerCard key={m} model={m} data={null} loading={true} />)}
            </div>
          </div>
        )}
        {history.map((entry, idx) => (
          <div key={idx} className="history-entry">
            <div className="history-header">
              <div className="history-q">
                <span className="q-num">Q{history.length - idx}</span>
                <span className="q-text">{entry.question}</span>
              </div>
              <span className="q-time">{entry.timestamp}</span>
            </div>
            <div className="compare-grid" style={{ padding: 14 }}>
              {displayModels.map((m) => <AnswerCard key={m} model={m} data={entry.results[m]} loading={false} />)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Benchmark View ────────────────────────────────────────────────────────
function BenchmarkView({ evalData }) {
  const [activeTab, setActiveTab] = useState("overview");

  if (!evalData) return (
    <div className="empty-state" style={{ padding: 60 }}>
      <div className="icon">📊</div>
      <p>No evaluation data found.<br />Run <code style={{ color: "var(--cyan)" }}>evaluate.py</code> first.</p>
    </div>
  );

  const models = evalData.models ?? [];

  const radarData = [
    { metric: "Hit Rate",        ...Object.fromEntries(models.map(m => [modelShort(m.model_name), m.metrics.source_hit_rate * 100])) },
    { metric: "Keyword Recall",  ...Object.fromEntries(models.map(m => [modelShort(m.model_name), m.metrics.avg_keyword_recall * 100])) },
    { metric: "Relevance Score", ...Object.fromEntries(models.map(m => [modelShort(m.model_name), m.metrics.avg_top_score * 100])) },
  ];

  const difficultyData = ["easy", "medium", "hard"].map((d) => ({
    difficulty: d,
    ...Object.fromEntries(models.map((m) => [modelShort(m.model_name), (m.by_difficulty?.[d] ?? 0) * 100])),
  }));

  return (
    <div className="stack">
      <div className="tabs">
        {["overview", "difficulty"].map((t) => (
          <div key={t} className={`tab ${activeTab === t ? "active" : ""}`} onClick={() => setActiveTab(t)}>{t}</div>
        ))}
      </div>

      {activeTab === "overview" && (
        <div className="stack">
          <div className="grid-3">
            {models.map((m) => {
              const color = modelColor(m.model_name);
              return (
                <div key={m.model_name} className="metric-card" style={{ borderColor: color + "40" }}>
                  <div className="model-name-label" style={{ color, marginBottom: 10 }}>{modelShort(m.model_name)}</div>
                  {[
                    { label: "Source Hit Rate", val: pct(m.metrics.source_hit_rate) },
                    { label: "Keyword Recall",  val: pct(m.metrics.avg_keyword_recall) },
                    { label: "Avg Response",    val: ms(m.metrics.avg_retrieval_time_ms) },
                  ].map(({ label, val }) => (
                    <div key={label} style={{ marginBottom: 10 }}>
                      <div className="metric-label">{label}</div>
                      <div className="metric-value" style={{ color, fontSize: 20 }}>{val}</div>
                      <div className="progress-bar">
                        <div className="progress-fill" style={{ width: val.includes("%") ? val : "60%", background: color }} />
                      </div>
                    </div>
                  ))}
                </div>
              );
            })}
          </div>

          <div className="card">
            <div className="card-title">Performance Radar</div>
            <ResponsiveContainer width="100%" height={280}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="var(--border)" />
                <PolarAngleAxis dataKey="metric" tick={{ fill: "var(--muted)", fontSize: 10, fontFamily: "'JetBrains Mono',monospace" }} />
                <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                {models.map((m) => (
                  <Radar key={m.model_name} name={modelShort(m.model_name)}
                    dataKey={modelShort(m.model_name)}
                    stroke={modelColor(m.model_name)} fill={modelColor(m.model_name)} fillOpacity={0.1} strokeWidth={2} />
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
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={difficultyData} barGap={4}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
              <XAxis dataKey="difficulty" tick={{ fill: "var(--muted)", fontSize: 11, fontFamily: "'JetBrains Mono',monospace" }} axisLine={false} tickLine={false} />
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
    </div>
  );
}

// ─── App Shell ─────────────────────────────────────────────────────────────
const VIEWS = [
  { id: "compare",   label: "A/B Compare", icon: "⊞" },
  { id: "benchmark", label: "Benchmark",   icon: "◈" },
];

export default function App() {
  const [view, setView]               = useState("compare");
  const [evalData, setEvalData]       = useState(null);
  const [indexedModels, setIndexedModels] = useState([]);
  const [apiStatus, setApiStatus]     = useState("checking");

  useEffect(() => {
    const style = document.createElement("style");
    style.textContent = CSS;
    document.head.appendChild(style);
    return () => document.head.removeChild(style);
  }, []);

  useEffect(() => {
    apiFetch("/health")
      .then((d) => { setIndexedModels(d.indexed_models ?? []); setApiStatus("online"); })
      .catch(() => setApiStatus("offline"));
    apiFetch("/evaluation").then(setEvalData).catch(() => {});
  }, []);

  const viewMeta = VIEWS.find((v) => v.id === view);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>RAG<br />Embedding<br />Lab</h1>
          <p>Tolkien · Kevin</p>
        </div>
        {VIEWS.map((v) => (
          <div key={v.id} className={`nav-item ${view === v.id ? "active" : ""}`} onClick={() => setView(v.id)}>
            <span>{v.icon}</span>{v.label}
          </div>
        ))}
        <div className="sidebar-footer">
          <div>
            <span className="dot" style={{ background: apiStatus === "online" ? "var(--emerald)" : "#f87171" }} />
            API {apiStatus}
          </div>
          <div style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
            NBI/Handelsakademin<br />v2.0 · April 2026
          </div>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <div>
            <div className="topbar-title">{viewMeta?.label}</div>
            <div className="topbar-sub">
              {view === "compare"   && "Query all models simultaneously"}
              {view === "benchmark" && "Evaluation results — 18 test questions"}
            </div>
          </div>
          <div className="topbar-sep" />
          <div style={{ display: "flex", gap: 6 }}>
            {(indexedModels.length > 0 ? indexedModels : Object.keys(MODEL_META)).map((m) => {
              const color = modelColor(m);
              return (
                <div key={m} className="model-badge"
                  style={{ background: color + "18", border: `1px solid ${color}40`, color }}>
                  {modelShort(m)}
                </div>
              );
            })}
          </div>
        </div>
        <div className="content">
          {view === "compare"   && <CompareView indexedModels={indexedModels} />}
          {view === "benchmark" && <BenchmarkView evalData={evalData} />}
        </div>
      </div>
    </div>
  );
}