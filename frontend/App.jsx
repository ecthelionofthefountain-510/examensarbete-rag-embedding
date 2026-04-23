import { useState, useEffect, useRef, useCallback } from "react";
import {
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from "recharts";

// ─── Config ────────────────────────────────────────────────────────────────
const API = "http://localhost:8000";

const MODEL_META = {
  "text-embedding-3-small": { color: "#22d3ee", short: "OpenAI", icon: "⬡" },
  "all-MiniLM-L6-v2": { color: "#a78bfa", short: "MiniLM", icon: "◈" },
  "multilingual-e5-base": { color: "#34d399", short: "E5-base", icon: "◉" },
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
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Space+Grotesk:wght@500;700&family=Syne:wght@400;700;800&display=swap');

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
    --num-font: 'Space Grotesk', 'Syne', sans-serif;
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

  .compare-empty {
    position: relative;
    text-align: center;
    padding: 64px 24px 80px;
    border-radius: 16px;
    border: 1px solid var(--border2);
    background:
      radial-gradient(circle at 50% 18%, rgba(34,211,238,.16), transparent 42%),
      linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)),
      rgba(7,12,24,.45);
  }
  .compare-empty::before {
    content: '';
    position: absolute;
    width: 280px;
    height: 280px;
    left: 50%;
    top: -120px;
    transform: translateX(-50%);
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34,211,238,.16), transparent 70%);
    filter: blur(8px);
    pointer-events: none;
    animation: palantirPulse 5s ease-in-out infinite;
  }
  .compare-empty-core {
    position: relative;
    z-index: 1;
    width: 158px;
    height: 158px;
    margin: 0 auto 22px;
    display: grid;
    place-items: center;
    border-radius: 50%;
    overflow: hidden;
    background:
      radial-gradient(circle at 34% 24%, rgba(255,255,255,.22), rgba(154,166,184,.12) 22%, rgba(16,20,30,.94) 58%, rgba(4,6,10,.99) 100%),
      radial-gradient(circle at 62% 74%, rgba(72,82,98,.2), transparent 56%);
    box-shadow:
      inset 0 0 46px rgba(220,228,242,.1),
      inset 0 -24px 36px rgba(0,0,0,.7),
      0 0 26px rgba(80, 150, 255, .22),
      0 0 48px rgba(255, 90, 30, .14),
      0 26px 42px rgba(0,0,0,.5);
    animation: palantirFloat 8.6s ease-in-out infinite, palantirBreath 6.8s ease-in-out infinite;
  }
  .palantir-liquid {
    position: absolute;
    inset: 6%;
    border-radius: 50%;
    background:
      radial-gradient(circle at 24% 30%, rgba(180, 220, 255, .42), transparent 36%),
      radial-gradient(circle at 70% 42%, rgba(120, 170, 240, .26), transparent 42%),
      radial-gradient(circle at 44% 74%, rgba(90, 130, 210, .22), transparent 46%),
      radial-gradient(circle at 52% 48%, rgba(18, 24, 48, .6), transparent 68%);
    filter: blur(2.8px);
    opacity: .7;
    mix-blend-mode: screen;
    animation: palantirLiquid 14s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-mist {
    position: absolute;
    inset: 7%;
    border-radius: 50%;
    background:
      radial-gradient(ellipse at 32% 24%, rgba(200, 225, 255, .28), transparent 48%),
      radial-gradient(ellipse at 66% 62%, rgba(140, 170, 220, .22), transparent 52%),
      radial-gradient(ellipse at 40% 72%, rgba(100, 140, 200, .2), transparent 46%);
    filter: blur(3px);
    mix-blend-mode: screen;
    opacity: .62;
    animation: palantirMistDrift 19s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-depth {
    position: absolute;
    inset: 11%;
    border-radius: 50%;
    background:
      conic-gradient(from 120deg,
        rgba(40, 10, 5, .22),
        rgba(120, 30, 10, .38),
        rgba(20, 8, 4, .18),
        rgba(90, 20, 5, .32),
        rgba(40, 10, 5, .22));
    filter: blur(5px);
    mix-blend-mode: multiply;
    opacity: .85;
    animation: palantirDepthRoll 24s linear infinite;
    pointer-events: none;
  }
  .palantir-vortex {
    position: absolute;
    inset: 9%;
    border-radius: 50%;
    background:
      conic-gradient(from 30deg,
        rgba(180, 210, 255, .2),
        rgba(50, 90, 160, .14),
        rgba(200, 225, 255, .24),
        rgba(40, 70, 140, .14),
        rgba(180, 210, 255, .2));
    filter: blur(6px);
    mix-blend-mode: screen;
    opacity: .58;
    animation: palantirVortexSpin 32s linear infinite;
    pointer-events: none;
  }
  .palantir-wisps {
    position: absolute;
    inset: 5%;
    border-radius: 50%;
    background:
      radial-gradient(ellipse at 22% 44%, rgba(200, 225, 255, .24), transparent 36%),
      radial-gradient(ellipse at 58% 28%, rgba(255, 170, 90, .22), transparent 34%),
      radial-gradient(ellipse at 74% 58%, rgba(255, 120, 50, .24), transparent 38%),
      radial-gradient(ellipse at 38% 76%, rgba(120, 160, 220, .24), transparent 40%);
    filter: blur(4px);
    mix-blend-mode: screen;
    opacity: .6;
    animation: palantirWispsShift 21s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-current {
    position: absolute;
    inset: 13%;
    border-radius: 50%;
    background:
      linear-gradient(124deg,
        rgba(200, 230, 255, .22) 8%,
        transparent 28%,
        rgba(150, 190, 240, .24) 46%,
        transparent 62%,
        rgba(180, 220, 250, .2) 78%,
        transparent 92%);
    filter: blur(3.5px);
    mix-blend-mode: screen;
    opacity: .5;
    animation: palantirCurrentSweep 17s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-cloud-a {
    position: absolute;
    inset: 4%;
    border-radius: 50%;
    background:
      radial-gradient(ellipse at 26% 38%, rgba(220, 240, 255, .36), transparent 34%),
      radial-gradient(ellipse at 66% 56%, rgba(140, 180, 230, .3), transparent 42%),
      radial-gradient(ellipse at 46% 70%, rgba(90, 130, 200, .34), transparent 40%);
    mix-blend-mode: screen;
    filter: blur(5px);
    opacity: .58;
    animation: palantirCloudA 12s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-cloud-b {
    position: absolute;
    inset: 10%;
    border-radius: 50%;
    background:
      radial-gradient(ellipse at 34% 62%, rgba(255, 170, 90, .3), transparent 36%),
      radial-gradient(ellipse at 74% 34%, rgba(220, 70, 25, .28), transparent 36%),
      radial-gradient(ellipse at 50% 50%, rgba(40, 10, 5, .45), transparent 60%);
    mix-blend-mode: screen;
    filter: blur(4px);
    opacity: .5;
    animation: palantirCloudB 9.5s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-streaks {
    position: absolute;
    inset: 8%;
    border-radius: 50%;
    background:
      linear-gradient(130deg,
        transparent 8%,
        rgba(200, 230, 255, .24) 24%,
        transparent 42%,
        rgba(255, 140, 60, .26) 56%,
        transparent 74%),
      linear-gradient(18deg,
        transparent 10%,
        rgba(180, 220, 255, .18) 30%,
        transparent 52%,
        rgba(220, 80, 30, .22) 66%,
        transparent 86%);
    mix-blend-mode: screen;
    filter: blur(2.8px);
    opacity: .48;
    animation: palantirStreakFlow 6.8s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-smoke {
    position: absolute;
    inset: 3%;
    border-radius: 50%;
    background:
      radial-gradient(circle at 28% 72%, rgba(0,0,0,.58), transparent 48%),
      radial-gradient(circle at 64% 62%, rgba(20, 5, 2, .5), transparent 54%),
      radial-gradient(circle at 52% 34%, rgba(120, 40, 20, .22), transparent 56%);
    filter: blur(2.8px);
    animation: palantirSmoke 12s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-blue-core {
    position: absolute;
    inset: 24%;
    border-radius: 50%;
    background: radial-gradient(
      circle at 50% 50%,
      #eaf6ff 0%,
      #9ccfff 14%,
      #3b8dff 34%,
      #1a4db0 58%,
      #0a1f50 80%,
      transparent 100%
    );
    filter: blur(5px);
    mix-blend-mode: screen;
    animation: palantirBlueCore 3.5s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-red-core {
    position: absolute;
    inset: 20%;
    border-radius: 50%;
    background: radial-gradient(
      ellipse 55% 40% at 70% 58%,
      #ffd080 0%,
      #ff7020 22%,
      #d02005 48%,
      #600805 72%,
      transparent 92%
    );
    filter: blur(8px);
    mix-blend-mode: screen;
    animation: palantirRedCore 6s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-clash {
    position: absolute;
    inset: 17%;
    border-radius: 50%;
    background: radial-gradient(
      circle at 50% 50%,
      rgba(180, 80, 200, .42) 0%,
      rgba(130, 50, 180, .2) 32%,
      transparent 65%
    );
    filter: blur(9px);
    mix-blend-mode: screen;
    animation: palantirClash 4s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-arc {
    position: absolute;
    inset: 10%;
    border-radius: 50%;
    border-top: 1px solid rgba(246,252,255,.24);
    border-right: 1px solid rgba(190,204,224,.16);
    border-left: 1px solid transparent;
    border-bottom: 1px solid transparent;
    mix-blend-mode: screen;
    pointer-events: none;
  }
  .palantir-arc.a {
    transform: rotate(18deg);
    animation: palantirArcOne 16s linear infinite;
    opacity: .22;
  }
  .palantir-arc.b {
    inset: 17%;
    border-top-width: 1.2px;
    border-top-color: rgba(236, 244, 255, .24);
    border-right-color: rgba(190, 204, 224, .18);
    transform: rotate(46deg);
    animation: palantirArcTwo 20s linear infinite reverse;
    opacity: .16;
  }
  .compare-empty-core::before {
    content: '';
    position: absolute;
    inset: -14%;
    border-radius: 50%;
    background:
      radial-gradient(circle at 34% 30%, rgba(248,252,255,.18), transparent 44%),
      radial-gradient(circle at 62% 66%, rgba(184,198,220,.2), transparent 48%),
      radial-gradient(circle at 52% 48%, rgba(16,20,30,.68), transparent 62%);
    filter: blur(8px) saturate(68%);
    mix-blend-mode: screen;
    animation: palantirSwirl 26s ease-in-out infinite;
    pointer-events: none;
  }
  .compare-empty-core::after {
    content: '';
    position: absolute;
    width: 52px;
    height: 16px;
    top: 15px;
    left: 19px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(255,255,255,.56), rgba(255,255,255,0) 74%);
    transform: rotate(-19deg);
    filter: blur(.8px);
    pointer-events: none;
  }
  .palantir-ring {
    position: absolute;
    width: 82%;
    height: 82%;
    border-radius: 50%;
    border: 1px solid rgba(134,154,184,.1);
    box-shadow: inset 0 0 10px rgba(132,162,208,.06);
    animation: palantirRingPulse 12s ease-in-out infinite;
    opacity: .35;
  }
  .palantir-glass-sheen {
    position: absolute;
    inset: 0;
    border-radius: 50%;
    background:
      radial-gradient(ellipse at 34% 16%, rgba(255,255,255,.44), rgba(255,255,255,0) 40%),
      radial-gradient(ellipse at 70% 30%, rgba(230,238,250,.16), rgba(230,238,250,0) 46%),
      radial-gradient(ellipse at 50% 86%, rgba(0,0,0,.26), rgba(0,0,0,0) 52%);
    mix-blend-mode: screen;
    opacity: .62;
    animation: palantirSheenDrift 12s ease-in-out infinite;
    pointer-events: none;
  }
  .palantir-sparkles {
    position: absolute;
    inset: -24%;
    animation: palantirOrbit 34s linear infinite;
    pointer-events: none;
    opacity: .24;
  }
  .palantir-spark {
    position: absolute;
    width: 3px;
    height: 3px;
    border-radius: 50%;
    background: rgba(198, 224, 255, .46);
    box-shadow: 0 0 6px rgba(164, 196, 236, .28);
    opacity: .2;
    animation: palantirSparkle 6.6s ease-in-out infinite;
  }
  .palantir-spark.s1 { top: 14%; left: 52%; animation-delay: .2s; }
  .palantir-spark.s2 { top: 30%; right: 8%; animation-delay: .9s; }
  .palantir-spark.s3 { bottom: 20%; right: 18%; animation-delay: 1.5s; }
  .palantir-spark.s4 { bottom: 8%; left: 36%; animation-delay: .5s; }
  .palantir-spark.s5 { top: 42%; left: 6%; animation-delay: 1.1s; }
  .palantir-sigil {
    display: none;
  }
  .compare-empty-title {
    position: relative;
    z-index: 1;
    font-family: 'Aniron', 'Syne', sans-serif;
    font-size: 19px;
    font-weight: 700;
    line-height: 1.3;
    margin-bottom: 18px;
    letter-spacing: .2px;
  }
  .compare-empty-copy {
    position: relative;
    z-index: 1;
    margin: 0 auto 22px;
    max-width: 620px;
    font-size: 12px;
    line-height: 1.8;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }
  .compare-empty-suggestions {
    position: relative;
    z-index: 1;
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
    margin-bottom: 14px;
  }
  .compare-empty-suggestion {
    border: 1px solid rgba(34,211,238,.2);
    background: rgba(34,211,238,.08);
    color: color-mix(in srgb, var(--cyan) 88%, white);
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .3px;
    font-family: 'JetBrains Mono', monospace;
    cursor: pointer;
    transition: all .2s;
  }
  .compare-empty-suggestion:hover {
    transform: translateY(-1px);
    border-color: rgba(34,211,238,.38);
    background: rgba(34,211,238,.14);
    box-shadow: 0 0 18px rgba(34,211,238,.2);
  }
  .compare-empty-tip {
    position: relative;
    z-index: 1;
    font-size: 10px;
    letter-spacing: .8px;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }
  @keyframes palantirPulse {
    0%, 100% { opacity: .68; transform: translateX(-50%) scale(1); }
    50% { opacity: 1; transform: translateX(-50%) scale(1.08); }
  }
  @keyframes palantirFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-3px); }
  }
  @keyframes palantirBreath {
    0%, 100% {
      box-shadow:
        inset 0 0 42px rgba(156,186,228,.1),
        inset 0 -24px 34px rgba(0,0,0,.66),
        0 0 26px rgba(80, 150, 255, .18),
        0 0 44px rgba(255, 90, 30, .1),
        0 26px 42px rgba(0,0,0,.5);
    }
    50% {
      box-shadow:
        inset 0 0 50px rgba(156,186,228,.15),
        inset 0 -22px 32px rgba(0,0,0,.62),
        0 0 36px rgba(80, 150, 255, .28),
        0 0 58px rgba(255, 90, 30, .18),
        0 30px 46px rgba(0,0,0,.56);
    }
  }
  @keyframes palantirSwirl {
    0% { transform: rotate(0deg) scale(1) translate(-1px, -1px); opacity: .62; }
    50% { transform: rotate(180deg) scale(1.06) translate(1px, 1px); opacity: .88; }
    100% { transform: rotate(360deg) scale(1) translate(-1px, -1px); opacity: .62; }
  }
  @keyframes palantirLiquid {
    0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: .5; }
    30% { transform: translate(-1px, 1.6px) rotate(5deg) scale(1.03); opacity: .66; }
    65% { transform: translate(1.2px, -1.1px) rotate(-4deg) scale(.98); opacity: .6; }
  }
  @keyframes palantirMistDrift {
    0%, 100% { transform: translate(0, 0) scale(1); opacity: .5; }
    40% { transform: translate(-2px, 1.2px) scale(1.06); opacity: .74; }
    72% { transform: translate(1.8px, -1.4px) scale(1.02); opacity: .58; }
  }
  @keyframes palantirDepthRoll {
    0% { transform: rotate(0deg) scale(1); opacity: .64; }
    50% { transform: rotate(180deg) scale(1.04); opacity: .84; }
    100% { transform: rotate(360deg) scale(1); opacity: .64; }
  }
  @keyframes palantirVortexSpin {
    0% { transform: rotate(0deg) scale(1); opacity: .44; }
    50% { transform: rotate(180deg) scale(1.06); opacity: .62; }
    100% { transform: rotate(360deg) scale(1); opacity: .44; }
  }
  @keyframes palantirWispsShift {
    0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: .42; }
    30% { transform: translate(-2px, 1.4px) rotate(5deg) scale(1.08); opacity: .64; }
    65% { transform: translate(1.6px, -1.8px) rotate(-6deg) scale(1.04); opacity: .52; }
  }
  @keyframes palantirCurrentSweep {
    0%, 100% { transform: translateX(-1px) rotate(-8deg) scale(1); opacity: .42; }
    50% { transform: translateX(2.6px) rotate(10deg) scale(1.08); opacity: .72; }
  }
  @keyframes palantirSheenDrift {
    0%, 100% { transform: translate(0, 0) rotate(0deg); opacity: .54; }
    50% { transform: translate(1px, -1px) rotate(4deg); opacity: .72; }
  }
  @keyframes palantirCloudA {
    0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: .5; }
    35% { transform: translate(-3px, 2px) rotate(6deg) scale(1.12); opacity: .76; }
    70% { transform: translate(3px, -2px) rotate(-7deg) scale(1.06); opacity: .64; }
  }
  @keyframes palantirCloudB {
    0%, 100% { transform: translate(0, 0) rotate(0deg) scale(1); opacity: .36; }
    40% { transform: translate(2px, -2px) rotate(-8deg) scale(1.12); opacity: .62; }
    78% { transform: translate(-2px, 2px) rotate(6deg) scale(1.04); opacity: .48; }
  }
  @keyframes palantirStreakFlow {
    0%, 100% { transform: translateX(-2px) rotate(-5deg) scale(1); opacity: .34; }
    50% { transform: translateX(3px) rotate(8deg) scale(1.1); opacity: .68; }
  }
  @keyframes palantirSmoke {
    0%, 100% { transform: translate(0, 0) scale(1.02); opacity: .54; }
    35% { transform: translate(-2px, 1px) scale(1.08); opacity: .76; }
    70% { transform: translate(2px, -1px) scale(1.04); opacity: .62; }
  }
  @keyframes palantirArcOne {
    0% { transform: rotate(18deg); }
    100% { transform: rotate(378deg); }
  }
  @keyframes palantirArcTwo {
    0% { transform: rotate(46deg); }
    100% { transform: rotate(406deg); }
  }
  @keyframes palantirRingPulse {
    0%, 100% { opacity: .24; transform: scale(1); }
    50% { opacity: .38; transform: scale(1.015); }
  }
  @keyframes palantirOrbit {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
  @keyframes palantirSparkle {
    0%, 100% { opacity: .08; transform: scale(.7); }
    50% { opacity: .3; transform: scale(1); }
  }
  @keyframes palantirBlueCore {
    0%, 100% { transform: scale(1) translateX(-5%);  opacity: 0.85; }
    50%      { transform: scale(1.12) translateX(0%); opacity: 1; }
  }
@keyframes palantirRedCore {
  0%   { transform: scale(0.9) translateX(6%);  opacity: 0.6; }
  15%  { transform: scale(1.25) translateX(-4%); opacity: 1; }
  25%  { transform: scale(1.05) translateX(2%); opacity: 0.75; }
  40%  { transform: scale(1.35) translateX(-6%); opacity: 1; }
  55%  { transform: scale(0.85) translateX(8%); opacity: 0.5; }
  75%  { transform: scale(1.2) translateX(-2%); opacity: 0.95; }
  100% { transform: scale(0.9) translateX(6%);  opacity: 0.6; }
}

@keyframes palantirClash {
  0%, 100% { opacity: 0.3; transform: scale(0.9); }
  30%      { opacity: 0.5; transform: scale(1); }
  45%      { opacity: 1;   transform: scale(1.3); }
  55%      { opacity: 0.6; transform: scale(1.1); }
  75%      { opacity: 0.95; transform: scale(1.25); }
}

.palantir-lightning {
  position: absolute;
  inset: 10%;
  border-radius: 50%;
  overflow: hidden;
  mix-blend-mode: screen;
  pointer-events: none;
  filter: drop-shadow(0 0 2px rgba(220, 240, 255, 0.9))
          drop-shadow(0 0 4px rgba(140, 180, 255, 0.5));
}

.palantir-lightning-svg {
  width: 100%;
  height: 100%;
  display: block;
}

.palantir-lightning .bolt {
  opacity: 0;
}

.palantir-lightning .bolt-1 {
  animation: boltFlash1 5.3s steps(1, end) infinite;
}

.palantir-lightning .bolt-2 {
  animation: boltFlash2 6.7s steps(1, end) infinite;
}

.palantir-lightning .bolt-3 {
  animation: boltFlash3 4.9s steps(1, end) infinite;
}

/* CRACKS — subtila sprickor i glaset som glimtar till */
.palantir-cracks {
  position: absolute;
  inset: 8%;
  border-radius: 50%;
  background:
    linear-gradient(68deg,
      transparent 48%,
      rgba(255, 200, 150, 0.35) 49.5%,
      transparent 51%),
    linear-gradient(125deg,
      transparent 42%,
      rgba(200, 220, 255, 0.3) 43.2%,
      transparent 44.5%),
    linear-gradient(-50deg,
      transparent 56%,
      rgba(255, 180, 120, 0.28) 57.5%,
      transparent 59%);
  filter: blur(0.6px);
  mix-blend-mode: screen;
  opacity: 0;
  animation: palantirCracks 7.1s ease-in-out infinite;
  pointer-events: none;
}

  @keyframes palantirSigilFlicker {
    0%, 100% { opacity: .86; filter: brightness(1); }
    45% { opacity: .94; filter: brightness(1.08); }
    52% { opacity: .8; filter: brightness(.95); }
    60% { opacity: .9; filter: brightness(1.06); }
  }

@keyframes boltFlash1 {
  0%, 80%, 100% { opacity: 0; }
  81%           { opacity: 1; }
  82%           { opacity: 0; }
  83%           { opacity: 0.85; }
  84%           { opacity: 0; }
}

@keyframes boltFlash2 {
  0%, 42%, 100% { opacity: 0; }
  43%           { opacity: 1; }
  44%           { opacity: 0; }
  46%           { opacity: 0.7; }
  47%           { opacity: 0; }
}

@keyframes boltFlash3 {
  0%, 65%, 100% { opacity: 0; }
  66%           { opacity: 0.9; }
  67%           { opacity: 0; }
  70%           { opacity: 0.6; }
  71%           { opacity: 0; }
}

  @media (max-width: 760px) {
    .compare-empty {
      padding: 52px 14px 42px;
    }
    .compare-empty-title {
      font-size: 16px;
    }
    .compare-empty-suggestions {
      gap: 8px;
    }
    .compare-empty-suggestion {
      width: 100%;
      max-width: 340px;
      text-align: center;
    }
    .compare-empty-core {
      width: 126px;
      height: 126px;
      margin-bottom: 16px;
    }
    .palantir-sigil {
      font-size: 22px;
    }
  }

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
  .overview-card {
    position: relative;
    overflow: hidden;
    padding: 22px;
    isolation: isolate;
    border-radius: 16px;
    background:
      linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0)),
      linear-gradient(135deg, rgba(17,24,39,.96), rgba(13,18,32,.96));
    box-shadow:
      inset 0 1px 0 rgba(255,255,255,.04),
      0 18px 40px rgba(0,0,0,.22),
      0 0 0 1px rgba(255,255,255,.02),
      0 0 32px var(--overview-glow, rgba(34,211,238,.12));
    animation: overviewCardPulse 6s ease-in-out infinite;
    animation-delay: var(--overview-delay, 0s);
  }
  .overview-card::before {
    content: '';
    position: absolute;
    inset: 0 0 auto 0;
    height: 3px;
    background: var(--overview-accent, var(--cyan));
    opacity: .8;
  }
  .overview-card::after {
    content: '';
    position: absolute;
    top: -60px;
    right: -50px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--overview-glow, rgba(34,211,238,.16)) 0%, rgba(34,211,238,0) 70%);
    filter: blur(8px);
    opacity: .9;
    pointer-events: none;
    z-index: -1;
    animation: overviewGlowFloat 6s ease-in-out infinite;
    animation-delay: var(--overview-delay, 0s);
  }
  @keyframes overviewCardPulse {
    0%, 100% {
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,.04),
        0 18px 40px rgba(0,0,0,.22),
        0 0 0 1px rgba(255,255,255,.02),
        0 0 24px var(--overview-glow, rgba(34,211,238,.1));
    }
    50% {
      box-shadow:
        inset 0 1px 0 rgba(255,255,255,.05),
        0 22px 52px rgba(0,0,0,.26),
        0 0 0 1px rgba(255,255,255,.04),
        0 0 44px var(--overview-glow, rgba(34,211,238,.18));
    }
  }
  @keyframes overviewGlowFloat {
    0%, 100% {
      transform: translate3d(0, 0, 0) scale(1);
      opacity: .72;
    }
    50% {
      transform: translate3d(-8px, 10px, 0) scale(1.08);
      opacity: 1;
    }
  }
  .overview-card-header {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 18px;
  }
  .overview-model-tag {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 7px 10px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid var(--overview-accent, var(--cyan));
    color: var(--overview-accent, var(--cyan));
    background: color-mix(in srgb, var(--overview-accent, var(--cyan)) 10%, transparent);
  }
  .overview-model-type {
    font-size: 10px;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 1.2px;
  }
  .overview-badges {
    position: relative;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 14px;
  }
  .overview-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 5px 9px;
    border-radius: 999px;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
    color: var(--overview-accent, var(--cyan));
    background: color-mix(in srgb, var(--overview-accent, var(--cyan)) 12%, transparent);
    border: 1px solid color-mix(in srgb, var(--overview-accent, var(--cyan)) 30%, transparent);
  }
  .overview-hero {
    position: relative;
    display: grid;
    grid-template-columns: 1fr auto;
    align-items: end;
    gap: 14px;
    margin-bottom: 16px;
  }
  .overview-hero-label {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 8px;
  }
  .overview-hero-value {
    font-size: 56px;
    font-weight: 800;
    line-height: .95;
    font-family: var(--num-font);
    font-variant-numeric: tabular-nums lining-nums;
    letter-spacing: -0.01em;
    color: var(--overview-accent, var(--cyan));
    text-shadow: 0 0 28px var(--overview-glow, rgba(34,211,238,.16));
  }
  .overview-watermark {
    font-size: 76px;
    line-height: 1;
    font-weight: 800;
    color: color-mix(in srgb, var(--overview-accent, var(--cyan)) 35%, transparent);
    opacity: .28;
    transform: translateY(4px);
    text-shadow: 0 0 28px var(--overview-glow, rgba(34,211,238,.16));
    user-select: none;
  }
  .overview-progress {
    position: relative;
    height: 8px;
    border-radius: 999px;
    overflow: hidden;
    background: rgba(255,255,255,.05);
    border: 1px solid rgba(255,255,255,.05);
    margin-bottom: 18px;
  }
  .overview-progress-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--overview-accent, var(--cyan)), color-mix(in srgb, var(--overview-accent, var(--cyan)) 55%, white));
  }
  .overview-stats {
    position: relative;
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
  }
  .overview-stat {
    min-width: 0;
    padding: 12px 12px 10px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.07);
    background: rgba(255,255,255,.025);
  }
  .overview-stat .metric-label {
    margin-bottom: 8px;
  }
  .overview-stat-value {
    font-size: 22px;
    font-weight: 800;
    line-height: 1;
    font-family: var(--num-font);
    font-variant-numeric: tabular-nums lining-nums;
    letter-spacing: -0.005em;
    color: var(--overview-accent, var(--cyan));
    word-break: break-word;
  }
  .overview-banner {
    position: relative;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding: 32px 24px;
    border-radius: 16px;
    border: 1px solid var(--border2);
    background:
      linear-gradient(90deg, rgba(34,211,238,.08), rgba(167,139,250,.08) 52%, rgba(52,211,153,.08)),
      linear-gradient(180deg, rgba(255,255,255,.02), rgba(255,255,255,0));
    box-shadow: inset 0 1px 0 rgba(255,255,255,.03), 0 16px 34px rgba(0,0,0,.16);
  }
  .overview-banner::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,.06) 48%, transparent 100%);
    transform: translateX(-120%);
    animation: overviewBannerSweep 9s linear infinite;
    pointer-events: none;
  }
  @keyframes overviewBannerSweep {
    to { transform: translateX(120%); }
  }
  .overview-banner-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: grid;
    place-items: center;
    font-size: 18px;
    color: var(--text);
    background: rgba(34,211,238,.12);
    border: 1px solid rgba(34,211,238,.2);
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
  }
  .overview-banner-header {
    display: flex;
    align-items: flex-start;
    gap: 16px;
  }
  .overview-banner-copy {
    position: relative;
    z-index: 1;
    flex: 1;
    font-size: 14px;
    line-height: 1.8;
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
  }
  .overview-banner-copy strong {
    color: var(--cyan);
    font-weight: 700;
  }
  .overview-insights {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .overview-insight-badge {
    position: relative;
    padding: 14px 12px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.06);
    background: rgba(255,255,255,.02);
    display: flex;
    flex-direction: column;
    gap: 6px;
    align-items: center;
    text-align: center;
  }
  .overview-insight-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: var(--muted);
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace;
  }
  .overview-insight-value {
    font-size: 18px;
    font-weight: 700;
    color: var(--cyan);
    font-family: var(--num-font);
    font-variant-numeric: tabular-nums lining-nums;
    letter-spacing: -0.005em;
  }
  .overview-insight-value.violet {
    color: var(--violet);
  }
  .overview-insight-value.emerald {
    color: var(--emerald);
  }
  .metric-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.5px;
    text-transform: uppercase; color: var(--muted);
    font-family: 'JetBrains Mono', monospace;
  }
  .metric-value {
    font-size: 28px;
    font-weight: 800;
    line-height: 1;
    font-family: var(--num-font);
    font-variant-numeric: tabular-nums lining-nums;
    letter-spacing: -0.01em;
  }
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
const pct = (v) => (v != null ? `${(v * 100).toFixed(1)}%` : "—");
const ms = (v) => (v != null ? `${v.toFixed(0)}ms` : "—");
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
      style={{
        background: color + "18",
        border: `1px solid ${color}40`,
        color,
      }}
    >
      <span>{meta.icon ?? "◆"}</span>
      <span>{meta.short ?? model}</span>
    </div>
  );
}

function AnswerCard({ model, data, loading }) {
  const color = modelColor(model);
  const meta = MODEL_META[model] ?? {};
  const info = {
    "text-embedding-3-small": "OpenAI",
    "all-MiniLM-L6-v2": "HuggingFace",
    "multilingual-e5-base": "HuggingFace",
  };

  return (
    <div
      className={`answer-card ${data ? "loaded" : ""}`}
      style={{ borderColor: data ? color + "50" : undefined }}
    >
      <div className="model-header">
        <div className="model-name" style={{ color }}>
          {meta.icon} {meta.short ?? model}
        </div>
        <span
          className="model-type-tag"
          style={{
            background: color + "15",
            color,
            border: `1px solid ${color}30`,
          }}
        >
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
                🎯{" "}
                <span className="val">
                  {(data.top_score * 100).toFixed(0)}%
                </span>
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]); // [{question, results, timestamp}]
  const [loadingQuestion, setLoadingQuestion] = useState(null);
  const bottomRef = useRef(null);

  const displayModels =
    indexedModels.length > 0 ? indexedModels : Object.keys(MODEL_META);

  const submit = useCallback(
    async (q) => {
      const text = (q || question).trim();
      if (!text) return;
      setLoading(true);
      setLoadingQuestion(text);
      setError(null);
      setQuestion("");

      const timestamp = new Date().toLocaleTimeString("sv-SE", {
        hour: "2-digit",
        minute: "2-digit",
      });

      // Lägg till frågan DIREKT med loading-state så den syns direkt
      setHistory((prev) => [
        { question: text, results: null, loading: true, timestamp },
        ...prev,
      ]);

      try {
        const data = await apiFetch("/compare", {
          method: "POST",
          body: JSON.stringify({ question: text, k: 4, threshold: 0.35 }),
        });
        const map = {};
        for (const r of data.results) map[r.model] = r;

        // Ersätt loading-entryn med riktiga svar
        setHistory((prev) =>
          prev.map((entry) =>
            entry.loading && entry.question === text
              ? { question: text, results: map, loading: false, timestamp }
              : entry,
          ),
        );
      } catch (e) {
        setError(e.message);
        // Ta bort loading-entryn om det gick fel
        setHistory((prev) =>
          prev.filter((entry) => !(entry.loading && entry.question === text)),
        );
      } finally {
        setLoading(false);
        setLoadingQuestion(null);
      }
    },
    [question],
  );

  // No auto-scroll needed — newest entry appears at top

  const onKey = (e) => {
    if (e.key === "Enter") submit();
  };

  return (
    <div className="stack">
      {/* Demo pills — only show once there's history */}
      {history.length > 0 && (
        <div className="demo-pills">
          {DEMO_QUESTIONS.map((q) => (
            <div
              key={q}
              className="demo-pill"
              onClick={() => {
                submit(q);
              }}
            >
              {q}
            </div>
          ))}
        </div>
      )}

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
        <button
          className="btn btn-primary"
          onClick={() => submit()}
          disabled={loading}
        >
          {loading ? <div className="spinner" /> : "▶"}
          {loading ? "Querying…" : "Compare"}
        </button>
        {history.length > 0 && (
          <button
            className="btn btn-ghost"
            onClick={() => setHistory([])}
            title="Clear history"
          >
            ✕ Clear
          </button>
        )}
      </div>

      {error && (
        <div
          style={{
            color: "var(--red)",
            fontSize: 12,
            fontFamily: "'JetBrains Mono', monospace",
            background: "rgba(248,113,113,.08)",
            padding: "12px 16px",
            borderRadius: 8,
            border: "1px solid rgba(248,113,113,.2)",
          }}
        >
          ⚠ {error}
        </div>
      )}

      {/* Empty state */}
      {history.length === 0 && !loading && (
        <div className="compare-empty">
          <div className="compare-empty-core">
            <div className="palantir-liquid" />
            <div className="palantir-mist" />
            <div className="palantir-depth" />
            <div className="palantir-vortex" />
            <div className="palantir-wisps" />
            <div className="palantir-current" />
            <div className="palantir-cloud-a" />
            <div className="palantir-cloud-b" />
            <div className="palantir-streaks" />
            <div className="palantir-smoke" />
            <div className="palantir-blue-core" />
            <div className="palantir-red-core" />
            <div className="palantir-clash" />
            <div className="palantir-lightning">
              <svg
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                className="palantir-lightning-svg"
              >
                {/* Bolt 1 — går ner-vänster */}
                <polyline
                  className="bolt bolt-1"
                  points="52,20 48,32 54,38 46,50 52,58 44,70 50,78"
                  fill="none"
                  stroke="rgba(220, 240, 255, 0.95)"
                  strokeWidth="0.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Bolt 1 branch */}
                <polyline
                  className="bolt bolt-1"
                  points="54,38 62,42 58,48"
                  fill="none"
                  stroke="rgba(220, 240, 255, 0.8)"
                  strokeWidth="0.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Bolt 2 — går diagonalt över mitten */}
                <polyline
                  className="bolt bolt-2"
                  points="28,42 38,46 32,54 42,58 36,66 46,70"
                  fill="none"
                  stroke="rgba(200, 225, 255, 0.95)"
                  strokeWidth="0.55"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {/* Bolt 2 branch */}
                <polyline
                  className="bolt bolt-2"
                  points="32,54 24,58 28,64"
                  fill="none"
                  stroke="rgba(200, 225, 255, 0.75)"
                  strokeWidth="0.35"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />

                {/* Bolt 3 — orange/varm, mindre */}
                <polyline
                  className="bolt bolt-3"
                  points="68,30 72,40 66,46 74,54 70,62"
                  fill="none"
                  stroke="rgba(255, 200, 140, 0.9)"
                  strokeWidth="0.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </div>{" "}
            <div className="palantir-cracks" />
            <div className="palantir-glass-sheen" />
            <div className="palantir-arc a" />
            <div className="palantir-arc b" />
          </div>
          <div className="compare-empty-title">The Palantír Is Waiting</div>
          <p className="compare-empty-copy">
            Ask a question to compare all three models side by side. Your
            conversation history appears here once you begin.
          </p>
          <div className="compare-empty-suggestions">
            {[
              "Who is Morgoth and why is he feared?",
              "Compare Sauron vs Saruman as villains",
              "What happened to the Two Trees of Valinor?",
            ].map((suggestion) => (
              <button
                key={suggestion}
                className="compare-empty-suggestion"
                onClick={() => submit(suggestion)}
              >
                {suggestion}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* History */}
      <div className="stack">
        {[...history].map((entry, idx) => (
          <div
            key={idx}
            style={{
              borderRadius: 12,
              overflow: "hidden",
              border: "1px solid var(--border)",
              background: "var(--bg2)",
            }}
          >
            {/* Question header */}
            <div
              style={{
                padding: "12px 20px",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                background: "rgba(255,255,255,.02)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <span
                  style={{
                    fontSize: 10,
                    fontFamily: "'JetBrains Mono',monospace",
                    color: "var(--cyan)",
                    fontWeight: 700,
                    letterSpacing: 1,
                  }}
                >
                  Q{idx + 1}
                </span>
                <span style={{ fontSize: 13, fontWeight: 600 }}>
                  {entry.question}
                </span>
              </div>
              <span
                style={{
                  fontSize: 10,
                  fontFamily: "'JetBrains Mono',monospace",
                  color: "var(--muted)",
                }}
              >
                {entry.timestamp}
              </span>
            </div>
            {/* Answers grid */}
            <div className="compare-grid" style={{ padding: 16 }}>
              {displayModels.map((m) => (
                <AnswerCard
                  key={m}
                  model={m}
                  data={entry.results?.[m]}
                  loading={entry.loading}
                />
              ))}
            </div>
          </div>
        ))}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}

// ─── View: Benchmark ───────────────────────────────────────────────────────
function BenchmarkView({ evalData }) {
  const [activeTab, setActiveTab] = useState("overview");
  const [questionFilter, setQuestionFilter] = useState("all");

  if (!evalData) {
    return (
      <div style={{ textAlign: "center", padding: 60, color: "var(--muted)" }}>
        <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
        <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>
          No evaluation data found.
          <br />
          Run <code style={{ color: "var(--cyan)" }}>evaluate.py</code> first.
        </div>
      </div>
    );
  }

  const models = evalData.models ?? [];
  const bestHitRate = Math.max(
    ...models.map((m) => m.metrics?.source_hit_rate ?? 0),
  );
  const bestKeywordRecall = Math.max(
    ...models.map((m) => m.metrics?.avg_keyword_recall ?? 0),
  );
  const fastestResponse = Math.min(
    ...models.map((m) => m.metrics?.avg_retrieval_time_ms ?? Infinity),
  );
  const hitRateLeaders = models.filter(
    (m) => m.metrics?.source_hit_rate === bestHitRate,
  );
  const recallLeader = models.find(
    (m) => m.metrics?.avg_keyword_recall === bestKeywordRecall,
  );
  const speedLeader = models.find(
    (m) => m.metrics?.avg_retrieval_time_ms === fastestResponse,
  );
  const hitRateLeaderText = hitRateLeaders
    .map((m) => modelShort(m.model_name))
    .join(" and ");

  // Radar chart data
  const radarData = [
    {
      metric: "Hit Rate",
      ...Object.fromEntries(
        models.map((m) => [
          modelShort(m.model_name),
          m.metrics.source_hit_rate * 100,
        ]),
      ),
    },
    {
      metric: "Keyword Recall",
      ...Object.fromEntries(
        models.map((m) => [
          modelShort(m.model_name),
          m.metrics.avg_keyword_recall * 100,
        ]),
      ),
    },
    {
      metric: "Relevance Score",
      ...Object.fromEntries(
        models.map((m) => [
          modelShort(m.model_name),
          m.metrics.avg_top_score * 100,
        ]),
      ),
    },
    {
      metric: "Speed (inv)",
      ...Object.fromEntries(
        models.map((m) => [
          modelShort(m.model_name),
          Math.max(0, 100 - m.metrics.avg_retrieval_time_ms / 20),
        ]),
      ),
    },
  ];

  // Speed comparison
  const speedData = models.map((m) => ({
    name: modelShort(m.model_name),
    time: m.metrics.avg_retrieval_time_ms,
    color: modelColor(m.model_name),
  }));

  // Per-question table
  const allQuestions =
    models[0]?.results?.map((r, i) => ({
      q: r.question,
      difficulty: r.difficulty,
      ...Object.fromEntries(
        models.map((m) => [
          m.model_name,
          {
            hit: m.results[i]?.source_hit,
            score: m.results[i]?.top_score,
            time: m.results[i]?.retrieval_time_ms,
          },
        ]),
      ),
    })) ?? [];

  // Cost data
  // OpenAI text-embedding-3-small: $0.020 per 1M tokens
  // Avg query ~15 tokens, so cost per query = 15 * 0.020 / 1_000_000
  const OPENAI_COST_PER_TOKEN = 0.2 / 1_000_000;
  const AVG_TOKENS_PER_QUERY = 15;
  const OPENAI_COST_PER_QUERY = OPENAI_COST_PER_TOKEN * AVG_TOKENS_PER_QUERY;

  const costMeta = {
    "text-embedding-3-small": {
      costPerQuery: OPENAI_COST_PER_QUERY,
      type: "paid",
      note: "0.20kr / 1M tokens (OpenAI API)",
    },
    "all-MiniLM-L6-v2": {
      costPerQuery: 0,
      type: "free",
      note: "Free — runs locally on CPU",
    },
    "multilingual-e5-base": {
      costPerQuery: 0,
      type: "free",
      note: "Free — runs locally on CPU",
    },
  };

  // Break-even chart: cumulative cost at N queries
  const breakEvenData = [
    1, 100, 1000, 10000, 50000, 100000, 500000, 1000000,
  ].map((n) => ({
    queries: n >= 1000000 ? "1M" : n >= 1000 ? `${n / 1000}k` : `${n}`,
    queriesRaw: n,
    OpenAI: parseFloat((n * OPENAI_COST_PER_QUERY).toFixed(4)),
    "Open-Source": 0,
  }));

  return (
    <div className="stack">
      <div className="tabs">
        {["overview", "speed", "cost", "questions"].map((t) => (
          <div
            key={t}
            className={`tab ${activeTab === t ? "active" : ""}`}
            onClick={() => setActiveTab(t)}
          >
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
              const badges = [
                m.metrics.source_hit_rate === bestHitRate
                  ? "Best hit rate"
                  : null,
                m.metrics.avg_keyword_recall === bestKeywordRecall
                  ? "Best recall"
                  : null,
                m.metrics.avg_retrieval_time_ms === fastestResponse
                  ? "Fastest"
                  : null,
              ].filter(Boolean);
              const metricCards = [
                {
                  label: "Keyword Recall",
                  val: pct(m.metrics.avg_keyword_recall),
                },
                {
                  label: "Avg Top Score",
                  val: pct(m.metrics.avg_top_score),
                },
                {
                  label: "Avg Response",
                  val: ms(m.metrics.avg_retrieval_time_ms),
                },
              ];

              return (
                <div
                  key={m.model_name}
                  className="metric-card overview-card"
                  style={{
                    borderColor: color + "40",
                    "--overview-accent": color,
                    "--overview-glow": color + "22",
                    "--overview-delay": `${models.indexOf(m) * 1.1}s`,
                  }}
                >
                  <div className="overview-card-header">
                    <div className="overview-model-tag">
                      <span>{MODEL_META[m.model_name]?.icon}</span>
                      <span>{modelShort(m.model_name)}</span>
                    </div>
                    <div className="overview-model-type">
                      {m.model_type === "openai" ? "OpenAI API" : "Local model"}
                    </div>
                  </div>

                  {badges.length > 0 && (
                    <div className="overview-badges">
                      {badges.map((badge) => (
                        <div key={badge} className="overview-badge">
                          <span>+</span>
                          <span>{badge}</span>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="overview-hero">
                    <div>
                      <div className="overview-hero-label">Source Hit Rate</div>
                      <div className="overview-hero-value">
                        {pct(m.metrics.source_hit_rate)}
                      </div>
                    </div>
                    <div className="overview-watermark">
                      {MODEL_META[m.model_name]?.icon}
                    </div>
                  </div>

                  <div className="overview-progress">
                    <div
                      className="overview-progress-fill"
                      style={{ width: `${m.metrics.source_hit_rate * 100}%` }}
                    />
                  </div>

                  <div className="overview-stats">
                    {metricCards.map(({ label, val }) => (
                      <div key={label} className="overview-stat">
                        <div className="metric-label">{label}</div>
                        <div className="overview-stat-value">{val}</div>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>

          <div className="overview-banner">
            <div className="overview-banner-header">
              <div className="overview-banner-icon">///</div>
              <div className="overview-banner-copy">
                <strong>{hitRateLeaderText}</strong> lead on hit rate,{" "}
                <strong>
                  {recallLeader
                    ? modelShort(recallLeader.model_name)
                    : "OpenAI"}
                </strong>{" "}
                leads keyword recall, and{" "}
                <strong>
                  {speedLeader ? modelShort(speedLeader.model_name) : "MiniLM"}
                </strong>{" "}
                is fastest.
              </div>
            </div>
            <div className="overview-insights">
              <div className="overview-insight-badge">
                <div className="overview-insight-label">Best Hit Rate</div>
                <div className="overview-insight-value">
                  {pct(
                    models.find(
                      (m) => m.metrics.source_hit_rate === bestHitRate,
                    )?.metrics.source_hit_rate || 0,
                  )}
                </div>
              </div>
              <div className="overview-insight-badge">
                <div className="overview-insight-label">Best Recall</div>
                <div className="overview-insight-value violet">
                  {pct(
                    models.find(
                      (m) => m.metrics.avg_keyword_recall === bestKeywordRecall,
                    )?.metrics.avg_keyword_recall || 0,
                  )}
                </div>
              </div>
              <div className="overview-insight-badge">
                <div className="overview-insight-label">Fastest Response</div>
                <div className="overview-insight-value emerald">
                  {ms(
                    models.find(
                      (m) =>
                        m.metrics.avg_retrieval_time_ms === fastestResponse,
                    )?.metrics.avg_retrieval_time_ms || 0,
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === "speed" && (
        <div className="stack">
          <div className="card">
            <div className="card-title">Average Response Time (ms)</div>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={speedData} layout="vertical">
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="var(--border)"
                  horizontal={false}
                />
                <XAxis
                  type="number"
                  tick={{ fill: "var(--muted)", fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v) => `${v}ms`}
                />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{
                    fill: "var(--muted)",
                    fontSize: 11,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                  axisLine={false}
                  tickLine={false}
                  width={70}
                />
                <Tooltip
                  formatter={(v) => `${v.toFixed(0)}ms`}
                  contentStyle={{
                    background: "var(--bg2)",
                    border: "1px solid var(--border2)",
                    borderRadius: 8,
                    fontSize: 11,
                  }}
                  cursor={{ fill: "transparent" }}
                />
                <Bar dataKey="time" radius={[0, 4, 4, 0]} maxBarSize={32}>
                  {speedData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Speed cards */}
          <div className="grid-3">
            {[...models]
              .sort(
                (a, b) =>
                  a.metrics.avg_retrieval_time_ms -
                  b.metrics.avg_retrieval_time_ms,
              )
              .map((m, i) => {
                const color = modelColor(m.model_name);
                return (
                  <div
                    key={m.model_name}
                    className="metric-card"
                    style={{ borderColor: color + "40" }}
                  >
                    <div className="model-name" style={{ color }}>
                      {modelShort(m.model_name)}
                    </div>
                    <div
                      className="metric-value"
                      style={{ color, fontSize: 32, marginTop: 8 }}
                    >
                      {ms(m.metrics.avg_retrieval_time_ms)}
                    </div>
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
              const meta = costMeta[m.model_name] ?? {
                costPerQuery: 0,
                type: "free",
                note: "",
              };
              const per1k = (meta.costPerQuery * 1000).toFixed(4);
              const per1M = (meta.costPerQuery * 1_000_000).toFixed(2);
              return (
                <div
                  key={m.model_name}
                  className="metric-card"
                  style={{ borderColor: color + "40" }}
                >
                  <div
                    className="model-name"
                    style={{ color, marginBottom: 12 }}
                  >
                    {MODEL_META[m.model_name]?.icon} {modelShort(m.model_name)}
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div className="metric-label">Cost per query</div>
                    <div
                      className="metric-value"
                      style={{ color, fontSize: 28 }}
                    >
                      {meta.type === "free"
                        ? "0 kr"
                        : `${meta.costPerQuery.toFixed(6)} kr`}
                    </div>
                  </div>
                  <div
                    style={{ display: "flex", flexDirection: "column", gap: 6 }}
                  >
                    <div
                      style={{
                        marginTop: 8,
                        fontSize: 10,
                        fontFamily: "'JetBrains Mono',monospace",
                        padding: "4px 8px",
                        borderRadius: 6,
                        border: `1px solid ${color}30`,
                        background: color + "10",
                        color,
                      }}
                    >
                      {meta.note}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Break-even chart */}
          <div className="card">
            <div className="card-title">
              Cumulative API Cost — OpenAI vs Open-Source
            </div>
            <p
              style={{
                fontSize: 11,
                color: "var(--muted)",
                marginBottom: 20,
                fontFamily: "'JetBrains Mono',monospace",
                lineHeight: 1.6,
              }}
            >
              Open-source models run locally at zero API cost. The more queries
              you run, the larger the saving.
            </p>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart
                data={breakEvenData}
                margin={{ top: 10, right: 30, bottom: 10, left: 60 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="var(--border)"
                  vertical={false}
                />
                <XAxis
                  dataKey="queries"
                  tick={{
                    fill: "var(--muted)",
                    fontSize: 10,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  tickFormatter={(v) => `${v} kr`}
                  tick={{
                    fill: "var(--muted)",
                    fontSize: 10,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                  axisLine={false}
                  tickLine={false}
                  width={55}
                />
                <Tooltip
                  formatter={(v, name) => [`${v} kr`, name]}
                  contentStyle={{
                    background: "var(--bg2)",
                    border: "1px solid var(--border2)",
                    borderRadius: 8,
                    fontSize: 11,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                />
                <Legend
                  wrapperStyle={{
                    fontSize: 11,
                    fontFamily: "'JetBrains Mono',monospace",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="OpenAI"
                  stroke="var(--cyan)"
                  strokeWidth={2}
                  dot={{ fill: "var(--cyan)", r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="Open-Source"
                  stroke="var(--emerald)"
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  dot={{ fill: "var(--emerald)", r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Summary insight */}
        </div>
      )}

      {activeTab === "questions" &&
        (() => {
          // Classify every row by model agreement
          const classified = allQuestions.map((row) => {
            const hits = models.map((m) => (row[m.model_name]?.hit ? 1 : 0));
            const hitCount = hits.reduce((a, b) => a + b, 0);
            let agreement;
            if (hitCount === models.length) agreement = "all-hit";
            else if (hitCount === 0) agreement = "all-miss";
            else if (hitCount === models.length - 1) agreement = "one-miss";
            else agreement = "one-hit";
            return { ...row, agreement, hitCount };
          });

          const filters = [
            { id: "all", label: "All questions", count: classified.length },
            {
              id: "disagree",
              label: "Disagreements",
              count: classified.filter(
                (r) => r.agreement === "one-miss" || r.agreement === "one-hit",
              ).length,
            },
            {
              id: "all-miss",
              label: "All missed",
              count: classified.filter((r) => r.agreement === "all-miss")
                .length,
            },
            {
              id: "all-hit",
              label: "All hit",
              count: classified.filter((r) => r.agreement === "all-hit").length,
            },
          ];

          const visible = classified.filter((r) => {
            if (questionFilter === "all") return true;
            if (questionFilter === "disagree")
              return r.agreement === "one-miss" || r.agreement === "one-hit";
            return r.agreement === questionFilter;
          });

          const agreementMeta = {
            "all-hit": {
              label: "All agree",
              color: "var(--emerald)",
              bg: "rgba(52,211,153,.1)",
              border: "rgba(52,211,153,.3)",
            },
            "all-miss": {
              label: "All missed",
              color: "var(--red)",
              bg: "rgba(248,113,113,.1)",
              border: "rgba(248,113,113,.3)",
            },
            "one-miss": {
              label: "Split 2-1",
              color: "var(--amber)",
              bg: "rgba(251,191,36,.1)",
              border: "rgba(251,191,36,.3)",
            },
            "one-hit": {
              label: "Split 1-2",
              color: "var(--amber)",
              bg: "rgba(251,191,36,.1)",
              border: "rgba(251,191,36,.3)",
            },
          };

          return (
            <div className="card">
              <div className="card-title">
                <span>
                  Per-Question Results ({visible.length} of {classified.length})
                </span>
              </div>

              {/* Filter pills */}
              <div
                style={{
                  display: "flex",
                  gap: 8,
                  flexWrap: "wrap",
                  marginBottom: 20,
                }}
              >
                {filters.map((f) => {
                  const isActive = questionFilter === f.id;
                  return (
                    <div
                      key={f.id}
                      onClick={() => setQuestionFilter(f.id)}
                      style={{
                        cursor: "pointer",
                        padding: "6px 12px",
                        borderRadius: 8,
                        fontSize: 10,
                        fontWeight: 700,
                        letterSpacing: 1,
                        textTransform: "uppercase",
                        fontFamily: "'JetBrains Mono', monospace",
                        border: `1px solid ${isActive ? "var(--cyan)" : "var(--border2)"}`,
                        background: isActive
                          ? "rgba(34,211,238,.08)"
                          : "var(--bg3)",
                        color: isActive ? "var(--cyan)" : "var(--muted)",
                        transition: "all .2s",
                        display: "inline-flex",
                        alignItems: "center",
                        gap: 6,
                      }}
                    >
                      {f.label}
                      <span
                        style={{
                          background: isActive
                            ? "var(--cyan)"
                            : "var(--border2)",
                          color: isActive ? "#000" : "var(--muted)",
                          borderRadius: 10,
                          padding: "1px 6px",
                          fontSize: 9,
                        }}
                      >
                        {f.count}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th style={{ width: 34 }}></th>
                      <th>Question</th>
                      <th>Difficulty</th>
                      <th>Agreement</th>
                      {models.map((m) => (
                        <th
                          key={m.model_name}
                          style={{ color: modelColor(m.model_name) }}
                        >
                          {modelShort(m.model_name)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {visible.map((row, i) => {
                      const am = agreementMeta[row.agreement];
                      const isInteresting =
                        row.agreement === "one-miss" ||
                        row.agreement === "one-hit" ||
                        row.agreement === "all-miss";

                      return (
                        <tr
                          key={i}
                          style={{
                            borderLeft: isInteresting
                              ? `3px solid ${am.color}`
                              : "3px solid transparent",
                          }}
                        >
                          <td style={{ paddingLeft: 12 }}>
                            <span
                              style={{
                                display: "inline-block",
                                width: 6,
                                height: 6,
                                borderRadius: "50%",
                                background: am.color,
                                opacity: isInteresting ? 1 : 0.35,
                              }}
                            />
                          </td>
                          <td
                            style={{
                              maxWidth: 280,
                              color: "var(--text)",
                              fontSize: 11,
                            }}
                          >
                            {row.q}
                          </td>
                          <td>
                            <span
                              style={{
                                fontSize: 9,
                                fontWeight: 700,
                                letterSpacing: 1,
                                padding: "2px 6px",
                                borderRadius: 4,
                                textTransform: "uppercase",
                                background:
                                  row.difficulty === "easy"
                                    ? "rgba(52,211,153,.1)"
                                    : row.difficulty === "medium"
                                      ? "rgba(251,191,36,.1)"
                                      : "rgba(248,113,113,.1)",
                                color:
                                  row.difficulty === "easy"
                                    ? "var(--emerald)"
                                    : row.difficulty === "medium"
                                      ? "var(--amber)"
                                      : "var(--red)",
                              }}
                            >
                              {row.difficulty}
                            </span>
                          </td>
                          <td>
                            <span
                              style={{
                                fontSize: 9,
                                fontWeight: 700,
                                letterSpacing: 1,
                                padding: "2px 8px",
                                borderRadius: 10,
                                textTransform: "uppercase",
                                background: am.bg,
                                color: am.color,
                                border: `1px solid ${am.border}`,
                              }}
                            >
                              {am.label}
                            </span>
                          </td>
                          {models.map((m) => {
                            const d = row[m.model_name];
                            return (
                              <td key={m.model_name}>
                                <span
                                  style={{
                                    color: d?.hit
                                      ? "var(--emerald)"
                                      : "var(--red)",
                                    fontSize: 14,
                                  }}
                                >
                                  {d?.hit ? "✓" : "✗"}
                                </span>
                                <span
                                  style={{
                                    color: "var(--muted)",
                                    marginLeft: 6,
                                    fontSize: 10,
                                  }}
                                >
                                  {d?.score != null
                                    ? (d.score * 100).toFixed(0) + "%"
                                    : ""}
                                </span>
                              </td>
                            );
                          })}
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {visible.length === 0 && (
                <div
                  style={{
                    textAlign: "center",
                    padding: 40,
                    color: "var(--muted)",
                    fontSize: 11,
                    fontFamily: "'JetBrains Mono', monospace",
                  }}
                >
                  No questions match this filter.
                </div>
              )}
            </div>
          );
        })()}
    </div>
  );
}

// ─── View: Embedding Explorer ──────────────────────────────────────────────
function ExplorerView({ evalData, embeddingCoords }) {
  const [selectedModel, setSelectedModel] = useState(null);
  const models = evalData?.models ?? [];

  useEffect(() => {
    if (!selectedModel && embeddingCoords) {
      const first = Object.keys(embeddingCoords)[0];
      if (first) setSelectedModel(first);
    }
  }, [embeddingCoords, selectedModel]);

  if (!evalData) {
    return (
      <div style={{ textAlign: "center", padding: 60, color: "var(--muted)" }}>
        <div style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: 12 }}>
          Load evaluation data to explore embeddings.
        </div>
      </div>
    );
  }

  if (!embeddingCoords) {
    return (
      <div className="card">
        <div className="card-title">Embedding Space Projection</div>
        <p
          style={{
            fontSize: 12,
            color: "var(--muted)",
            fontFamily: "'JetBrains Mono',monospace",
            lineHeight: 1.8,
          }}
        >
          UMAP coordinates not yet computed.
          <br />
          Run the following from the project root:
          <br />
          <code
            style={{
              display: "inline-block",
              marginTop: 10,
              padding: "6px 10px",
              background: "var(--bg3)",
              border: "1px solid var(--border2)",
              borderRadius: 6,
              color: "var(--cyan)",
            }}
          >
            python -m src.compute_embedding_coords
          </code>
        </p>
      </div>
    );
  }

  const diffColor = { easy: "#34d399", medium: "#fbbf24", hard: "#f87171" };
  const availableModels = Object.keys(embeddingCoords);
  const activeModel = selectedModel ?? availableModels[0];
  const activePoints = embeddingCoords[activeModel] ?? [];
  const activeColor = modelColor(activeModel);
  const activeEvaluation =
    models.find((model) => model.model_name === activeModel) ?? null;
  const activeHits =
    activeEvaluation?.results?.filter((result) => result.source_hit).length ??
    0;
  const activeTotal =
    activeEvaluation?.results?.length ?? activePoints.length ?? 0;
  const activeDimension = {
    "text-embedding-3-small": 1536,
    "all-MiniLM-L6-v2": 384,
    "multilingual-e5-base": 768,
  }[activeModel];
  const activeType = {
    "text-embedding-3-small": "OpenAI API",
    "all-MiniLM-L6-v2": "HuggingFace",
    "multilingual-e5-base": "HuggingFace",
  }[activeModel];

  return (
    <div className="stack" style={{ gap: 14 }}>
      <div className="card" style={{ padding: "18px 20px" }}>
        <div className="card-title">Embedding Space - UMAP Projection</div>
        <p
          style={{
            fontSize: 12,
            color: "var(--muted)",
            marginBottom: 14,
            fontFamily: "'JetBrains Mono',monospace",
            lineHeight: 1.65,
          }}
        >
          Each point represents a test question projected to 2D using UMAP.
          <br />
          Points close together are semantically similar.
        </p>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {availableModels.map((m) => {
            const isActive = m === activeModel;
            const color = modelColor(m);
            return (
              <button
                key={m}
                onClick={() => setSelectedModel(m)}
                style={{
                  cursor: "pointer",
                  padding: "8px 14px",
                  borderRadius: 8,
                  fontSize: 11,
                  fontWeight: 700,
                  letterSpacing: 1,
                  textTransform: "uppercase",
                  fontFamily: "'JetBrains Mono', monospace",
                  outline: "none",
                  border: `1px solid ${isActive ? color : "var(--border2)"}`,
                  background: isActive ? color + "18" : "var(--bg3)",
                  color: isActive ? color : "var(--muted)",
                  transition: "all .2s",
                }}
              >
                {MODEL_META[m]?.icon} {modelShort(m)}
              </button>
            );
          })}
        </div>
      </div>

      <div className="card" style={{ padding: "18px 20px 14px" }}>
        <div className="card-title">
          <span style={{ color: activeColor }}>{modelShort(activeModel)}</span>
          <span style={{ marginLeft: 8, opacity: 0.7, fontSize: 11 }}>
            - {activePoints.length} questions
          </span>
        </div>

        <div
          style={{
            display: "flex",
            gap: 14,
            marginBottom: 10,
            fontSize: 10,
            fontFamily: "'JetBrains Mono',monospace",
            color: "var(--muted)",
            textTransform: "uppercase",
            letterSpacing: 1,
          }}
        >
          <LegendDot color="var(--emerald)" label="Hit" />
          <LegendDot color="var(--red)" label="Miss" />
        </div>

        <ResponsiveContainer width="100%" height={440}>
          <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              type="number"
              dataKey="x"
              name="UMAP-1"
              tick={{
                fill: "var(--muted)",
                fontSize: 10,
                fontFamily: "'JetBrains Mono',monospace",
              }}
              axisLine={false}
              tickLine={false}
              label={{
                value: "UMAP-1",
                position: "insideBottom",
                offset: -10,
                fill: "var(--muted)",
                fontSize: 10,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="UMAP-2"
              tick={{
                fill: "var(--muted)",
                fontSize: 10,
                fontFamily: "'JetBrains Mono',monospace",
              }}
              axisLine={false}
              tickLine={false}
              label={{
                value: "UMAP-2",
                angle: -90,
                position: "insideLeft",
                dx: -10,
                fill: "var(--muted)",
                fontSize: 10,
                fontFamily: "'JetBrains Mono',monospace",
              }}
            />
            <ZAxis range={[100, 100]} />
            <Tooltip
              content={({ payload }) => {
                if (!payload?.length) return null;
                const d = payload[0]?.payload;
                if (!d) return null;
                return (
                  <div
                    style={{
                      background: "var(--bg2)",
                      border: "1px solid var(--border2)",
                      borderRadius: 8,
                      padding: "10px 14px",
                      fontSize: 11,
                      fontFamily: "'JetBrains Mono',monospace",
                      maxWidth: 260,
                    }}
                  >
                    <div
                      style={{
                        color: activeColor,
                        fontWeight: 700,
                        marginBottom: 6,
                        wordBreak: "break-word",
                      }}
                    >
                      {d.question}
                    </div>
                    <div
                      style={{
                        color: diffColor[d.difficulty] ?? "var(--muted)",
                        marginBottom: 4,
                      }}
                    >
                      {d.difficulty?.toUpperCase()}
                    </div>
                    <div
                      style={{
                        color: d.source_hit ? "var(--emerald)" : "var(--red)",
                      }}
                    >
                      {d.source_hit ? "Source hit" : "Source miss"}
                      {d.top_score != null &&
                        ` - ${(d.top_score * 100).toFixed(0)}% score`}
                    </div>
                  </div>
                );
              }}
            />
            <Scatter
              name="Hit"
              data={activePoints.filter((p) => p.source_hit)}
              fill="var(--emerald)"
              fillOpacity={0.85}
              stroke={activeColor}
              strokeWidth={1.5}
              shape={(props) => <OutcomeMarker {...props} />}
            />
            <Scatter
              name="Miss"
              data={activePoints.filter((p) => !p.source_hit)}
              fill="var(--red)"
              fillOpacity={0.85}
              stroke={activeColor}
              strokeWidth={1.5}
              shape={(props) => <OutcomeMarker {...props} />}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {activeEvaluation && (
        <div
          className="metric-card"
          style={{ borderColor: activeColor, padding: "16px 18px" }}
        >
          <div
            className="model-name"
            style={{ color: activeColor, marginBottom: 8 }}
          >
            {MODEL_META[activeModel]?.icon} {modelShort(activeModel)}
          </div>
          <div
            style={{
              fontSize: 11,
              color: "var(--muted)",
              fontFamily: "'JetBrains Mono',monospace",
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            <div>
              <div style={{ marginBottom: 4 }}>
                Source Hits:{" "}
                <span style={{ color: activeColor }}>
                  {activeHits}/{activeTotal}
                </span>
              </div>
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{
                    width: `${activeTotal ? (activeHits / activeTotal) * 100 : 0}%`,
                    background: activeColor,
                  }}
                />
              </div>
            </div>
            <div style={{ fontSize: 10 }}>
              Dim:{" "}
              <span style={{ color: "var(--text)" }}>{activeDimension}</span>
            </div>
            <div style={{ fontSize: 10 }}>
              Type: <span style={{ color: "var(--text)" }}>{activeType}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function LegendDot({ color, label, shape = "solid" }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: "50%",
          background: shape === "solid" ? color : "transparent",
          border: shape === "ring" ? `2px solid ${color}` : "none",
        }}
      />
      {label}
    </span>
  );
}

function OutcomeMarker({ cx, cy, fill, stroke }) {
  if (cx == null || cy == null) return null;
  return (
    <g>
      <circle cx={cx} cy={cy} r={6} fill={fill} fillOpacity={0.92} />
      <circle
        cx={cx}
        cy={cy}
        r={6}
        fill="transparent"
        stroke={stroke}
        strokeWidth={1.5}
      />
    </g>
  );
}

// ─── App Shell ─────────────────────────────────────────────────────────────
const VIEWS = [
  { id: "compare", label: "The Palantír", icon: "⊞" },
  { id: "benchmark", label: "The Halls of Mandos", icon: "◈" },
  { id: "explorer", label: " The Unseen Realm", icon: "⬡" },
];

export default function App() {
  const [view, setView] = useState("compare");
  const [evalData, setEvalData] = useState(null);
  const [embeddingCoords, setEmbeddingCoords] = useState(null);
  const [indexedModels, setIndexedModels] = useState([]);
  const [apiStatus, setApiStatus] = useState("checking");

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

    apiFetch("/embedding_coords")
      .then(setEmbeddingCoords)
      .catch(() => setEmbeddingCoords(null));
  }, []);

  const viewMeta = VIEWS.find((v) => v.id === view);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-brand">
          <h1>
            The
            <br />
            Red Book
            <br />
            of Westmarch
          </h1>
          <p>Tolkien Chatbot</p>
        </div>

        {VIEWS.map((v) => (
          <div
            key={v.id}
            className={`nav-item ${view === v.id ? "active" : ""}`}
            onClick={() => setView(v.id)}
          >
            <span className="icon">{v.icon}</span>
            {v.label}
          </div>
        ))}

        <div className="sidebar-footer">
          <div>
            <span
              className={`status-dot`}
              style={{
                background:
                  apiStatus === "online" ? "var(--emerald)" : "var(--red)",
              }}
            />
            API {apiStatus}
          </div>
          {indexedModels.length > 0 && (
            <div style={{ marginTop: 4 }}>
              {indexedModels.length} model{indexedModels.length > 1 ? "s" : ""}{" "}
              indexed
            </div>
          )}
          <div
            style={{
              marginTop: 8,
              borderTop: "1px solid var(--border)",
              paddingTop: 8,
            }}
          >
            NBI/Handelsakademin
            <br />
            Examensarbete 2026
          </div>
        </div>
      </aside>

      <div className="main">
        <div className="topbar">
          <div>
            <div className="topbar-title">{viewMeta?.label}</div>
            <div className="topbar-sub">
              {view === "compare" && "Three seeing-stones gaze in unison"}
              {view === "benchmark" && "Evaluation results — 19 test questions"}
              {view === "explorer" &&
                "Visualize retrieval quality distribution"}
            </div>
          </div>
          <div className="topbar-sep" />
          <div style={{ display: "flex", gap: 8 }}>
            {(indexedModels.length > 0
              ? indexedModels
              : Object.keys(MODEL_META)
            ).map((m) => (
              <ModelBadge key={m} model={m} />
            ))}
          </div>
        </div>

        <div className="content">
          {view === "compare" && <CompareView indexedModels={indexedModels} />}
          {view === "benchmark" && <BenchmarkView evalData={evalData} />}
          {view === "explorer" && (
            <ExplorerView
              evalData={evalData}
              embeddingCoords={embeddingCoords}
            />
          )}
        </div>
      </div>
    </div>
  );
}
