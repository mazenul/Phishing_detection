// SurfShield — Popup Script

// ── DOM refs ──────────────────────────────────────────────────────────────────
const stateLoading    = document.getElementById("stateLoading");
const stateNA         = document.getElementById("stateNA");
const stateError      = document.getElementById("stateError");
const stateResult     = document.getElementById("stateResult");

const scoreRing       = document.getElementById("scoreRing");
const scoreNumber     = document.getElementById("scoreNumber");
const verdictBadge    = document.getElementById("verdictBadge");
const riskBarFill     = document.getElementById("riskBarFill");
const analysisBadge   = document.getElementById("analysisBadge");

const dlProb          = document.getElementById("dlProb");
const dlConf          = document.getElementById("dlConf");
const mlStats         = document.getElementById("mlStats");
const mlProb          = document.getElementById("mlProb");
const mlConf          = document.getElementById("mlConf");
const mlUnavailable   = document.getElementById("mlUnavailable");

const urlText         = document.getElementById("urlText");
const reanalyzeBtn    = document.getElementById("reanalyzeBtn");


// ── Colour helpers ────────────────────────────────────────────────────────────

const VERDICT_META = {
  safe:       { label: "SAFE",       css: "color-safe",       bar: "#00c853" },
  suspicious: { label: "SUSPICIOUS", css: "color-suspicious", bar: "#ff9800" },
  unsafe:     { label: "UNSAFE",     css: "color-unsafe",     bar: "#f44336" },
};

function applyVerdict(verdict, score) {
  const meta = VERDICT_META[verdict] || VERDICT_META.suspicious;

  // Score ring
  scoreRing.className  = `score-ring ${meta.css}`;
  scoreNumber.textContent = Math.round(score);

  // Verdict badge
  verdictBadge.className   = `verdict-badge ${meta.css}`;
  verdictBadge.textContent = meta.label;

  // Risk bar
  riskBarFill.style.width           = `${Math.min(score, 100)}%`;
  riskBarFill.style.backgroundColor = meta.bar;
}

function pct(value) {
  return value !== null && value !== undefined
    ? `${(value * 100).toFixed(1)}%`
    : "—";
}


// ── Render result ─────────────────────────────────────────────────────────────

function showResult(data) {
  hideAll();
  stateResult.style.display = "block";

  applyVerdict(data.verdict, data.risk_score);

  // Analysis type
  analysisBadge.textContent =
    data.analysis_type === "full"
      ? "Full Analysis  •  DL + ML"
      : "Partial Analysis  •  DL Only";

  // DL model
  dlProb.textContent = pct(data.dl_probability);
  dlConf.textContent = pct(data.dl_confidence);

  // ML model
  if (data.ml_probability !== null && data.ml_probability !== undefined) {
    mlStats.style.display      = "flex";
    mlUnavailable.style.display = "none";
    mlProb.textContent = pct(data.ml_probability);
    mlConf.textContent = pct(data.ml_confidence);
  } else {
    mlStats.style.display      = "none";
    mlUnavailable.style.display = "block";
  }

  // URL
  const url = data.url || "";
  urlText.textContent = url;
  urlText.title       = url;
}

function showLoading() {
  hideAll();
  stateLoading.style.display = "flex";
}

function showNA() {
  hideAll();
  stateNA.style.display = "flex";
}

function showError() {
  hideAll();
  stateError.style.display = "flex";
}

function hideAll() {
  stateLoading.style.display = "none";
  stateNA.style.display      = "none";
  stateError.style.display   = "none";
  stateResult.style.display  = "none";
}


// ── Main: load result for current tab ────────────────────────────────────────

async function init() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) { showError(); return; }

  const key    = `tab_${tab.id}`;
  const stored = await chrome.storage.local.get(key);
  const data   = stored[key];

  if (!data) {
    showLoading();
    return;
  }

  switch (data.status) {
    case "loading": showLoading(); break;
    case "na":      showNA();      break;
    case "error":   showError();   break;
    case "done":    showResult(data); break;
    default:        showLoading();
  }

  // Re-analyze button
  reanalyzeBtn.addEventListener("click", async () => {
    if (!tab.url || !tab.url.startsWith("http")) return;
    reanalyzeBtn.disabled = true;
    reanalyzeBtn.textContent = "Analyzing…";
    showLoading();

    chrome.runtime.sendMessage(
      { type: "REANALYZE", tabId: tab.id, url: tab.url },
      () => {
        // Poll storage until result arrives
        const interval = setInterval(async () => {
          const s = await chrome.storage.local.get(key);
          const d = s[key];
          if (d && d.status !== "loading") {
            clearInterval(interval);
            if (d.status === "done")  showResult(d);
            else if (d.status === "error") showError();
            else if (d.status === "na")    showNA();
            reanalyzeBtn.disabled     = false;
            reanalyzeBtn.textContent  = "↻ Re-Analyze";
          }
        }, 500);
      }
    );
  });
}

init();
