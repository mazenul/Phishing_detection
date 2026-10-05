// SurfShield — Background Service Worker
// • Auto-detects URL on every tab navigation
// • Right-click context menu: "Check link with SurfShield"
// • Shows Chrome notification with result (no page visit needed)

const API_URL = "http://127.0.0.1:8000/predict";

const VERDICT_META = {
  safe:       { color: "#00c853", label: "SAFE",       icon: "icons/safe_48.png"       },
  suspicious: { color: "#ff9800", label: "SUSPICIOUS", icon: "icons/suspicious_48.png" },
  unsafe:     { color: "#f44336", label: "UNSAFE",     icon: "icons/unsafe_48.png"     },
  loading:    { color: "#2196F3", label: "…",          icon: "icons/loading_48.png"    },
  na:         { color: "#757575", label: "",            icon: "icons/loading_48.png"    },
};


// ════════════════════════════════════════════════════════════════════════════
// Icon drawing (tab icon — OffscreenCanvas)
// ════════════════════════════════════════════════════════════════════════════

function makeIconImageData(statusColor, size) {
  const canvas = new OffscreenCanvas(size, size);
  const ctx    = canvas.getContext("2d");

  // ── Shield body (dark navy, always the same) ──────────────────────────────
  const pad = size * 0.09;
  ctx.fillStyle = "#1a3a5c";
  ctx.beginPath();
  ctx.moveTo(pad + size * 0.13, pad);
  ctx.lineTo(size - pad - size * 0.13, pad);
  ctx.quadraticCurveTo(size - pad, pad,          size - pad, pad + size * 0.13);
  ctx.lineTo(size - pad, size * 0.56);
  ctx.quadraticCurveTo(size - pad, size * 0.77,  size * 0.5, size - pad * 0.4);
  ctx.quadraticCurveTo(pad,        size * 0.77,  pad,        size * 0.56);
  ctx.lineTo(pad, pad + size * 0.13);
  ctx.quadraticCurveTo(pad, pad,                 pad + size * 0.13, pad);
  ctx.closePath();
  ctx.fill();

  // ── "S" centred in the shield body ───────────────────────────────────────
  ctx.fillStyle    = "rgba(255,255,255,0.92)";
  ctx.font         = `bold ${Math.floor(size * 0.42)}px Arial`;
  ctx.textAlign    = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("S", size * 0.5, size * 0.41);

  // ── Status dot — bottom-right corner ─────────────────────────────────────
  const dotR  = Math.round(size * 0.21);
  const dotX  = size - dotR;
  const dotY  = size - dotR;

  // white ring so dot is visible against any browser theme
  ctx.fillStyle = "#ffffff";
  ctx.beginPath();
  ctx.arc(dotX, dotY, dotR + Math.max(1, Math.round(size * 0.055)), 0, 2 * Math.PI);
  ctx.fill();

  // coloured status dot
  ctx.fillStyle = statusColor;
  ctx.beginPath();
  ctx.arc(dotX, dotY, dotR, 0, 2 * Math.PI);
  ctx.fill();

  return ctx.getImageData(0, 0, size, size);
}

function setTabIcon(tabId, colorHex) {
  try {
    chrome.action.setIcon({
      imageData: {
        16: makeIconImageData(colorHex, 16),
        32: makeIconImageData(colorHex, 32),
      },
      tabId,
    });
  } catch (_) {}
}

function setTabBadge(tabId, text, bgColor) {
  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color: bgColor, tabId });
}

function clearTabBadge(tabId) {
  chrome.action.setBadgeText({ text: "", tabId });
}


// ════════════════════════════════════════════════════════════════════════════
// Storage helpers
// ════════════════════════════════════════════════════════════════════════════

function saveResult(tabId, data) {
  chrome.storage.local.set({ [`tab_${tabId}`]: data });
}

function clearResult(tabId) {
  chrome.storage.local.remove(`tab_${tabId}`);
}


// ════════════════════════════════════════════════════════════════════════════
// Tab auto-detection
// ════════════════════════════════════════════════════════════════════════════

async function analyzeTab(tabId, url) {
  if (!url || !url.startsWith("http")) {
    setTabIcon(tabId, VERDICT_META.na.color);
    clearTabBadge(tabId);
    saveResult(tabId, { status: "na", url });
    return;
  }

  setTabIcon(tabId, VERDICT_META.loading.color);
  setTabBadge(tabId, "…", VERDICT_META.loading.color);
  saveResult(tabId, { status: "loading", url });

  try {
    const resp = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

    const data    = await resp.json();
    const meta    = VERDICT_META[data.verdict] || VERDICT_META.na;
    const score   = Math.round(data.risk_score);

    setTabIcon(tabId, meta.color);
    setTabBadge(tabId, String(score), meta.color);
    saveResult(tabId, { status: "done", ...data });

  } catch (err) {
    setTabIcon(tabId, VERDICT_META.na.color);
    setTabBadge(tabId, "!", "#757575");
    saveResult(tabId, { status: "error", url, error: err.message });
  }
}

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "complete" && tab.url) {
    analyzeTab(tabId, tab.url);
  }
});

chrome.tabs.onActivated.addListener(async (activeInfo) => {
  try {
    const tab    = await chrome.tabs.get(activeInfo.tabId);
    const stored = await chrome.storage.local.get(`tab_${activeInfo.tabId}`);
    const cached = stored[`tab_${activeInfo.tabId}`];

    if (!cached) {
      analyzeTab(activeInfo.tabId, tab.url);
    } else if (cached.status === "done") {
      const meta  = VERDICT_META[cached.verdict] || VERDICT_META.na;
      const score = Math.round(cached.risk_score);
      setTabIcon(activeInfo.tabId, meta.color);
      setTabBadge(activeInfo.tabId, String(score), meta.color);
    }
  } catch (_) {}
});

chrome.tabs.onRemoved.addListener((tabId) => clearResult(tabId));


// ════════════════════════════════════════════════════════════════════════════
// Context menu — "Check link with SurfShield"
// ════════════════════════════════════════════════════════════════════════════

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id:       "surfshield_check_link",
    title:    "Check link with SurfShield",
    contexts: ["link"],           // only shows when right-clicking a hyperlink
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "surfshield_check_link" && info.linkUrl) {
    checkLinkAndNotify(tab.id, info.linkUrl);
  }
});


// ════════════════════════════════════════════════════════════════════════════
// Link check + notification
// ════════════════════════════════════════════════════════════════════════════


// ── In-page toast (injected into the tab via scripting API) ──────────────────
// Bypasses OS notification permissions entirely — draws directly on the page.

function showToast(tabId, opts) {
  chrome.scripting.executeScript({
    target: { tabId },
    func: (o) => {
      // Remove any existing SurfShield toast first
      const existing = document.getElementById("__surfshield_toast__");
      if (existing) existing.remove();

      const ACCENT = {
        safe:       "#22c55e",
        suspicious: "#f97316",
        unsafe:     "#ef4444",
        loading:    "#6b7280",
        error:      "#6b7280",
      };
      const accent = ACCENT[o.verdict] || ACCENT.loading;

      const toast = document.createElement("div");
      toast.id = "__surfshield_toast__";

      toast.innerHTML = `
        <div style="display:flex;align-items:flex-start;gap:12px;">
          <div style="font-size:22px;line-height:1;flex-shrink:0;">${o.icon}</div>
          <div style="flex:1;min-width:0;">
            <div style="font-weight:700;font-size:14px;color:#f1f5f9;">${o.title}</div>
            <div style="font-size:11px;color:#94a3b8;margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${o.url}">${o.url}</div>
            ${o.sub ? `<div style="font-size:11px;color:#64748b;margin-top:4px;">${o.sub}</div>` : ""}
          </div>
          <div id="__pg_close__" style="cursor:pointer;color:#64748b;font-size:14px;flex-shrink:0;padding:2px 4px;line-height:1;user-select:none;">✕</div>
        </div>
      `;

      Object.assign(toast.style, {
        position:     "fixed",
        bottom:       "24px",
        right:        "24px",
        zIndex:       "2147483647",
        background:   "#1e293b",
        borderLeft:   `4px solid ${accent}`,
        borderRadius: "10px",
        padding:      "14px",
        boxShadow:    "0 8px 30px rgba(0,0,0,0.55)",
        minWidth:     "300px",
        maxWidth:     "380px",
        fontFamily:   "system-ui,-apple-system,sans-serif",
        opacity:      "0",
        transform:    "translateY(10px)",
        transition:   "opacity 0.22s ease, transform 0.22s ease",
        pointerEvents: "all",
      });

      document.body.appendChild(toast);

      // Entrance animation
      requestAnimationFrame(() => {
        toast.style.opacity   = "1";
        toast.style.transform = "translateY(0)";
      });

      const dismiss = () => {
        toast.style.opacity   = "0";
        toast.style.transform = "translateY(10px)";
        setTimeout(() => toast.remove(), 250);
      };

      document.getElementById("__pg_close__").onclick = dismiss;
      setTimeout(dismiss, 7000);   // auto-dismiss after 7 s
    },
    args: [opts],
  });
}

async function checkLinkAndNotify(tabId, url) {
  const shortUrl = url.length > 60 ? url.slice(0, 57) + "…" : url;

  // ── Step 1: "Analyzing…" toast ────────────────────────────────────────────
  showToast(tabId, {
    icon:    "🔍",
    title:   "SurfShield — Checking link…",
    url:     shortUrl,
    verdict: "loading",
  });

  // ── Step 2: Call backend ──────────────────────────────────────────────────
  let result;
  try {
    const controller = new AbortController();
    const timer      = setTimeout(() => controller.abort(), 30000);
    const resp = await fetch(API_URL, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ url }),
      signal:  controller.signal,
    });
    clearTimeout(timer);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    result = await resp.json();
  } catch (err) {
    showToast(tabId, {
      icon:    "❌",
      title:   "SurfShield — Server Offline",
      url:     "Start the server: start_server.bat",
      verdict: "error",
    });
    return;
  }

  // ── Step 3: Result toast ──────────────────────────────────────────────────
  const ICONS   = { safe: "✅", suspicious: "⚠️", unsafe: "🚨" };
  const score   = Math.round(result.risk_score);
  const label   = result.verdict.charAt(0).toUpperCase() + result.verdict.slice(1);
  const dlPct   = (result.dl_probability * 100).toFixed(1);
  const mlPct   = result.ml_probability !== null
    ? (result.ml_probability * 100).toFixed(1) + "%"
    : "—";
  const sub     = result.analysis_type === "partial"
    ? "Partial analysis — DL only (feature extraction failed)"
    : `DL: ${dlPct}%  |  ML: ${mlPct}`;

  showToast(tabId, {
    icon:    ICONS[result.verdict] || "❓",
    title:   `${label} — Risk Score: ${score} / 100`,
    url:     shortUrl,
    sub,
    verdict: result.verdict,
  });
}


// ════════════════════════════════════════════════════════════════════════════
// Messages from popup (re-analyze button)
// ════════════════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "REANALYZE") {
    clearResult(msg.tabId);
    analyzeTab(msg.tabId, msg.url).then(() => sendResponse({ ok: true }));
    return true;
  }
});
