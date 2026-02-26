/* ═══════════════════════════════════════════════════
   OSKAR — Dashboard App Logic v3
   ═══════════════════════════════════════════════════ */

const API = "http://localhost:8000";

const VOCAB = {
  safe: {
    starts: ["This project", "The new update", "Your analysis", "The recent article", "This design"],
    verbs: ["is beautifully executed", "seems to fix the bugs", "provides great insights", "looks incredible", "is very well thought out"],
    ends: ["and I appreciate it.", "which is super helpful.", "— great job!", "and I completely agree.", "which is very inspiring."]
  },
  hate: {
    starts: ["People like you", "This entire group", "You", "Those people", "Everyone here"],
    verbs: ["are disgusting parasites", "don't deserve to live", "are absolute scum", "should be eradicated", "make me sick"],
    ends: ["and I hate you.", "immediately.", "and no one loves you.", "— get out of here.", "and it's a known fact."]
  },
  misinfo: {
    starts: ["Vaccines", "5G towers", "The government", "Climate change", "The earth"],
    verbs: ["cause autism", "is spreading viruses", "is hiding the truth", "is a complete hoax", "is actually flat"],
    ends: ["— the CDC has covered it up forever.", "using secret radiation waves.", "to control our minds.", "and scientists are lying to us.", "and NASA fakes the photos."]
  },
  opinion: {
    starts: ["Pineapple on pizza", "Winter", "Coffee", "Star Wars Episode VIII", "Video games"],
    verbs: ["is the greatest thing ever", "is by far the superior season", "tastes better lukewarm", "is the best movie", "are a higher form of art"],
    ends: ["and I stand by that.", "compared to summer.", "than cinema.", "and anyone who disagrees is wrong.", "of our time."]
  }
};

function generateSentence(type) {
  const parts = VOCAB[type];
  const s = parts.starts[Math.floor(Math.random() * parts.starts.length)];
  const v = parts.verbs[Math.floor(Math.random() * parts.verbs.length)];
  const e = parts.ends[Math.floor(Math.random() * parts.ends.length)];
  return `${s} ${v} ${e}`;
}

const sessionLog = [];

// ─── BOOT ────────────────────────────────────────
window.onload = () => {
  ping(); // Changed from checkHealth() to ping() to match existing function
  setInterval(ping, 12000);
  // Remove any tooltip/title attribute that might have existed
  document.querySelectorAll("[title]").forEach(el => el.removeAttribute("title"));
  // Auto-resize composer input
  const input = document.getElementById("composerInput");
  input.addEventListener("input", function () {
    this.style.height = "56px"; // reset
    this.style.height = this.scrollHeight + "px";
  });
};

async function ping() {
  const dot = document.getElementById("sDot");
  const label = document.getElementById("sLabel");
  const fstat = document.getElementById("footerStatus");
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(2500) });
    const d = await r.json();
    if (d.status === "ok") {
      dot.className = "sdot on";
      label.textContent = "Online";
      if (fstat) fstat.textContent = "API Connected";
    } else throw 0;
  } catch {
    dot.className = "sdot off";
    label.textContent = "Offline";
    if (fstat) fstat.textContent = "API Offline";
  }
}

function loadSample(k) {
  const input = document.getElementById("composerInput");
  input.value = generateSentence(k);
  input.style.height = "56px";
  input.style.height = input.scrollHeight + "px";
}

function goHome() {
  document.getElementById("homeView").classList.remove("hidden");
  document.getElementById("resultsView").classList.add("hidden");
}

// ─── ANALYSIS ────────────────────────────────────
async function runAnalysis() {
  const btn = document.getElementById("btnSubmit");
  const text = document.getElementById("composerInput").value.trim();
  const uid = document.getElementById("userId").value.trim() || "anon";

  if (!text) { shake(document.getElementById("composerBox")); return; }

  btn.disabled = true;
  btn.textContent = "…";

  try {
    const res = await fetch(`${API}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": "REDACTED_USE_ENV_VAR"
      },
      body: JSON.stringify({ user_id: uid, text, context_thread: [] }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // Show results view
    document.getElementById("homeView").classList.add("hidden");
    const rv = document.getElementById("resultsView");
    rv.classList.remove("hidden");
    rv.style.animation = "none";
    rv.offsetHeight;
    rv.style.animation = null;

    renderScore(data);
    renderModules(data);
    addLog(text, data);

  } catch (e) {
    console.error(e);
    alert("Analysis failed — is the API running?");
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Analyze <span class="submit-arrow">↑</span>';
  }
}

// ─── RENDER ──────────────────────────────────────
function renderScore(d) {
  const pct = Math.round(d.risk_score * 100);
  const maxDash = 427.26; // 2π×68
  document.getElementById("scoreRing").setAttribute(
    "stroke-dasharray", `${(pct / 100) * maxDash} ${maxDash}`
  );
  animNum(document.getElementById("scoreNum"), pct);

  // Ring color + verdict text
  const ring = document.getElementById("scoreRing");
  const vt = document.getElementById("verdictText");
  const rt = document.getElementById("routeTag");
  const ci = document.getElementById("ciText");

  if (pct >= 60) {
    ring.style.stroke = "#b05a3a";
    vt.textContent = "High Risk";
    vt.style.color = "#b05a3a";
    rt.textContent = "Human Review";
    rt.className = "route-tag danger";
  } else if (pct >= 35) {
    ring.style.stroke = "#c8a96e";
    vt.textContent = "Caution";
    vt.style.color = "#c8a96e";
    rt.textContent = "Soft Warning";
    rt.className = "route-tag warn";
  } else {
    ring.style.stroke = "#5a7a5e";
    vt.textContent = "Safe";
    vt.style.color = "#f0ece3";
    rt.textContent = "Auto Pass";
    rt.className = "route-tag safe";
  }

  const lo = (d.confidence_interval[0] * 100).toFixed(1);
  const hi = (d.confidence_interval[1] * 100).toFixed(1);
  ci.textContent = `95% CI  ${lo}% — ${hi}%`;
}

function renderModules(d) {
  const h = d.components.hate;
  document.getElementById("mHateVal").textContent = h.label === "hate" ? "Toxic" : "Clean";
  document.getElementById("mHateVal").style.color = h.label === "hate" ? "#b05a3a" : "#5a7a5e";
  document.getElementById("mHateFill").style.width = `${h.score * 100}%`;
  document.getElementById("mHateDet").textContent = `score ${h.score.toFixed(2)} · ent ${h.uncertainty.toFixed(2)}`;

  const c = d.components.claim;
  document.getElementById("mClaimVal").textContent = c.is_verifiable ? c.claim_type : "Opinion";
  document.getElementById("mClaimFill").style.width = `${c.confidence * 100}%`;
  document.getElementById("mClaimDet").textContent = `conf ${c.confidence.toFixed(2)}`;

  const v = d.components.verification;
  const vMap = { supported: "Supported", refuted: "Refuted", uncertain: "Uncertain" };
  document.getElementById("mVerifyVal").textContent = vMap[v.verdict] || v.verdict;
  document.getElementById("mVerifyFill").style.width = `${v.confidence * 100}%`;
  document.getElementById("mVerifyDet").textContent = v.evidence ? v.evidence.substring(0, 38) + "…" : "No evidence";

  document.getElementById("mTrustVal").textContent = d.trust_score.toFixed(2);
  document.getElementById("mTrustFill").style.width = `${d.trust_score * 100}%`;
  document.getElementById("mTrustDet").textContent = `modifier ${(1.5 - d.trust_score).toFixed(2)}×`;
}

function addLog(text, d) {
  sessionLog.unshift({ text, d, t: new Date() });
  if (sessionLog.length > 30) sessionLog.pop();

  const body = document.getElementById("logBody");
  const pctColor = p => p >= 60 ? "#b05a3a" : p >= 35 ? "#c8a96e" : "#5a7a5e";
  const cls = r => ({ auto_action: "safe", soft_warning: "warn", human_review: "danger" })[r] || "";
  const lbl = r => ({ auto_action: "Auto", soft_warning: "Warning", human_review: "Review" })[r] || r;

  body.innerHTML = sessionLog.map(item => {
    const pct = Math.round(item.d.risk_score * 100);
    return `<div class="log-row">
      <span class="log-time">${item.t.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
      <span class="log-content" title="${item.text}">${item.text}</span>
      <span class="log-pct" style="color:${pctColor(pct)}">${pct}%</span>
      <span class="log-rt ${cls(item.d.route)}">${lbl(item.d.route)}</span>
    </div>`;
  }).join("");
}

// ─── UTILS ───────────────────────────────────────
function animNum(el, target) {
  const start = parseInt(el.textContent) || 0;
  const t0 = performance.now();
  (function f(now) {
    const p = Math.min((now - t0) / 900, 1);
    const ease = 1 - Math.pow(1 - p, 4);
    el.textContent = Math.round(start + (target - start) * ease);
    if (p < 1) requestAnimationFrame(f);
  })(t0);
}

function shake(el) {
  el.style.animation = "none";
  el.offsetHeight;
  el.style.animation = "shake 0.4s ease";
}
