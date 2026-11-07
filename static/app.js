/* ── CONFIG ─────────────────────────────────────────────────
   In production on Render the frontend is served by FastAPI
   so /predict is the same origin. For local dev change this
   to e.g. "http://localhost:8000/predict"
─────────────────────────────────────────────────────────── */
const API = "/predict";

/* ── UTILS ──────────────────────────────────────────────── */
function $(id) { return document.getElementById(id); }

/* ── TOAST ──────────────────────────────────────────────── */
function showToast(msg, isError = false) {
  const toast = $("toast");
  $("toast-icon").textContent = isError ? "✕" : "✓";
  $("toast-msg").textContent  = msg;
  toast.style.borderColor     = isError
    ? "rgba(239,68,68,0.3)"
    : "rgba(34,197,94,0.3)";
  toast.classList.add("show");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.remove("show"), 4000);
}

/* ── FIELD VALIDATION ───────────────────────────────────── */
function validateField(el) {
  const wrap = el.closest(".field");
  const invalid = !el.validity.valid || (el.tagName === "SELECT" && !el.value);
  wrap.classList.toggle("has-error", invalid);
  el.classList.toggle("error", invalid);
  return !invalid;
}

function attachValidation() {
  document.querySelectorAll("input[required], select[required]").forEach(el => {
    el.addEventListener("blur",  () => validateField(el));
    el.addEventListener("input", () => {
      if (el.classList.contains("error")) validateField(el);
    });
  });
}

/* ── GAUGE NEEDLE ───────────────────────────────────────── */
function animateNeedle(prob) {
  // prob 0→1 maps to rotation −90°→+90°
  const deg = -90 + prob * 180;
  $("needle").setAttribute("transform", `rotate(${deg}, 100, 100)`);
}

/* ── COLOR MAP ──────────────────────────────────────────── */
const COLOR_HEX = { green: "#22c55e", amber: "#f59e0b", red: "#ef4444" };
function hex(color) { return COLOR_HEX[color] ?? "#7c6ff7"; }

/* ── INSIGHT TEXT ───────────────────────────────────────── */
function buildInsight(data, form) {
  const pct   = data.confidence_pct;
  const grade = form.loan_grade;
  const prior = form.cb_person_default_on_file;
  const lpi   = parseFloat(form.loan_percent_income);

  const tips = [];
  if (prior === "Y")
    tips.push("prior default on record significantly increases risk");
  if (["E","F","G"].includes(grade))
    tips.push(`loan grade <strong>${grade}</strong> is in the high-risk tier`);
  if (lpi > 0.5)
    tips.push(`loan-to-income ratio of <strong>${(lpi * 100).toFixed(0)}%</strong> is elevated`);

  const tipStr = tips.length
    ? tips.join("; ") + "."
    : "borderline loan-to-income and grade combination.";

  if (data.risk_color === "green")
    return `Model confidence: <strong>${pct}%</strong> probability of default —
            applicant profile appears stable.
            ${tips.length ? "Note: " + tipStr : ""}`;

  if (data.risk_color === "amber")
    return `Moderate risk flagged at <strong>${pct}%</strong> default probability.
            Key factors: ${tipStr}`;

  return `High default risk — <strong>${pct}%</strong> probability exceeds the
          calibrated threshold of <strong>${(data.threshold * 100).toFixed(1)}%</strong>.
          ${tips.length ? "Contributing factors: " + tipStr : ""}`;
}

/* ── RENDER RESULT ──────────────────────────────────────── */
function renderResult(data, formData) {
  $("result-empty").style.display = "none";
  $("verdict").classList.add("show");

  const { default_probability: prob, confidence_pct: pct, risk_color: color } = data;
  const c = hex(color);

  /* gauge */
  animateNeedle(prob);
  $("gauge-pct").textContent = `${pct}%`;
  $("gauge-pct").style.color = c;

  /* risk chip */
  $("risk-chip-wrap").innerHTML = `
    <div class="risk-chip ${color}">
      <span class="dot"></span>${data.risk_band}
    </div>`;

  /* probability bar */
  const fill = $("bar-fill");
  fill.style.background = c;
  requestAnimationFrame(() => { fill.style.width = `${Math.min(pct, 100)}%`; });
  $("prob-pct-txt").textContent = `${pct}%`;
  $("prob-pct-txt").style.color = c;

  /* stats */
  const setStatVal = (id, val, colored = true) => {
    $(id).textContent  = val;
    if (colored) $(id).style.color = c;
  };

  setStatVal("stat-prob",       prob.toFixed(4));
  setStatVal("stat-threshold",  data.threshold.toFixed(4), false);
  setStatVal("stat-verdict",    data.default_prediction === 1 ? "Default" : "No Default");
  setStatVal("stat-confidence", `${pct}%`);

  /* insight */
  $("insight-box").innerHTML = buildInsight(data, formData);
}

/* ── BUILD API PAYLOAD ──────────────────────────────────── */
function buildPayload(fd) {
  return {
    person_age:                 parseInt(fd.get("person_age")),
    person_income:              parseFloat(fd.get("person_income")),
    person_home_ownership:      fd.get("person_home_ownership"),
    person_emp_length:          parseFloat(fd.get("person_emp_length")),
    loan_intent:                fd.get("loan_intent"),
    loan_grade:                 fd.get("loan_grade"),
    loan_amnt:                  parseFloat(fd.get("loan_amnt")),
    loan_int_rate:              parseFloat(fd.get("loan_int_rate")),
    loan_percent_income:        parseFloat(fd.get("loan_percent_income")),
    cb_person_default_on_file:  fd.get("cb_person_default_on_file"),
    cb_person_cred_hist_length: parseInt(fd.get("cb_person_cred_hist_length")),
  };
}

/* ── LOADING STATE ──────────────────────────────────────── */
function setLoading(on) {
  $("submit-btn").disabled    = on;
  $("spinner").style.display  = on ? "block" : "none";
  $("btn-label").textContent  = on ? "Analyzing…" : "Analyze Risk →";
}

/* ── FORM SUBMIT ────────────────────────────────────────── */
async function handleSubmit(e) {
  e.preventDefault();

  /* validate all fields */
  const fields = document.querySelectorAll("input[required], select[required]");
  let allValid = true;
  fields.forEach(f => { if (!validateField(f)) allValid = false; });
  if (!allValid) {
    showToast("Please fix the highlighted fields.", true);
    return;
  }

  const payload = buildPayload(new FormData(e.target));
  setLoading(true);

  try {
    const res = await fetch(API, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(payload),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `Server error ${res.status}`);
    }

    const data = await res.json();
    renderResult(data, payload);
    showToast("Analysis complete.");

  } catch (err) {
    showToast(err.message || "Network error — is the API running?", true);
    console.error(err);
  } finally {
    setLoading(false);
  }
}

/* ── INIT ───────────────────────────────────────────────── */
document.addEventListener("DOMContentLoaded", () => {
  attachValidation();
  $("form").addEventListener("submit", handleSubmit);
});

