// The page's behaviour: loads Pyodide and dstoolkit, then handles the dropzone,
// cleaning plan and downloads. Kept out of index.html so the page's
// Content-Security-Policy can forbid inline scripts.
"use strict";

const PYODIDE_INDEX = "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/";
const WHEEL = "__WHEEL__";            // filled in by app/build.py
const $ = id => document.getElementById(id);

let py = null, core = null, ready = null, current = null, cleaned = null;

function setStatus(text, { busy = false, error = false } = {}) {
  const el = $("status");
  el.className = "status" + (error ? " error" : "");
  el.innerHTML = (busy ? '<span class="spinner" aria-hidden="true"></span>' : "") + escapeHtml(text);
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function boot() {
  const started = performance.now();
  setStatus("Loading Python (the first visit downloads about 25 MB; later visits are faster)…", { busy: true });
  py = await loadPyodide({ indexURL: PYODIDE_INDEX });
  setStatus("Loading pandas and matplotlib…", { busy: true });
  await py.loadPackage(["micropip", "pandas", "matplotlib"]);
  setStatus("Loading dstoolkit…", { busy: true });
  const micropip = py.pyimport("micropip");
  await micropip.install.callKwargs(new URL(WHEEL, location.href).href, { deps: false });
  const source = await (await fetch("app_core.py")).text();
  py.FS.writeFile("/home/pyodide/app_core.py", source);
  core = py.pyimport("app_core");
  const seconds = ((performance.now() - started) / 1000).toFixed(0);
  setStatus(`Ready (loaded in ${seconds}s). Drop a file or try the sample.`);
  document.body.dataset.ready = "true";
}

async function analyze(name, bytes) {
  if (!(await ready)) return;   // Python failed to load; the status line already says so
  setStatus(`Profiling ${name}…`, { busy: true });
  $("result-card").classList.add("hidden");
  await new Promise(r => setTimeout(r, 30));  // let the status paint before Python blocks the page
  try {
    if (/\.xlsx?m?$/i.test(name)) {
      setStatus("Loading Excel support…", { busy: true });
      await py.pyimport("micropip").install("openpyxl");
    }
    py.FS.writeFile("/tmp/upload", bytes);
    const result = JSON.parse(core.analyze_file(name, "/tmp/upload"));
    if (!result.ok) { setStatus(result.error, { error: true }); return; }
    current = { name, ...result };
    renderPlan(result);
    renderReport(result);
    setStatus(`Profiled ${name}: ${result.rows.toLocaleString()} rows × ${result.columns} columns.`);
    document.body.dataset.analyzed = "true";
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong reading that file. Check that it's a table (CSV or Excel).", { error: true });
  }
}

function renderPlan(result) {
  const steps = result.suggestions.filter(s => s.actionable);
  const notes = result.suggestions.filter(s => !s.actionable);
  $("plan-sub").textContent = steps.length
    ? `${steps.length} suggested step${steps.length === 1 ? "" : "s"}. Untick any you don't want.`
    : "No cleaning steps suggested for this file.";
  $("plan").innerHTML = steps.map(s => `
    <li><label>
      <input type="checkbox" data-id="${s.id}" checked>
      <span class="sev ${s.severity}">${s.severity}</span>
      <span>${s.column ? `<code>${escapeHtml(s.column)}</code>: ` : ""}${escapeHtml(s.message)}</span>
    </label></li>`).join("");
  $("notes").innerHTML = notes.map(s =>
    `<li>${s.column ? `<code>${escapeHtml(s.column)}</code>: ` : ""}${escapeHtml(s.message)}</li>`).join("");
  $("plan-card").classList.remove("hidden");
}

function renderReport(result) {
  if (!$("report-style")) {
    const style = document.createElement("style");
    style.id = "report-style";
    style.textContent = result.css;
    document.head.appendChild(style);
  }
  $("report").innerHTML = result.html;   // generated and escaped by dstoolkit
}

async function runClean() {
  const ids = [...document.querySelectorAll("#plan input:checked")].map(i => Number(i.dataset.id));
  setStatus("Cleaning…", { busy: true });
  await new Promise(r => setTimeout(r, 30));
  const result = JSON.parse(core.clean(JSON.stringify(ids)));
  if (!result.ok) { setStatus(result.error, { error: true }); return; }
  cleaned = result;
  $("result-sub").textContent = `${result.rows.toLocaleString()} rows × ${result.columns} columns`;
  $("log").textContent = result.log;
  $("preview").innerHTML = result.preview;   // escaped by pandas
  $("result-card").classList.remove("hidden");
  $("result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  setStatus(`Applied ${ids.length} step${ids.length === 1 ? "" : "s"}. Download the results below.`);
  document.body.dataset.cleaned = "true";
}

function download(text, name, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- wiring ----------
const drop = $("drop");
["dragenter", "dragover"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", async ev => {
  const file = ev.dataTransfer.files[0];
  if (file) analyze(file.name, new Uint8Array(await file.arrayBuffer()));
});
$("file").addEventListener("change", async ev => {
  const file = ev.target.files[0];
  if (file) analyze(file.name, new Uint8Array(await file.arrayBuffer()));
  ev.target.value = "";
});
$("sample").addEventListener("click", async () => {
  try {
    const response = await fetch("sample/titanic3.csv");
    if (!response.ok) throw new Error(response.statusText);
    analyze("titanic3.csv", new Uint8Array(await response.arrayBuffer()));
  } catch (err) {
    setStatus("Couldn't load the sample file. Try dropping your own file instead.", { error: true });
  }
});
$("all").addEventListener("click", () => document.querySelectorAll("#plan input").forEach(i => i.checked = true));
$("none").addEventListener("click", () => document.querySelectorAll("#plan input").forEach(i => i.checked = false));
$("clean").addEventListener("click", runClean);
$("dl-csv").addEventListener("click", () => download(cleaned.csv, cleaned.clean_name, "text/csv"));
$("dl-script").addEventListener("click", () => download(cleaned.script, cleaned.clean_name.replace(/_clean\.csv$/, "_clean.py"), "text/x-python"));
$("dl-report").addEventListener("click", () => download(core.report_html(), current.name.replace(/\.[^.]+$/, "") + "_report.html", "text/html"));

ready = boot().then(() => true, err => {
  console.error(err);
  setStatus("Couldn't load Python in this browser. Check your connection and reload the page.", { error: true });
  return false;
});
