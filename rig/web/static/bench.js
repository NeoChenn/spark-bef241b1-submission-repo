(() => {
  const H = HUD, { S, esc } = H;
  const M = {
    success: ["Success", "%", 0, 100], over_40N: ["Episodes over 40 N", "%", 1, 100], peak_force_N: ["Peak force", "N", 1, 1],
    time_to_clean_s: ["Time to clean", "s", 1, 1], fraction_removed: ["Rust removed", "%", 0, 100],
    leak_isolate_s: ["Leak isolated", "s", 1, 1], leak_detect_s: ["Leak detected", "s", 1, 1], jobs_per_hour: ["Jobs per hour", "", 0, 1],
    unsafe_executed: ["Unsafe actions run", "/ ep", 1, 1], approval_wait_s: ["Approval wait", "s", 1, 1], makespan_s: ["Makespan", "s", 1, 1],
    unsafe: ["Unsafe outcomes", "%", 1, 100], f1: ["F1", "", 0, 1], precision: ["Precision", "", 0, 1], recall: ["Recall", "", 0, 1], auroc: ["AUROC", "", 0, 1],
  };
  const SHORT = { rule: "Rule-based", edge: "Edge rules only", planner: "Gemini planner", planner_cut: "Gemini, link cut" };
  const TITLE = { scrub: "Scrub control", detect: "Rust detector", orch: "Orchestration", orch_pfleet: "Orchestration · fleet", safety: "Paint-over safety" };
  const isM = (v) => v && typeof v === "object" && "mean" in v && (v.mean == null || typeof v.mean === "number");
  const big = (rs) => Object.entries(rs || {}).filter(([, v]) => v && typeof v === "object").sort((a, b) => (b[1].n || 0) - (a[1].n || 0))[0];
  function arms(b) {
    const raw = b.arms || b.isaac_lab?.rig || b.isaac_lab?.train || b.cpu_mock?.rig?.arms || b.conditions || {};
    const list = Array.isArray(raw) ? raw.map((a) => [a.name, a]) : Object.entries(raw);
    let scope = b.arms ? "" : b.isaac_lab ? "Isaac Lab · rig scenarios" : b.cpu_mock ? "CPU mock · rig scenarios" : b.conditions ? "recorded scenarios" : "";
    const out = list.map(([k, a]) => {
      if (!a || typeof a !== "object") return null;
      let m = a.summary || a, ds = "";
      if (a.results) { const d = big(a.results); if (d) { m = d[1]; ds = d[0]; } }
      return { label: SHORT[k] || (a.name && a.name !== k ? a.name : a.label && a.label.length < 24 ? a.label : k), status: a.status, m, ds, n: a.episodes || m.n };
    }).filter(Boolean);
    const ds = out.find((a) => a.ds)?.ds;
    if (ds) scope = ds.replace(/_/g, " ");
    return [out, scope];
  }
  H.benchModel = (b) => {
    const [as, scope] = arms(b), ran = as.filter((a) => Object.values(a.m).some(isM));
    const keys = Object.keys(M).filter((k) => ran.some((a) => isM(a.m[k]) && a.m[k].mean != null)).slice(0, 4);
    const groups = keys.map((k) => {
      const [label, unit, lower, mul] = M[k], rows = ran.map((a) => ({ label: a.label, v: a.m[k]?.mean, lo: a.m[k]?.lo, hi: a.m[k]?.hi, mul })).filter((r) => r.v != null);
      const best = rows.reduce((x, r) => (x == null || (lower ? r.v < x.v : r.v > x.v) ? r : x), null);
      return { k, label, unit, lower: !!lower, rows: rows.map((r) => ({ ...r, best: r === best })) };
    });
    const pending = as.filter((a) => !ran.includes(a)).map((a) => ({ label: a.label, status: a.status || "not run" }));
    const pub = b.published?.real_test_881;
    const note = pub ? `Published on 881 real photos: F1 ${pub.f1}, accuracy ${pub.accuracy} (${b.published.model || "deployed model"})` : "";
    return { title: TITLE[b.name] || b.benchmark || b.name, scope, eps: b.episodes || (b.isaac_lab || b.cpu_mock ? ran[0]?.n : 0), n: ran[0]?.n, groups, pending, note };
  };
  const fmt = (v, mul) => { const x = v * mul; return Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toFixed(2).replace(/\.?0+$/, "") || "0"; };
  function card(b) {
    const m = H.benchModel(b);
    if (!m.groups.length && !m.pending.length) return "";
    const g = m.groups.map((gr) => {
      const hi = Math.max(...gr.rows.map((r) => Math.max(r.v, r.hi ?? r.v)), 1e-9), w = (v) => Math.max(0, Math.min(100, (v / hi) * 100));
      return `<div class="bm"><h5>${esc(gr.label)}${gr.unit ? ` <em>${esc(gr.unit)}</em>` : ""}<small>${gr.lower ? "lower is better" : "higher is better"}</small></h5>${gr.rows.map((r) =>
        `<div class="br ${r.best ? "best" : ""}"><span title="${esc(r.label)}">${esc(r.label)}</span><i><s style="width:${w(r.v)}%"></s>${r.lo != null && r.hi != null && r.hi > r.lo ? `<u style="left:${w(r.lo)}%;width:${w(r.hi) - w(r.lo)}%"></u>` : ""}</i><b class="tnum">${fmt(r.v, r.mul)}</b></div>`).join("")}</div>`;
    }).join("");
    const p = m.pending.map((a) => `<span class="pend">${esc(a.label)} · ${esc(String(a.status).replace(/ - /g, " — "))}</span>`).join("");
    return `<div class="bench"><b>${esc(m.title)}</b><em>${esc([m.scope, m.eps ? m.eps + " episodes" : m.n ? "n " + m.n : ""].filter(Boolean).join(" · "))}</em>${g}${p ? `<div class="pends">${p}</div>` : ""}${m.note ? `<p>${esc(m.note)}</p>` : ""}</div>`;
  }
  const ORDER = ["scrub", "orch_pfleet", "orch", "safety", "detect"];
  H.view({ id: "bench", title: "Benchmarks", icon: "chart", key: "⇧B", keys: "evidence charts ppo gemini greedy detector orchestration results",
    desc: "Measured results from bench/ (runs_sample/bench/*.json): means with 95 % intervals over the same seeded scenarios. The best arm per metric is highlighted.",
    render(el) {
      if (!el._l) { el._l = 1; H.json("/api/bench").then((b) => (S.bench = Array.isArray(b) ? b : [])); }
      const bs = (S.bench || []).slice().sort((a, b) => (ORDER.indexOf(a.name) + 99) % 99 - (ORDER.indexOf(b.name) + 99) % 99);
      const cards = bs.map(card).filter(Boolean).join("");
      const html = !S.bench ? `<p class="none">Loading benchmarks…</p>` : cards ? `<div class="benches">${cards}</div>` : `<p class="none">No benchmark results yet · bench/ writes runs_sample/bench/*.json</p>`;
      if (el._h !== html) { el._h = html; el.innerHTML = html; }
    } });
})();
