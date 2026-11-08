(() => {
  const H = HUD, { S, q, set, esc, spark, now } = H;
  const short = (id) => H.rname({ id }).split(" ·")[0];
  const fleetBots = () => (S.robots.length ? S.robots : (S.fleet?.robots || []).map((r) => ({ ...r, name: H.rname(r) })));
  const bot = () => fleetBots().find((r) => r.id === S.focus) || fleetBots()[0];
  const chips = (cur) => fleetBots().map((r) => `<button data-id="${esc(r.id)}" class="${r.id === cur ? "on" : ""}">${esc(r.name || H.rname(r))}</button>`).join("");
  const onFocus = (el) => { if (!el._f) { el._f = 1; el.addEventListener("click", (e) => { const b = e.target.closest("[data-id]"); if (b) H.focus(b.dataset.id); }); } };
  const last = (k) => S.series[k]?.at(-1)?.[1];
  const DECKS = [["auto", "Busiest deck"], ["top_deck", "Top"], ["middle_deck", "Middle"], ["lower_deck", "Lower"], ["all", "All decks"]];
  H.view({ id: "map", title: "Mission map", icon: "map", key: "⇧M", keys: "top down rig decks walkways plume people dock no-fly fleet",
    desc: "Top-down rig with decks as layers: robots with heading and status, walkways, jobs, gas plume, people and docks. Click a robot to focus it.",
    preview: (el) => { const x = q(el, `<canvas class="rig big" data-k="c"></canvas>`); if (now() - (el._t || 0) > 0.1) { el._t = now(); H.drawRig(x.c, S, { big: true }); } },
    render(el) {
      const x = q(el, `<div class="mm"><div class="seg" data-k="decks"></div><canvas data-k="c"></canvas></div><aside class="side" data-k="side"></aside>`);
      if (!x.c._b) {
        x.c._b = 1;
        x.c.onclick = (e) => { const r = x.c.getBoundingClientRect(), hit = (x.c._hits || []).find((p) => Math.hypot(p.x - e.clientX + r.left, p.y - e.clientY + r.top) < 18); if (hit) H.focus(hit.id); };
        x.decks.onclick = (e) => { const b = e.target.closest("button"); if (b) S.deck = b.dataset.d; };
        onFocus(x.side);
      }
      if (now() - (el._t || 0) > 0.066) { el._t = now(); x.c._hits = H.drawRig(x.c, S, { big: true, deck: S.deck }); }
      const d = `${DECKS.map(([k, l]) => `<button data-d="${k}" class="${(S.deck || "auto") === k ? "on" : ""}">${l}</button>`).join("")}`;
      if (x.decks._h !== d) { x.decks._h = d; x.decks.innerHTML = d; }
      const rows = (S.fleet?.robots || []).map((r) => `<div class="mrow ${r.id === S.focus ? "on" : ""}" data-id="${esc(r.id)}"><i style="background:${H.COL[H.tone(r.job_state || r.state)]}"></i><b>${esc(H.rname(r))}</b><span>${esc(H.nice(r.job_state || r.state))}${r.task ? " · " + esc(r.task) : ""}</span><em class="tnum">${r.battery?.soc == null ? "" : Math.round(r.battery.soc * 100) + " %"}</em></div>`).join("")
        + `<div class="legend col"><span><i class="lg r"></i>robot · heading</span><span><i class="lg t"></i>drone</span><span><i class="lg g"></i>gas plume</span><span><i class="lg p"></i>person · 3 m</span><span><i class="lg d"></i>dock</span><span><i class="lg n"></i>no-fly zone</span></div>`;
      if (x.side._h !== rows) { x.side._h = rows; x.side.innerHTML = (S.fleet ? "" : `<p class="none">No fleet snapshot yet</p>`) + rows; }
    } });
  H.view({ id: "fpv", title: "First person", icon: "cam", key: "⇧F", keys: "camera robot pov perception mask gauge overlay",
    desc: "The focused robot's camera with what it perceives: rust mask, gauge reading, gas reading and Gemini's points.",
    render(el) {
      const r = bot(), id = r?.id, f = S.fpv?.[id] || (r && r === fleetBots()[0] && S.inspection ? S.inspection : null);
      const x = q(el, `<div class="seg" data-k="who"></div><div class="fp"><div class="fp-img"><img data-k="img" alt=""><svg data-k="pts"></svg><p class="none" data-k="none"></p></div><aside class="side" data-k="side"></aside></div>`);
      onFocus(x.who);
      const c = chips(id);
      if (x.who._h !== c) { x.who._h = c; x.who.innerHTML = c; }
      if (f?.jpg && x.img._j !== f.jpg) { x.img._j = f.jpg; x.img.src = "data:image/jpeg;base64," + f.jpg; }
      x.img.hidden = !f; set(x.none, f ? "" : `No camera feed from ${r ? r.name || H.rname(r) : "any robot"} in this run`);
      const g = S.gemini, gk = f && g && f === S.inspection ? S.gkey : "";
      if (x.pts._k !== gk) { x.pts._k = gk; x.pts.setAttribute("viewBox", `0 0 ${f?.w || 1} ${f?.h || 1}`); x.pts.setAttribute("preserveAspectRatio", "xMidYMid meet"); x.pts.innerHTML = gk ? (g.points || []).map(([a, b]) => `<circle cx="${a}" cy="${b}" r="${f.w / 40}" stroke-width="${f.w / 160}"/>`).join("") : ""; }
      const gas = last(id + ".gas_lel") ?? (S.gasMax || null), lead = r === fleetBots()[0];
      const kv = [["Phase", r?.caption?.text || H.nice(r?.state)], ["Rust mask", f === S.inspection && f ? "outlined in red" : "—"], ["Gauge", f?.gauge ?? "—"],
        ["Gas", gas == null ? "—" : (gas * 100).toFixed(1) + " % LEL"], ["Brush force", lead ? S.force.toFixed(1) + " N" : "—"], ["Detector", lead && S.conf != null ? S.conf.toFixed(2) : "—"],
        ["Gemini", lead && g ? `${g.verdict.replace("_", " ")} · ${(g.confidence || 0).toFixed(2)}` : "—"]];
      const side = kv.map(([k, v]) => `<div class="kv"><span>${k}</span><b>${esc(v)}</b></div>`).join("");
      if (x.side._h !== side) { x.side._h = side; x.side.innerHTML = side; }
    } });
  const MET = [["gas_lel", "Gas", "% LEL", 100, 1, { min: 0, max: 30, color: "#F85149", refs: [[10, "rgba(227,179,65,.5)"]] }],
    ["thermal_c", "Thermal", "°C", 1, 1, { min: 0, max: 80, color: "rgba(255,255,255,.75)" }],
    ["force_n", "Contact force", "N", 1, 1, { min: 0, max: 70, ramp: H.fc, refs: [[40, "rgba(248,81,73,.6)"]] }],
    ["battery", "Battery", "%", 100, 0, { min: 0, max: 1, color: "#3FB950", refs: [[0.2, "rgba(227,179,65,.5)"]] }],
    ["reach", "IK reach", "%", 100, 0, { min: 0, max: 1, color: "#2EC4B6" }]];
  H.view({ id: "sensors", title: "Sensors", icon: "wave", key: "⇧S", keys: "gas ppm thermal force battery reach sparkline per robot",
    desc: "Per robot: gas, thermal, contact force against the 40 N line, battery and IK reach, over the last 30 s.",
    render(el) {
      const ids = fleetBots().map((r) => r.id), head = `<div class="sg" style="--n:${MET.length}"><span></span>${MET.map((m) => `<span class="sh">${m[1]} <em>${m[2]}</em></span>`).join("")}${ids.map((id) =>
        `<b class="${id === S.focus ? "on" : ""}">${esc(short(id))}</b>${MET.map((m) => `<div class="cell"><b class="tnum" data-k="${id}.${m[0]}"></b><canvas data-k="${id}.${m[0]}C"></canvas></div>`).join("")}`).join("")}</div>`;
      const x = q(el, ids.length ? head : `<p class="none">No robots reporting yet</p>`);
      if (now() - (el._t || 0) < 0.1) return;
      el._t = now();
      ids.forEach((id, n) => MET.forEach(([m, , , mul, d, o]) => {
        const k = id + "." + m, data = S.series[k] || (n === 0 && m === "force_n" ? S.series.force : null);
        const v = data?.at(-1)?.[1] ?? (m === "battery" ? (S.fleet?.robots || []).find((r) => r.id === id)?.battery?.soc : null);
        set(x[k], v == null ? "—" : (v * mul).toFixed(d)); spark(x[k + "C"], data || [], o);
      }));
    } });
  const GROUP = [["Running", /RUNNING|ASSIGNED|active|assigned/], ["Waiting approval", /WAITING/], ["Queued", /QUEUED|pending/], ["Done", /DONE|FINDING|done/], ["Crew and failed", /HUMAN|FAIL|escalat/]];
  const decided = new Map();
  function decide(job, ok) {
    H.event(ok ? "approve" : "reject", { job }, () => decided.set(job, ok ? "QUEUED" : "REJECTED"));
    H.addLog({ t: S.t, from: "Operator", to: ok ? "APPROVED" : "REJECTED", reason: job });
  }
  H.view({ id: "orch", title: "Orchestrator", icon: "inbox", key: "⇧O", keys: "jobs queue approvals inbox approve reject decisions why",
    desc: "The job queue by status, the approval inbox with Gemini's reasoning, and every decision with its why.",
    render(el) {
      const x = q(el, `<div class="orch"><section><h3>Job queue</h3><div data-k="jobs"></div></section><section><h3>Approval inbox</h3><div data-k="inbox"></div></section><section><h3>Decisions</h3><ol class="log long" data-k="log"></ol></section></div>`);
      if (!x.inbox._b) { x.inbox._b = 1; x.inbox.onclick = (e) => { const b = e.target.closest("[data-job]"); if (b) decide(b.dataset.job, b.dataset.a === "1"); }; }
      const o = S.orchestrator, jobs = (o?.jobs || (S.fleet?.tasks || []).map((t) => ({ id: t.id, kind: t.kind, target: t.poi, status: t.status, robot: t.robot, priority: t.priority }))).map((j) => (decided.has(j.id) ? { ...j, status: decided.get(j.id) } : j));
      const aps = (o?.approvals || []).filter((a) => !decided.has(a.job)), c = H.ctl(S), key = JSON.stringify(o) + decided.size + jobs.length + JSON.stringify(c?.agent) + JSON.stringify(c?.vetoed);
      if (x.jobs._k !== key) {
        x.jobs._k = key;
        x.jobs.innerHTML = GROUP.map(([g, re]) => { const js = jobs.filter((j) => re.test(j.status)); return js.length ? `<h4>${g} <em>${js.length}</em></h4>` + js.slice(0, 8).map((j) =>
          `<div class="job"><i class="p${Math.min(5, Math.round(j.priority || 3))}"></i><span>${esc((j.kind || "").replace(/_/g, " "))} <em>${esc(j.target || "")}</em></span><em>${j.robot ? esc(short(j.robot)) : ""}</em>${j.level ? `<kbd>${esc(j.level)}</kbd>` : ""}</div>`).join("") : ""; }).join("") || `<p class="none">No jobs yet</p>`;
        x.inbox.innerHTML = aps.map((a) => `<div class="ap"><b>${esc((a.kind || "").replace(/_/g, " "))} · ${esc(a.target || a.job)}</b><p>${esc(a.text)}</p><div><button data-job="${esc(a.job)}" data-a="1" class="yes">Approve</button><button data-job="${esc(a.job)}" data-a="0">Reject</button></div></div>`).join("") || `<p class="none">Nothing waiting for you</p>`;
        x.inbox.innerHTML += c?.agent || c?.vetoed?.length ? `<h3 class="sub3">Planner</h3>${H.planner(c.agent, 8)}${H.vetoes(c, 3)}` : "";
      }
      if (x.log._n !== S.logN) { x.log._n = S.logN; x.log.innerHTML = H.logRows(S, 16); }
    } });
  const rowsOf = (b) => (b.rows || b.methods || Object.entries(b.results || {}).map(([label, v]) => ({ label, ...(typeof v === "object" ? v : { value: v }) })))
    .map((r) => ({ label: r.label || r.name || r.method, v: r.value ?? r.mean, ci: r.ci || r.ci95 || (r.lo != null ? [r.lo, r.hi] : null) })).filter((r) => typeof r.v === "number");
  function benchCard(b) {
    const rs = rowsOf(b), hi = Math.max(...rs.map((r) => (r.ci ? r.ci[1] : r.v)), 1e-9), w = (v) => Math.max(0, Math.min(100, (v / hi) * 100));
    return `<div class="bench"><b>${esc(b.title || b.name)}</b><em>${esc(b.unit || b.metric || "")}${b.lower_is_better ? " · lower is better" : ""}</em>${rs.map((r) =>
      `<div class="br"><span>${esc(r.label)}</span><i><s style="width:${w(r.v)}%"></s>${r.ci ? `<u style="left:${w(r.ci[0])}%;width:${w(r.ci[1]) - w(r.ci[0])}%"></u>` : ""}</i><b class="tnum">${+r.v.toPrecision(3)}</b></div>`).join("")}${b.source ? `<p>${esc(b.source)}</p>` : ""}</div>`;
  }
  H.view({ id: "bench", title: "Benchmarks", icon: "chart", key: "⇧B", keys: "evidence charts ppo gemini greedy detector orchestration results",
    desc: "Measured results from bench/ (runs_sample/bench/*.json): means with 95 % intervals over the same seeded scenarios.",
    render(el) {
      if (!el._l) { el._l = 1; H.json("/api/bench").then((b) => (S.bench = b || [])); }
      const html = !S.bench ? `<p class="none">Loading…</p>` : S.bench.length ? `<div class="benches">${S.bench.map(benchCard).join("")}</div>` : `<p class="none">No benchmark results yet. bench/ writes runs_sample/bench/*.json.</p>`;
      if (el._h !== html) { el._h = html; el.innerHTML = html; }
    } });
  H.on((h) => {
    const ap = h.orchestrator?.approvals;
    if (!ap) return;
    [...Array(6).keys()].forEach((n) => H.uncommand("ap." + n));
    ap.filter((a) => !decided.has(a.job)).slice(0, 6).forEach((a, n) => H.command({ id: "ap." + n, group: "Control centre", label: `Approve ${(a.kind || "").replace(/_/g, " ")} · ${a.target || a.job}`, icon: "check",
      keys: "approve approval supv inbox", desc: a.text, hint: "SUPV", score: () => 0.92, run: () => decide(a.job, true) }));
  });
  H.command({ id: "intent", group: "Control centre", label: "Tell the planner…", icon: "spark", key: "⇧T", keys: "intent instruct operator gemini planner order",
    desc: "Type an instruction in plain words, such as “inspect deck 2, isolate any leak, don't paint without me”. Gemini turns it into jobs; the safety envelope still applies.",
    hint: () => (S.intent || S.control?.agent?.intent ? "Has an intent" : ""), score: () => (S.control ? 0.25 : 0),
    run: () => setTimeout(() => H.ask("Tell the planner", "Inspect deck 2, isolate any leak, don't paint without me", (text) => H.event("intent", { text }, () => { S.intent = text; H.addLog({ t: S.t, from: "Operator", to: "INTENT", reason: text }); })), 0) });
  const T = {};
  const U = (id, label, ic, key, name, args, local, o = {}) => H.command({ id: "u." + id, group: "Unscripted change", label, icon: ic, key, keys: "unscripted inject judge change " + (o.keys || ""), desc: o.desc,
    active: o.tog ? () => !!T[id] : undefined, score: o.score, run: () => { if (o.tog) T[id] = !T[id]; if (o.pre) o.pre(); H.event(name, typeof args === "function" ? args() : args, local); } });
  const scrub = () => S.state === "SCRUB";
  U("rust", "New rust patch", "spark", "⇧1", "spawn_patch", {}, () => {
    if (S.rust_grid) { const [bi, bj] = S.brush || [20, 30]; S.rust_grid = S.rust_grid.map((row, i) => row.map((v, j) => { const d = Math.hypot((i - bi + 6) / 3.5, (((j - bj + 13 + 64) % 64) / 4) - 1.5); return Math.max(v, d < 1.6 ? 0.85 * (1 - d / 1.6) + 0.1 : 0); })); }
    H.judge("SPRAY", "RE-PLAN: new rust injected by judge -> spray it, extend soak", "SPRAY");
  }, { desc: "Injects a fresh patch next to the brush. The planner re-plans: spray it, then extend the soak.", score: () => (scrub() ? 0.5 : 0) });
  U("glare", "Sun glare", "sun", "⇧2", "set_glare", () => ({ on: !!T.glare }), () => { document.getElementById("flash").classList.toggle("on", !!T.glare); H.judge("DETECT", T.glare ? "glare on inspection view -> detector unsure" : "glare cleared"); }, { tog: 1, desc: "Floods the inspection view with glare, so the detector loses confidence." });
  U("mist", "Lens mist", "cloud", "⇧3", "occlude_camera", () => ({ on: !!T.mist }), () => { S.mist = !!T.mist; H.judge(T.mist ? "HOLD" : "VERIFY", T.mist ? "perception lost (lens mist) -> hold position" : "camera view restored", T.mist ? "HOLD" : "VERIFY"); }, { tog: 1, desc: "Fogs the inspection lens. The robot holds until it can see again.", score: () => (S.state === "VERIFY" ? 0.6 : 0) });
  U("bead", "Weld bead", "bolt", "⇧4", "force_spike", {}, () => { S.spike = S.force = 64.8; S.peak = Math.max(S.peak, 64.8); H.push("force", 64.8); setTimeout(() => (S.spike = 0), 400); H.judge("BACK_OFF", "contact force 65 N > 40 N limit -> retract", "BACK_OFF"); }, { desc: "The brush hits a weld bead at 65 N. The arm backs off and retries softer.", score: () => (scrub() ? 0.8 : 0) });
  U("worker", "Worker walks up", "person", "⇧5", "person_at", () => ({ distance: T.worker ? 2.4 : null }), () => H.judge(T.worker ? "PAUSE_PERSON" : "SCRUB", T.worker ? "person at 2.4 m < 3.0 m -> pause" : "person clear -> resume", T.worker ? "PAUSE_PERSON" : "SCRUB"), { tog: 1, desc: "A person steps within 3 m. Every arm stops until they leave.", score: () => (scrub() ? 0.4 : 0) });
  let leak = 0;
  U("leak", "Gas leak", "wind", "⇧6", "gas_leak", () => ({ on: !!T.leak }), () => H.judge("GAS", T.leak ? "flange F2 leaking -> detectors rising" : "leak isolated"), { tog: 1, keys: "lel plume alarm", desc: "Flange F2 starts leaking. Detectors climb, the plume grows and Control centre ranks a response.", score: () => (S.started && !S.gasAlarm ? 0.35 : 0) });
  setInterval(() => {
    if (!S.sim || (!T.leak && leak < 0.003)) return;
    leak += ((T.leak ? 0.18 : 0) - leak) * 0.06;
    H.apply({ gas: { sensors: [["GD-1", "Flange F2", 1], ["GD-2", "Separator", 0.42], ["GD-3", "Compressor", 0.12], ["GD-4", "Wellhead", 0.05]].map(([i, at, k]) => ({ id: i, at, lel: Math.max(0.01, leak * k) })), alarm: leak >= 0.1, wind: { dir_deg: 225, speed_ms: 6 } } });
  }, 250);
  U("battery", "Robot battery fault", "battery", "⇧7", "battery_fault", () => ({ robot: bot()?.id || "unit-1" }), () => { const r = bot(); H.banner("warn", "Battery fault", `${r ? r.name || H.rname(r) : "Unit 1"} · to the dock, its job goes back to the queue`, "battery"); H.judge("TO_DOCK", "battery fault -> dock, requeue job"); }, { keys: "fault dock power", desc: "The focused robot's battery faults. It docks and the coordinator re-assigns its job." });
  U("link", "Cut the Gemini link", "link", "⇧8", "cut_link", () => ({ on: !!T.link }), null, { tog: 1, keys: "edge offline autonomy cloud", desc: "Drops the cloud link. The coordinator and robots keep working on their last plan, on the edge.",
    pre: () => { S.linkCut = !!T.link; H.banner(T.link ? "warn" : "ok", T.link ? "Gemini link cut" : "Gemini link back", T.link ? "Robots keep their last plan · edge only" : "Planner and second opinions resume", "link"); H.addLog({ t: S.t, from: "Judge", to: "EDGE", reason: T.link ? "cloud link cut -> edge autonomy, last plan kept" : "cloud link restored" }); },
    score: () => (S.control || S.gemini ? 0.3 : 0) });
  ["calm", "wind", "rain", "fog", "storm"].forEach((lv) => U("w" + lv, "Weather: " + lv, lv === "storm" || lv === "wind" ? "wind" : lv === "calm" ? "sun" : "cloud", lv === "storm" ? "⇧9" : undefined, "set_weather",
    { level: lv, ...(lv === "wind" || lv === "storm" ? { wind_dir: 225, gust: lv === "storm" ? 26 : 14 } : {}) }, null,
    { keys: "weather " + lv, desc: `Sets the weather to ${lv}. Gusts make the arm retract and wait; rain and fog cut detector confidence.`, pre: () => { S.weather = { level: lv }; H.addLog({ t: S.t, from: "Judge", to: "WEATHER", reason: "weather -> " + lv }); if (lv === "storm") H.banner("warn", "Storm", "Gusts to 26 m/s · the arm retracts in each gust", "wind"); } }));
  [["obstacle", "Drop an obstacle", "block", "⇧0", "drop_obstacle", "obstacle on the walkway -> re-route"], ["walkway", "Close a walkway", "block", undefined, "close_walkway", "walkway closed -> re-route via the next segment"],
    ["stairs", "Close the stairs", "block", undefined, "close_stairs", "stairs closed -> jobs on other decks wait"], ["clear", "Clear all changes", "x", undefined, "clear", "scenario cleared"]]
    .forEach(([id, label, ic, key, name, why]) => U(id, label, ic, key, name, {}, null, { keys: "nav route block path", desc: "Planner re-routes around it; robots on the segment wait.", pre: () => H.addLog({ t: S.t, from: "Judge", to: name.toUpperCase(), reason: why }) }));
})();
