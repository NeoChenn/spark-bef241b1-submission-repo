(() => {
  const Q = new URLSearchParams(location.search);
  const MOCK = Q.has("mock"), SPEED = parseFloat(Q.get("speed") || "1");
  const $ = (id) => document.getElementById(id);
  const B = document.body;
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const now = () => performance.now() / 1000;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const cap = (s) => s.charAt(0) + s.slice(1).toLowerCase().replace(/_/g, " ");
  const IC = {
    mission: "M5 21V4h11l-2 4 2 4H5", vitals: "M3 12h4l2-6 4 12 2-6h6", force: "M12 14l4-4M4 18a8 8 0 1 1 16 0",
    map: "M4 5h16v14H4zM4 12h16M12 5v14", gemini: "M12 3c.6 4.5 3.5 7.4 8 8-4.5.6-7.4 3.5-8 8-.6-4.5-3.5-7.4-8-8 4.5-.6 7.4-3.5 8-8z",
    log: "M9 6h11M9 12h11M9 18h11M4.5 6h.01M4.5 12h.01M4.5 18h.01", insp: "M4 8h3l2-3h6l2 3h3v11H4zM12 16a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    gas: "M3 8h11a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h8", plume: "M7 18a4 4 0 0 1-.5-8A6 6 0 0 1 18 9a4.5 4.5 0 0 1-1 9z",
    control: "M12 3v4M12 17v4M3 12h4M17 12h4M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z", fleet: "M6 9h12v8H6zM9.5 13h.01M14.5 13h.01M12 9V5M8 21v-4M16 21v-4",
    labels: "M3 12V4h8l10 10-8 8zM7.5 8h.01", subtitle: "M3 6h18v12H3zM7 14h4M13 14h4M7 10.5h10", help: "M3 7h18v11H3zM7 11h.01M11 11h.01M15 11h.01M8 14.5h8",
    camera: "M3 7h12v10H3zM15 11l6-3v8l-6-3", bolt: "M13 3L5 13h6l-1 8 8-10h-6z", play: "M7 5v14l11-7z",
    eye: "M3 3l18 18M10.6 6.1A9 9 0 0 1 12 6c6 0 9.5 6 9.5 6a16 16 0 0 1-3 3.6M6.6 6.6C3.9 8.4 2.5 12 2.5 12s3.5 6 9.5 6a9 9 0 0 0 4.4-1.1",
    reset: "M4 4v6h6M5.6 15A8 8 0 1 0 7 6.5L4 10", search: "M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zM20 20l-4-4",
    pin: "M9 4h6l-1 6 3 3H7l3-3zM12 13v7", close: "M6 6l12 12M18 6L6 18", warn: "M12 4l9 16H3zM12 10v4M12 17h.01",
    hand: "M8 13V5.5a1.5 1.5 0 0 1 3 0V11m0-6.5a1.5 1.5 0 0 1 3 0V11m0-5a1.5 1.5 0 0 1 3 0v8a7 7 0 0 1-14 0v-2a1.5 1.5 0 0 1 3 0", check: "M5 12l5 5 9-10",
  };
  const icon = (k) => `<svg class="i" viewBox="0 0 24 24"><path d="${IC[k] || IC.bolt}"/></svg>`;
  const STEPS = ["PATROL", "DETECT", "SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT", "DONE"];
  const STEP_OF = { PATROL: 0, DETECT: 1, SECOND_VIEW: 1, APPROACH: 1, SPRAY: 2, SOAK: 3, SCRUB: 4, VERIFY: 5, PAINT: 6, VERIFY_PAINT: 6, DONE: 7 };
  const SAFETY = { BACK_OFF: "warn", HOLD: "warn", PAUSE_PERSON: "stop", ESCALATE_TO_HUMAN: "stop" };
  const LABEL = { ESCALATE_TO_HUMAN: "Escalated", PAUSE_PERSON: "Person near", BACK_OFF: "Back off", SECOND_VIEW: "Second view", VERIFY_PAINT: "Verify paint" };
  const PHASE = {
    PATROL: "Patrolling the lower deck, looking for corrosion", DETECT: "Rust spotted on the riser. Taking a closer look",
    SECOND_VIEW: "Unsure. Moving for a second view", APPROACH: "Approaching the patch", SPRAY: "Spraying rust converter",
    SOAK: "Letting the converter soak in", SCRUB: "Scrubbing back to bare steel", VERIFY: "Checking the steel is clean",
    PAINT: "Sealing the pipe with primer", VERIFY_PAINT: "Checking the coat", DONE: "Repair verified and sealed",
    BACK_OFF: "Contact force spiked. Backing off", HOLD: "Camera obscured. Holding position", PAUSE_PERSON: "Person nearby. Work paused",
    ESCALATE_TO_HUMAN: "Handing over to the control room",
  };
  const ALARM = {
    BACK_OFF: ["warn", "Back off", "Contact force over the 40 N limit · retracting"],
    PAUSE_PERSON: ["stop", "Person in work zone", "Worker within 3 m · arm stopped"],
    HOLD: ["warn", "Holding", "Inspection camera obscured"],
    ESCALATE_TO_HUMAN: ["stop", "Escalated", "Operator needed in control room"],
  };
  const PANELS = [
    ["mission", "Mission", "i", "stepper soak progress"], ["control", "Control centre", "a", "gemini argon situation report actions"], ["gas", "Gas sensors", "s", "lel h2s detectors"],
    ["plume", "Gas plume", "o", "pip wind dispersion"], ["vitals", "Vitals", "v", "force rust confidence sparkline"], ["force", "Force sensor", "f", "brush newton limit"],
    ["map", "Pipe map", "m", "heatmap rust grid"], ["insp", "Inspection camera", "p", "pip view"], ["gemini", "Second opinion", "g", "gemini robotics verifier"],
    ["fleet", "Fleet", "t", "robots battery"], ["log", "Decisions", "l", "log events why"], ["help", "Keyboard shortcuts", "?", "controls keys help", "Help"],
  ].map(([id, name, key, kw, group]) => ({ id, name, key, kw, group }));
  const ACTIONS = [
    { id: "rust", label: "New rust patch", key: "1", ev: () => ({ name: "spawn_patch", args: {} }) },
    { id: "glare", label: "Glare", key: "2", toggle: 1, ev: (on) => ({ name: "set_glare", args: { on } }) },
    { id: "mist", label: "Lens mist", key: "3", toggle: 1, ev: (on) => ({ name: "occlude_camera", args: { on } }) },
    { id: "bead", label: "Weld bead (force spike)", key: "4", ev: () => ({ name: "force_spike", args: {} }) },
    { id: "worker", label: "Worker walks up", key: "5", toggle: 1, ev: (on) => ({ name: "person_at", args: { distance: on ? 2.4 : null } }) },
    { id: "leak", label: "Gas leak at separator", key: "6", toggle: 1, ev: (on) => ({ name: "gas_leak", args: { on } }) },
  ];
  const SHOTS = ["Wide rig", "Hero deck", "Close-up"];
  const ROBOTS = [
    { id: "spot1", name: "Spot-1", role: "Repair", x: 0.47, y: 0.56, bat: 78 },
    { id: "spot2", name: "Spot-2", role: "Gas sweep", x: 0.66, y: 0.66, bat: 64 },
    { id: "drone", name: "Drone-1", role: "Overwatch", x: 0.3, y: 0.24, bat: 51 },
  ];
  const GAS = [
    { id: "GD-101", where: "Wellhead A", v: 1.1, base: 1.1 }, { id: "GD-204", where: "Separator B", v: 0.8, base: 0.8 },
    { id: "GD-310", where: "Compressor", v: 2.0, base: 2.0 }, { id: "GD-412", where: "Mezzanine", v: 0.4, base: 0.4 },
  ];
  const FORCE_LIMIT = 40, VW = 30, FW = 20;
  const RAMP = [[0, [184, 182, 174]], [18, [181, 200, 74]], [25, [63, 185, 80]], [35, [63, 185, 80]], [38, [227, 179, 65]], [40, [227, 179, 65]], [40.01, [248, 81, 73]]];
  const forceColor = (f) => {
    let k = 1;
    while (k < RAMP.length - 1 && f > RAMP[k][0]) k++;
    const [a, ca] = RAMP[k - 1], [b, cb] = RAMP[k], u = Math.max(0, Math.min(1, (f - a) / (b - a || 1)));
    return `rgb(${ca.map((c, n) => Math.round(c + (cb[n] - c) * u)).join(",")})`;
  };
  const store = {
    get(k, d) { try { const v = localStorage.getItem("dock." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("dock." + k, JSON.stringify(v)); } catch {} },
  };
  const DEF_PINS = ["mission", "control", "gas", "plume", "vitals", "map", "insp", "log"];
  let pins = store.get("pins", DEF_PINS);
  const open = new Set(Q.has("panels") ? Q.get("panels").split(",") : store.get("open", []));
  let recent = store.get("recent", []);
  const S = { state: null, t: 0, grid: null, sprayed: null, painted: null, brush: null, insp: null, peak: 0, log: [], started: false, shot: 1, gas: 0, at: 0 };
  const series = { force: [], rust: [], conf: [] };
  const tweens = new Map();
  function num(el, v, d = 1) {
    if (v == null || Number.isNaN(v)) { tweens.delete(el); el.textContent = "—"; el._v = null; return; }
    if (el._v == null || RM) { el._v = v; el.textContent = v.toFixed(d); }
    tweens.set(el, { to: v, d });
  }
  function stepTweens() {
    tweens.forEach((tw, el) => {
      el._v += (tw.to - el._v) * 0.2;
      if (Math.abs(tw.to - el._v) < Math.pow(10, -tw.d) / 2) { el._v = tw.to; tweens.delete(el); }
      el.textContent = el._v.toFixed(tw.d);
    });
  }
  function fit(c) {
    const dpr = Math.min(devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return [g, w, h];
  }
  function spark(c, data, o) {
    const [g, w, h] = fit(c), t1 = now(), t0 = t1 - o.window;
    const x = (t) => ((t - t0) / o.window) * w, y = (v) => h - 2 - ((Math.min(o.max, v) - o.min) / (o.max - o.min)) * (h - 4);
    (o.refs || []).forEach((r) => {
      g.strokeStyle = r.color; g.lineWidth = 1; g.setLineDash(r.dash || []);
      g.beginPath(); g.moveTo(0, Math.round(y(r.v)) + 0.5); g.lineTo(w, Math.round(y(r.v)) + 0.5); g.stroke();
    });
    g.setLineDash([]);
    const pts = data.filter((p) => p[0] >= t0 - 1);
    if (pts.length < 2) return;
    g.lineWidth = o.width || 1.4; g.lineJoin = g.lineCap = "round";
    if (o.fill) {
      g.beginPath(); pts.forEach((p, k) => (k ? g.lineTo(x(p[0]), y(p[1])) : g.moveTo(x(p[0]), y(p[1]))));
      g.lineTo(x(pts[pts.length - 1][0]), h); g.lineTo(x(pts[0][0]), h); g.fillStyle = o.fill; g.fill();
    }
    for (let k = 1; k < pts.length; k++) {
      g.strokeStyle = o.ramp ? o.ramp(Math.max(pts[k - 1][1], pts[k][1])) : o.color;
      g.beginPath(); g.moveTo(x(pts[k - 1][0]), y(pts[k - 1][1])); g.lineTo(x(pts[k][0]), y(pts[k][1])); g.stroke();
    }
    const last = pts[pts.length - 1];
    g.fillStyle = o.ramp ? o.ramp(last[1]) : o.color;
    g.beginPath(); g.arc(Math.min(w - 2.5, x(last[0])), y(last[1]), 2.2, 0, 7); g.fill();
  }
  function push(k, v) {
    if (v == null) return;
    const s = series[k], t = S.at || now();
    s.push([t, v]);
    while (s.length && s[0][0] < t - VW - 2) s.shift();
  }
  function drawMap() {
    const [g, w, h] = fit($("map-canvas"));
    g.fillStyle = "rgba(255,255,255,0.04)"; g.fillRect(0, 0, w, h);
    if (!S.grid) return;
    const rows = S.grid.length, cols = S.grid[0].length, cw = w / cols, ch = h / rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const r = S.grid[i][j], pa = S.painted && S.painted[i][j] === "1", sp = S.sprayed && S.sprayed[i][j] === "1";
      const col = pa ? (r > 0.08 ? `rgba(181,83,42,${0.35 + r * 0.6})` : "rgba(143,168,148,0.9)") : r > 0.02 ? `rgba(181,83,42,${0.25 + r * 0.75})` : null;
      if (col) { g.fillStyle = col; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }
      if (sp && !pa) { g.fillStyle = "rgba(122,164,222,0.38)"; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }
    }
    if (S.brush) {
      g.strokeStyle = "rgba(255,255,255,.92)"; g.lineWidth = 1.5;
      g.beginPath(); g.arc((S.brush[1] + 0.5) * cw, (S.brush[0] + 0.5) * ch, 4.5, 0, 7); g.stroke();
    }
  }
  function drawPlume() {
    const [g, w, h] = fit($("plume")), t = now(), lvl = Math.min(1, GAS[1].v / 22);
    g.strokeStyle = "rgba(255,255,255,.14)"; g.lineWidth = 1;
    g.strokeRect(w * 0.08, h * 0.12, w * 0.84, h * 0.76);
    g.beginPath(); g.moveTo(w * 0.5, h * 0.12); g.lineTo(w * 0.5, h * 0.88); g.moveTo(w * 0.08, h * 0.5); g.lineTo(w * 0.92, h * 0.5); g.stroke();
    const sx = w * 0.62, sy = h * 0.62;
    for (let k = 0; k < 7 && lvl > 0.06; k++) {
      const d = ((t * 0.12 + k / 7) % 1), px = sx + d * w * 0.32, py = sy - d * h * 0.44, r = 8 + d * w * 0.16;
      const gr = g.createRadialGradient(px, py, 0, px, py, r);
      gr.addColorStop(0, `rgba(227,179,65,${0.6 * lvl * (1 - d * 0.7)})`); gr.addColorStop(1, "rgba(227,179,65,0)");
      g.fillStyle = gr; g.beginPath(); g.arc(px, py, r, 0, 7); g.fill();
    }
    g.fillStyle = lvl > 0.45 ? "#F85149" : "rgba(255,255,255,.5)"; g.beginPath(); g.arc(sx, sy, 3, 0, 7); g.fill();
    [[0.4, 0.66], [0.7, 0.72], [0.28, 0.3]].forEach(([u, v], k) => {
      g.fillStyle = k ? "rgba(255,255,255,.75)" : "#D97757"; g.beginPath(); g.arc(w * u, h * v, 3.5, 0, 7); g.fill();
    });
  }
  let islandT = null;
  function fitIsland() {
    const l = $("island").querySelector($("island").classList.contains("big") ? ".ban" : ".cmp");
    $("island").style.width = l.scrollWidth + "px";
    $("island").style.height = l.scrollHeight + "px";
  }
  function tone() {
    const t = S.gas > 10 ? "stop" : SAFETY[S.state] || (S.state === "DONE" ? "ok" : "");
    $("island").classList.remove("warn", "stop", "ok");
    if (t) $("island").classList.add(t);
  }
  function banner(t, title, sub, ic = "warn", ms = 3600) {
    if (S.at && now() - S.at > 1) return;
    B.classList.add("alarm");
    $("b-title").textContent = title; $("b-sub").textContent = sub; $("b-icon").innerHTML = icon(ic);
    $("island").classList.remove("warn", "stop", "ok");
    if (t) $("island").classList.add(t);
    $("island").classList.add("big"); fitIsland();
    clearTimeout(islandT);
    islandT = setTimeout(() => { $("island").classList.remove("big"); B.classList.remove("alarm"); tone(); fitIsland(); }, RM ? ms + 1000 : ms);
  }
  function subtitle(text) {
    const el = $("subtitle");
    if (el.dataset.k === text) return;
    el.dataset.k = text;
    el.classList.add("swap");
    setTimeout(() => { el.innerHTML = text; el.classList.remove("swap"); }, RM ? 0 : 260);
  }
  function setState(st) {
    if (st === S.state) return;
    S.state = st;
    $("i-state").textContent = LABEL[st] || cap(st);
    subtitle(PHASE[st] || cap(st));
    tone(); fitIsland();
    if (ALARM[st]) banner(ALARM[st][0], ALARM[st][1], ALARM[st][2], st === "PAUSE_PERSON" ? "hand" : "warn");
    if (st === "DONE") banner("ok", "Repair complete", "Steel verified clean and sealed", "check", 3000);
    if (st in STEP_OF) {
      const k = STEP_OF[st];
      [...$("seg").children].forEach((el, n) => { el.className = n < k ? "done" : n === k ? "cur" : ""; });
      $("seg").classList.toggle("ok", st === "DONE");
      $("m-step").textContent = cap(STEPS[k]);
      $("m-of").textContent = `${k + 1} of 8`;
    }
    renderRobots(); renderControl();
    if (st === "DONE" && MOCK) setTimeout(() => shot(1), 3500);
  }
  function addEvents(evs) {
    if (!evs.length) return;
    evs.forEach((e) => S.log.unshift(e));
    S.log = S.log.slice(0, 5);
    $("log-list").innerHTML = S.log.map((e) => {
      const c = SAFETY[e.to] === "stop" ? "s" : SAFETY[e.to] || /^RE-PLAN/.test(e.reason || "") ? "w" : e.to === "DONE" ? "k" : "";
      return `<li class="${c}"><span class="t num">${Number(e.t).toFixed(1)}</span><span>${esc(LABEL[e.from] || cap(e.from))} → <b>${esc(LABEL[e.to] || cap(e.to))}</b></span><span class="why">${esc(e.reason || "")}</span></li>`;
    }).join("");
    if (!open.has("log")) badge("log");
  }
  function showGemini(gm) {
    const key = JSON.stringify(gm);
    if (!gm || key === S.gKey) return;
    S.gKey = key;
    const v = gm.verdict || "cannot_tell";
    $("g-verdict").className = "chip " + v;
    $("g-verdict").textContent = { rust: "Rust", clean: "Clean", cannot_tell: "Can't tell" }[v] || v;
    $("g-lat").textContent = Math.round(gm.latency_ms || 0) + " ms";
    num($("g-conf"), gm.confidence || 0, 2);
    $("g-reason").textContent = gm.reason || "";
    if (S.insp) {
      $("g-img").src = "data:image/jpeg;base64," + S.insp.jpg;
      const svg = $("g-pts"), r = S.insp.w / 40;
      svg.setAttribute("viewBox", `0 0 ${S.insp.w} ${S.insp.h}`);
      svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
      svg.innerHTML = (gm.points || []).map(([x, y]) => `<circle cx="${x}" cy="${y}" r="${r}" stroke-width="${r / 4}"/>`).join("");
    }
    if (!open.has("gemini")) badge("gemini");
  }
  function apply(h) {
    if (h.state) setState(h.state);
    if (h.t != null) { S.t = h.t; num($("i-time"), h.t, 1); }
    if (h.events) addEvents(h.events);
    if (h.rust_grid) S.grid = h.rust_grid;
    ["sprayed", "painted"].forEach((k) => { if (h[k]) S[k] = h[k]; });
    if (h.brush) S.brush = h.brush;
    if (h.force_n != null) {
      const f = S.spike ? Math.max(h.force_n, S.spike) : h.force_n;
      push("force", f); S.f = f; S.peak = Math.max(S.peak, f);
      num($("v-force"), f, 1); num($("fm-now"), f, 1); num($("fm-max"), S.peak, 1);
      $("fm-now").style.color = forceColor(f);
      $("fm-gauge").style.width = Math.min(100, (f / 50) * 100) + "%";
      $("fm-gauge").style.background = forceColor(f);
    }
    if (h.rust_fraction != null) { S.rust = h.rust_fraction * 100; push("rust", S.rust); num($("v-rust"), S.rust, 1); }
    if (h.detector_conf != null) { push("conf", h.detector_conf); num($("v-conf"), h.detector_conf, 2); }
    if (h.inspection) { S.insp = h.inspection; $("pip-img").src = "data:image/jpeg;base64," + h.inspection.jpg; }
    if (h.gemini) showGemini(h.gemini);
    S.soak = h.soak || null;
    $("soak").classList.toggle("on", !!S.soak);
    if (S.soak) { num($("soak-left"), S.soak.left, 0); $("soak-ring").style.strokeDashoffset = String(50.3 * (1 - S.soak.left / S.soak.total)); }
    keyNumber();
  }
  function keyNumber() {
    const [v, d, u] = S.gas > 10 ? [S.gas, 1, "% LEL"] : S.soak ? [S.soak.left, 0, "s left"]
      : ["SCRUB", "SPRAY", "PAINT", "BACK_OFF"].includes(S.state) && S.f != null ? [S.f, 1, "N"] : [S.rust, 1, S.rust == null ? "" : "% rust"];
    num($("i-key"), v, d);
    if ($("i-unit").textContent !== u) { $("i-unit").textContent = u; fitIsland(); }
  }
  function tickGas() {
    const leak = ACTIONS[5].on;
    GAS.forEach((g, k) => {
      const tgt = leak ? [3.2, 23.5, 7.4, 1.6][k] : g.base;
      g.v = Math.max(0, g.v + (tgt - g.v) * 0.12 + (Math.random() - 0.5) * 0.12);
    });
    const peak = Math.max(...GAS.map((g) => g.v)), was = S.gas;
    S.gas = peak;
    if (peak > 10 && was <= 10) {
      banner("stop", `Gas · ${peak.toFixed(0)}% LEL`, "Separator B · repair paused, robots to muster", "gas", 4200);
      ["gas", "control", "plume"].forEach((p) => !open.has(p) && badge(p, "stop"));
      renderRobots(); renderControl();
    }
    if (peak <= 10 && was > 10) { tone(); renderRobots(); renderControl(); }
    if (open.has("gas")) $("gas-list").innerHTML = GAS.map((g) => {
      const c = g.v > 20 ? "stop" : g.v > 10 ? "warn" : "";
      return `<li class="${c}" title="${g.id}"><b>${g.where}</b><span class="num">${g.v.toFixed(1)}<small>% LEL</small></span><div class="bar"><i style="width:${Math.min(100, g.v * 4)}%"></i></div></li>`;
    }).join("");
    $("plume-note").textContent = leak ? `Drifting NE at 4 kn · ${peak.toFixed(0)}% LEL at source` : "Wind 4 kn from SW · no plume";
    keyNumber();
  }
  function robotTask(r) {
    if (S.gas > 10) return r.id === "spot2" ? ["Isolating V-14", "warn"] : r.id === "drone" ? ["Surveying plume", ""] : ["To muster", "stop"];
    if (r.id === "spot1") return [LABEL[S.state] || cap(S.state || "STANDBY"), SAFETY[S.state] || ""];
    return [r.id === "spot2" ? "Sweeping deck B" : "Holding orbit", ""];
  }
  function renderRobots() {
    const t = ROBOTS.map(robotTask);
    $("fleet-list").innerHTML = ROBOTS.map((r, k) => `<li class="${t[k][1]}"><b>${r.name}</b><span>${r.role} · ${t[k][0]}</span><span class="num">${r.bat}<small>%</small></span></li>`).join("");
    $("labels").innerHTML = ROBOTS.map((r, k) => `<div class="lab ${t[k][1]}" data-r="${r.id}" style="left:${r.x * 100}%;top:${r.y * 100}%"><div><i></i>${r.name}<span>${esc(t[k][0])}</span></div></div>`).join("");
  }
  function renderControl() {
    const gas = S.gas > 10;
    const sum = gas ? `Separator B is at ${S.gas.toFixed(0)}% LEL and rising. The plume is drifting north-east toward the Spot-1 work zone. Repair is paused.`
      : SAFETY[S.state] ? `Spot-1 is in ${LABEL[S.state] || cap(S.state)}. The safety layer has it; no operator action is needed yet.`
      : `Nominal. Spot-1 is ${(PHASE[S.state] || "standing by").toLowerCase()}. Gas readings are at background on all four detectors.`;
    const acts = gas ? [["Send Spot-1 to muster", "Clears the plume path", 0.93], ["Spot-2 isolates V-14", "Upstream of the leak", 0.81], ["Drone-1 maps the plume", "Survey from above", 0.64]]
      : [["Continue the repair", "Under the 40 N cap", 0.9], ["Drone pass of deck B", "Last survey 41 min ago", 0.58], ["Notify the supervisor", "Routine", 0.4]];
    if ($("cc-sum").textContent === sum) return;
    $("cc-sum").textContent = sum;
    $("cc-list").innerHTML = acts.map(([a, why, p]) => `<li><div><b>${a}</b><span>${Math.round(p * 100)}% · ${why}</span></div><button>Run</button></li>`).join("");
    [...$("cc-list").querySelectorAll("button")].forEach((b, k) => b.onclick = () => {
      b.textContent = "Queued"; b.classList.add("done");
      addEvents([{ t: S.t, from: "OPERATOR", to: "ACTION", reason: acts[k][0] }]);
    });
  }
  function badge(id, cls = "") {
    const b = $("dock").querySelector(`[data-p="${id}"]`);
    if (b) b.className = (b.classList.contains("on") ? "on " : "") + "badge " + cls;
  }
  function renderDock() {
    $("dock").innerHTML = pins.map((id) => {
      const p = PANELS.find((x) => x.id === id);
      return p ? `<button data-p="${id}" class="${open.has(id) ? "on" : ""}" aria-label="${p.name}">${icon(id)}<span>${p.name}</span></button>` : "";
    }).join("") + `<hr><button data-p="spot" aria-label="Search">${icon("search")}<span>Search  ⌘K</span></button>`;
    $("dock").querySelectorAll("button").forEach((b) => b.onclick = () => b.dataset.p === "spot" ? openSpot() : togglePanel(b.dataset.p));
  }
  let dockT = null;
  function peekDock(ms = 2400) {
    B.classList.add("dock-on");
    clearTimeout(dockT);
    dockT = setTimeout(() => { if (!$("dock").matches(":hover") && !B.classList.contains("spot")) B.classList.remove("dock-on"); }, ms);
  }
  function togglePanel(id, on = !open.has(id)) {
    on ? open.add(id) : open.delete(id);
    store.set("open", [...open]);
    const sec = $("side").querySelector(`[data-p="${id}"]`);
    sec.classList.toggle("on", on);
    $("side").classList.toggle("empty", !open.size);
    renderDock();
    if (on) $("side").scrollTop = sec.offsetTop - 8;
  }
  function buildSide() {
    $("side").querySelectorAll("section").forEach((sec) => {
      const p = PANELS.find((x) => x.id === sec.dataset.p);
      sec.insertAdjacentHTML("afterbegin", `<header>${icon(p.id)}<span>${p.name}</span><button class="x" aria-label="Close">${icon("close")}</button></header>`);
      sec.querySelector(".x").onclick = () => togglePanel(p.id, false);
      sec.classList.toggle("on", open.has(p.id));
    });
    $("side").classList.toggle("empty", !open.size);
    $("seg").innerHTML = STEPS.map(() => "<i></i>").join("");
  }
  const CMDS = [
    ...PANELS.map((p) => ({ id: "p:" + p.id, group: p.group || "Panels", label: p.name, icon: p.id, key: p.key.toUpperCase(), kw: p.kw, pin: p.id, on: () => open.has(p.id), run: () => togglePanel(p.id) })),
    { id: "o:labels", group: "Panels", label: "Robot labels", icon: "labels", key: "R", kw: "overlay names", on: () => !B.classList.contains("no-labels"), run: () => B.classList.toggle("no-labels") },
    { id: "o:sub", group: "Panels", label: "Phase subtitle", icon: "subtitle", kw: "caption overlay", on: () => !B.classList.contains("no-subtitle"), run: () => B.classList.toggle("no-subtitle") },
    ...SHOTS.map((s, k) => ({ id: "c:" + k, group: "Cameras", label: s, icon: "camera", key: String(k + 1), kw: "shot view", on: () => S.shot === k + 1, run: () => shot(k + 1) })),
    ...ACTIONS.map((a) => ({ id: "e:" + a.id, group: "Scenario events", label: a.label, icon: a.id === "leak" ? "gas" : "bolt", key: "⇧" + a.key, kw: "unscripted judge inject", on: a.toggle ? () => !!a.on : null, run: () => fire(a) })),
    { id: "d:start", group: "Demo", label: "Start demo", icon: "play", key: "Space", kw: "begin run", run: () => begin() },
    { id: "d:cine", group: "Demo", label: "Cinematic mode", icon: "eye", key: "C", kw: "hide all clean", on: () => B.classList.contains("cine"), run: () => B.classList.toggle("cine") },
    { id: "d:close", group: "Demo", label: "Close all panels", icon: "close", kw: "clear", run: () => [...open].forEach((p) => togglePanel(p, false)) },
    { id: "d:reset", group: "Demo", label: "Reset dock", icon: "reset", kw: "pins default layout", run: () => { pins = DEF_PINS.slice(); store.set("pins", pins); renderDock(); peekDock(); } },
    ...ROBOTS.map((r) => ({ id: "r:" + r.id, group: "Robots", label: `Locate ${r.name}`, icon: "fleet", hint: r.role, kw: "find focus robot", run: () => locate(r) })),
  ];
  const GROUPS = ["Panels", "Cameras", "Scenario events", "Demo", "Robots", "Help"];
  let rows = [], sel = 0;
  function score(c, q) {
    const s = (c.label + " " + c.group + " " + (c.kw || "")).toLowerCase(), l = c.label.toLowerCase();
    if (l.startsWith(q)) return [100, [[0, q.length]]];
    const w = l.indexOf(" " + q);
    if (w >= 0) return [80, [[w + 1, q.length]]];
    if (l.includes(q)) return [60, [[l.indexOf(q), q.length]]];
    let i = 0, hits = [];
    for (let k = 0; k < l.length && i < q.length; k++) if (l[k] === q[i]) { hits.push([k, 1]); i++; }
    if (i === q.length) return [40 - hits[hits.length - 1][0] + hits[0][0], hits];
    return q.split(" ").every((t) => s.includes(t)) ? [20, []] : [0, []];
  }
  function mark(label, hits) {
    let out = "", k = 0;
    hits.forEach(([a, n]) => { out += esc(label.slice(k, a)) + "<mark>" + esc(label.slice(a, a + n)) + "</mark>"; k = a + n; });
    return out + esc(label.slice(k));
  }
  function renderResults() {
    const q = $("q").value.trim().toLowerCase();
    let groups;
    if (!q) {
      const rec = recent.map((id) => CMDS.find((c) => c.id === id)).filter(Boolean).slice(0, 4);
      const sug = (S.started ? ["p:control", "p:gas", "e:bead", "e:leak", "d:cine"] : ["d:start", "p:control", "c:1"])
        .map((id) => CMDS.find((c) => c.id === id)).filter((c) => !rec.includes(c)).slice(0, 7 - rec.length);
      groups = [["Recent", rec.map((c) => [c, []])], ["Suggested", sug.map((c) => [c, []])]];
    } else {
      const m = CMDS.map((c) => [c, ...score(c, q)]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]);
      const top = m[0] && m[0][0];
      groups = (top ? [["Top hit", [[top, m[0][2]]]]] : []).concat(GROUPS.map((g) => [g, m.filter((x) => x[0].group === g && x[0] !== top).map((x) => [x[0], x[2]])]));
    }
    rows = [];
    $("results").innerHTML = groups.filter((g) => g[1].length).map(([g, list]) => `<div class="grp">${g}</div>` + list.map(([c, hits]) => {
      rows.push(c);
      const n = rows.length - 1, on = c.on && c.on();
      const hint = c.hint || (c.pin ? (open.has(c.pin) ? "Open" : "") : c.on ? (on ? "On" : "") : "");
      return `<div class="row" role="option" data-n="${n}"><span class="ic">${icon(c.icon)}</span><span class="lb">${mark(c.label, hits)}</span>`
        + `${on ? '<span class="st"></span>' : ""}<span class="ht">${hint}</span>`
        + `${c.pin ? `<button class="pn ${pins.includes(c.pin) ? "on" : ""}" data-pin="${c.pin}" title="Pin to dock">${icon("pin")}</button>` : ""}${c.key ? `<kbd>${c.key}</kbd>` : ""}</div>`;
    }).join("")).join("");
    sel = Math.min(sel, Math.max(0, rows.length - 1));
    highlight();
  }
  function highlight() {
    $("results").querySelectorAll(".row").forEach((el) => el.classList.toggle("sel", +el.dataset.n === sel));
    const el = $("results").querySelector(".row.sel");
    if (el) el.scrollIntoView({ block: "nearest" });
  }
  function run(c) {
    recent = [c.id, ...recent.filter((x) => x !== c.id)].slice(0, 6);
    store.set("recent", recent);
    closeSpot();
    c.run();
    peekDock();
  }
  function pin(id) {
    pins = pins.includes(id) ? pins.filter((x) => x !== id) : [...pins, id];
    store.set("pins", pins);
    renderDock(); renderResults();
  }
  function openSpot() {
    B.classList.add("spot", "dock-on");
    $("q").value = ""; sel = 0; renderResults();
    $("q").focus();
  }
  function closeSpot() { B.classList.remove("spot"); $("q").blur(); peekDock(1600); }
  function locate(r) {
    B.classList.remove("no-labels");
    const el = $("labels").querySelector(`[data-r="${r.id}"]`);
    if (el) { el.classList.add("focus"); setTimeout(() => el.classList.remove("focus"), 2600); }
  }
  function post(ev) {
    return fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ev) }).catch(() => null);
  }
  function fire(a) {
    a.on = a.toggle ? !a.on : true;
    if (a.id === "leak") { tickGas(); return; }
    if (!MOCK) { post(a.ev(a.on)); return; }
    const judge = (to, reason) => addEvents([{ t: S.t, from: "JUDGE", to, reason }]);
    if (a.id === "rust" && S.grid) {
      const [bi, bj] = S.brush || [20, 30];
      S.grid = S.grid.map((row, i) => row.map((v, j) => {
        const d = Math.hypot((i - bi + 6) / 3.5, ((j - bj + 13 + 64) % 64) / 4 - 1.5);
        return Math.max(v, d < 1.6 ? 0.85 * (1 - d / 1.6) + 0.1 : 0);
      }));
      judge("SPRAY", "RE-PLAN: new rust injected by judge -> spray it, extend soak");
      setState("SPRAY");
    }
    if (a.id === "glare") { $("flash").classList.toggle("on", a.on); judge("DETECT", a.on ? "glare on inspection view -> detector unsure" : "glare cleared"); }
    if (a.id === "mist") { $("pip-img").parentNode.classList.toggle("mist", a.on); judge(a.on ? "HOLD" : "VERIFY", a.on ? "perception lost (lens mist) -> hold" : "camera view restored"); setState(a.on ? "HOLD" : "VERIFY"); }
    if (a.id === "bead") { S.spike = 64.8; setTimeout(() => { S.spike = 0; }, 400); judge("BACK_OFF", "contact force 65 N > 40 N limit -> retract"); setState("BACK_OFF"); }
    if (a.id === "worker") { judge(a.on ? "PAUSE_PERSON" : "SCRUB", a.on ? "person at 2.4 m < 3.0 m -> pause" : "person clear -> resume"); setState(a.on ? "PAUSE_PERSON" : "SCRUB"); }
  }
  let side = 0;
  function shot(n) {
    S.shot = n;
    if (!MOCK) { post({ name: "camera", args: { shot: n } }); return; }
    const a = $(side ? "shotA" : "shotB"), b = $(side ? "shotB" : "shotA");
    side ^= 1;
    a.classList.remove("orbit"); a.style.transition = "none";
    a.style.backgroundImage = `url(/api/mock/shot/${n})`;
    a.style.transform = n === 1 ? "scale(1.18)" : "scale(1.35) translateY(-4%)";
    void a.offsetWidth;
    a.style.transition = ""; a.classList.add("on"); a.style.transform = "scale(1.04)";
    b.classList.remove("on");
    if (n === 1) setTimeout(() => { a.style.transform = ""; a.classList.add("orbit"); }, 4000);
  }
  let session = null, playT0 = null, cursor = 0, offset = parseFloat(Q.get("t") || "0");
  function playMock() {
    if (!session || playT0 == null) return;
    const el = (now() - playT0) * SPEED + offset, snaps = session.snapshots;
    while (cursor < snaps.length && snaps[cursor].pt <= el) {
      const s = snaps[cursor++];
      S.at = now() - (el - s.pt) / SPEED;
      apply(s);
    }
    S.at = 0;
  }
  function begin() {
    if (S.started) return;
    S.started = true;
    shot(2);
    subtitle(PHASE.PATROL);
    if (MOCK) playT0 = now();
  }
  function liveState() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/state");
    ws.onmessage = (ev) => { try { apply(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(liveState, 1500);
  }
  function keys(e) {
    const k = e.key.toLowerCase();
    if ((e.metaKey || e.ctrlKey) && k === "k") { e.preventDefault(); B.classList.contains("spot") ? closeSpot() : openSpot(); return; }
    if (B.classList.contains("spot")) {
      if (k === "escape") { e.preventDefault(); closeSpot(); }
      else if (k === "arrowdown" || k === "arrowup") { e.preventDefault(); sel = (sel + (k === "arrowdown" ? 1 : -1) + rows.length) % rows.length; highlight(); }
      else if (k === "enter" && rows[sel]) { e.preventDefault(); e.shiftKey && rows[sel].pin ? pin(rows[sel].pin) : run(rows[sel]); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (k === "/") { e.preventDefault(); openSpot(); return; }
    if (e.code === "Space") { e.preventDefault(); begin(); return; }
    if (k === "escape") { B.classList.remove("cine"); return; }
    const a = e.shiftKey && ACTIONS.find((x) => "Digit" + x.key === e.code);
    if (a) { fire(a); return; }
    const c = CMDS.find((x) => x.key && x.key.toLowerCase() === k && !x.key.startsWith("⇧"));
    if (c) { c.run(); if (c.pin) peekDock(); }
  }
  let lastGas = 0;
  function loop() {
    playMock();
    stepTweens();
    if (open.has("vitals")) {
      spark($("sp-force"), series.force, { window: VW, min: 0, max: 70, ramp: forceColor, refs: [{ v: FORCE_LIMIT, color: "rgba(248,81,73,.5)", dash: [3, 3] }] });
      spark($("sp-rust"), series.rust, { window: VW, min: 0, max: 100, color: "#D9825C", fill: "rgba(181,83,42,.14)" });
      spark($("sp-conf"), series.conf, { window: VW, min: 0, max: 1, color: "#D97757" });
    }
    if (open.has("force")) spark($("fm-chart"), series.force, { window: FW, min: 0, max: Math.max(50, S.peak * 1.15), ramp: forceColor, width: 1.6, refs: [{ v: FORCE_LIMIT, color: "rgba(248,81,73,.8)", dash: [4, 3] }] });
    if (open.has("map")) drawMap();
    const t = now();
    if (open.has("plume") && !RM) drawPlume();
    if (t - lastGas > 0.5) { lastGas = t; tickGas(); }
    requestAnimationFrame(loop);
  }
  $("spot-mag").outerHTML = icon("search");
  $("help-list").innerHTML = [["⌘K or /", "Search commands"], ["Esc", "Close / leave cinematic"], ["Space", "Start demo"], ["1 2 3", "Cameras"], ["⇧1 to ⇧6", "Scenario events"], ["C", "Cinematic mode"], ["R", "Robot labels"],
    ...PANELS.filter((p) => p.key !== "?").map((p) => [p.key.toUpperCase(), p.name])].map(([k, v]) => `<li><span>${v}</span><kbd>${k}</kbd></li>`).join("");
  buildSide(); renderDock(); renderRobots(); renderControl(); tickGas();
  $("q").addEventListener("input", () => { sel = 0; renderResults(); });
  $("results").addEventListener("mousemove", (e) => { const r = e.target.closest(".row"); if (r && +r.dataset.n !== sel) { sel = +r.dataset.n; highlight(); } });
  $("results").addEventListener("click", (e) => { const p = e.target.closest(".pn"); if (p) return pin(p.dataset.pin); const r = e.target.closest(".row"); if (r) run(rows[+r.dataset.n]); });
  $("scrim").onclick = closeSpot;
  addEventListener("keydown", keys);
  addEventListener("pointermove", (e) => { if (innerHeight - e.clientY < 72) peekDock(); });
  addEventListener("resize", fitIsland);
  $("i-state").textContent = "Standby";
  if (!Q.has("labels")) B.classList.add("no-labels");
  subtitle(Q.has("autostart") ? PHASE.PATROL : "Press <kbd>Space</kbd> to begin · <kbd>⌘K</kbd> for commands");
  fitIsland();
  if (MOCK) {
    shot(1);
    fetch("/api/mock").then((r) => r.json()).then((s) => { session = s; });
  } else liveState();
  if (Q.has("autostart")) begin();
  if (Q.has("spot")) openSpot();
  requestAnimationFrame(loop);
})();
