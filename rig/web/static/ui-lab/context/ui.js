(() => {
  const Q = new URLSearchParams(location.search);
  const MOCK = Q.has("mock");
  const SPEED = parseFloat(Q.get("speed") || "1");
  const $ = (id) => document.getElementById(id);
  const B = document.body;
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const now = () => performance.now() / 1000;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const NAME = { PATROL: "Patrol", DETECT: "Detect", SECOND_VIEW: "Second view", APPROACH: "Approach", SPRAY: "Spray", SOAK: "Soak",
    SCRUB: "Scrub", BACK_OFF: "Back off", VERIFY: "Verify", HOLD: "Hold", PAUSE_PERSON: "Person near", PAINT: "Paint",
    VERIFY_PAINT: "Verify paint", DONE: "Done", ESCALATE_TO_HUMAN: "Escalated", JUDGE: "Judge" };
  const LINE = { PATROL: "Patrolling the east deck pipe rack", DETECT: "Possible corrosion spotted, taking a closer look",
    SECOND_VIEW: "Checking from a second angle", APPROACH: "Moving in to the pipe", SPRAY: "Spraying rust converter",
    SOAK: "Letting the converter soak in", SCRUB: "Scrubbing, brush force held under 40 N", BACK_OFF: "Force spike on a weld bead, backing off",
    VERIFY: "Checking the patch is clean", HOLD: "Camera view lost, holding position", PAUSE_PERSON: "Worker within 3 m, paused",
    PAINT: "Painting the bare steel", VERIFY_PAINT: "Checking the new coat", DONE: "Repair complete, pipe protected",
    ESCALATE_TO_HUMAN: "Handing over to the control room" };
  const STEP_OF = { PATROL: 1, DETECT: 2, SECOND_VIEW: 2, APPROACH: 2, SPRAY: 3, SOAK: 4, SCRUB: 5, BACK_OFF: 5, VERIFY: 6, HOLD: 6, PAUSE_PERSON: 5, PAINT: 7, VERIFY_PAINT: 7, DONE: 7 };
  const TONE = { BACK_OFF: "warn", HOLD: "warn", PAUSE_PERSON: "stop", ESCALATE_TO_HUMAN: "stop", DONE: "ok" };
  const PERCEIVE = new Set(["DETECT", "SECOND_VIEW", "APPROACH", "VERIFY", "VERIFY_PAINT", "HOLD"]);
  const FORCE_LIMIT = 40, WINDOW = 20, DWELL = 1.6, GEMINI_FRESH = 9;
  const GAS_BASE = [["G1", "Wellhead A", 0.4], ["G2", "Separator", 0.6], ["G3", "Manifold B", 0.5], ["G4", "Compressor", 0.3], ["G5", "Quarters", 0.2]];
  const GAS_LEAK = [2.4, 7.8, 18.6, 4.1, 0.3];
  const CONTROL = { model: "Gemini 4 Argon", latency_ms: 2140,
    summary: "H₂S rising at manifold B, 18 ppm with a 2.3 m/s north-east wind. Likely source is the XV-204 flange. Atlas is 14 m downwind.",
    actions: [["Pause Atlas and retreat upwind", 0.92], ["Send Kestrel to localise the leak", 0.81], ["Close XV-204 from the control room", 0.64], ["Muster deck crew at lifeboat B", 0.38]], chosen: 0 };
  const ROBOTS = [
    { id: "atlas", name: "Atlas", kind: "Rust repair quadruped", at: { 1: [0.47, 0.3], 2: [0.6, 0.47], 3: [0.52, 0.34] } },
    { id: "kestrel", name: "Kestrel", kind: "Gas sniffer", at: { 1: [0.6, 0.31], 2: [0.24, 0.55] } },
    { id: "tern", name: "Tern", kind: "Inspection drone", at: { 1: [0.33, 0.19], 2: [0.66, 0.14] } },
  ];
  const store = {
    get(k, d) { try { const v = localStorage.getItem("ctx." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("ctx." + k, JSON.stringify(v)); } catch {} },
  };
  const S = { state: null, t: 0, force: 0, peak: 0, rust: null, conf: null, soak: null, grid: null, sprayed: null, painted: null, brush: null,
    insp: null, gemini: null, gemAt: -99, log: [], gas: GAS_BASE.map(([id, at, ppm]) => ({ id, at, ppm, to: ppm })), gasOn: false,
    control: null, shot: 2, started: false, robots: null, subtitle: null, ev: {} };
  const series = { force: [], rust: [], conf: [] };
  const pins = new Set(store.get("pins", []));
  let recent = store.get("recent", []);
  let auto = store.get("auto", true), presenter = false, slot = null, slotAt = -99;

  const ICON = {
    gauge: "M4 15a8 8 0 1 1 16 0M12 15l4-5", timer: "M12 8v5l3 2M9 2h6M12 22a9 9 0 1 0 0-18 9 9 0 0 0 0 18z",
    pulse: "M3 12h4l3-7 4 14 3-7h4", grid: "M3 5h18v14H3zM3 12h18M9 5v14M15 5v14", camera: "M4 7h3l2-2h6l2 2h3v12H4zM12 16a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    spark: "M12 3l2 6 6 2-6 2-2 6-2-6-6-2 6-2z", wave: "M3 15c2-4 4-4 6 0s4 4 6 0 4-4 6 0M3 9c2-4 4-4 6 0s4 4 6 0 4-4 6 0",
    cloud: "M7 18h10a4 4 0 0 0 0-8 6 6 0 0 0-11.5 1.5A3.5 3.5 0 0 0 7 18z", monitor: "M4 5h16v11H4zM8 20h8M12 16v4",
    list: "M9 6h11M9 12h11M9 18h11M4 6h.01M4 12h.01M4 18h.01", robot: "M6 9h12v9H6zM12 5v4M9.5 13.5h.01M14.5 13.5h.01",
    stack: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5", text: "M5 6h14M12 6v13M9 19h6", tag: "M3 12V4h8l10 10-8 8zM7.5 8.5h.01",
    wand: "M15 4v3M13.5 5.5h3M5 19L16 8l2 2L7 21z", pin: "M9 3h6l-1 6 4 4H6l4-4zM12 13v8", video: "M3 6h12v12H3zM15 10l6-3v10l-6-3z",
    bolt: "M13 2L4 14h7l-1 8 9-12h-7z", play: "M7 4l13 8-13 8z", eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z",
    keys: "M3 6h18v12H3zM7 10h.01M11 10h.01M15 10h.01M7 14h10", sun: "M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5L19 19M5 19l1.5-1.5M17.5 6.5L19 5",
    person: "M12 8a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM6 22v-6a6 6 0 0 1 12 0v6", plus: "M12 5v14M5 12h14",
  };
  const svg = (k) => `<svg viewBox="0 0 24 24"><path d="${ICON[k]}"/></svg>`;

  const PANELS = [
    ["force", "Brush force", "gauge", "newton sensor mod scrub"], ["soak", "Soak timer", "timer", "ring converter wait"],
    ["insp", "Inspection camera", "camera", "pip picture view"], ["gemini", "Gemini second opinion", "spark", "robotics er verdict"],
    ["gas", "Gas sensors", "wave", "h2s ppm leak alarm"], ["plume", "Gas plume", "cloud", "pip dispersion wind map"],
    ["control", "Control centre", "monitor", "argon situation report actions"], ["vitals", "Vitals", "pulse", "sparkline rust confidence"],
    ["map", "Pipe map", "grid", "heatmap rust coverage"], ["log", "Decision log", "list", "events history"],
    ["fleet", "Fleet", "robot", "robots status battery"], ["modules", "Modules", "stack", "health fps connection"],
  ];
  function pick() {
    const gasMax = Math.max(...S.gas.map((g) => g.ppm));
    if (S.gasOn || gasMax > 10) return S.control ? "control" : "gas";
    if (S.state === "ESCALATE_TO_HUMAN") return "control";
    if (now() - S.gemAt < GEMINI_FRESH / SPEED) return "gemini";
    if (S.state === "SCRUB" || S.state === "BACK_OFF") return "force";
    if (S.state === "SOAK") return "soak";
    if (PERCEIVE.has(S.state) && S.insp) return "insp";
    if (S.state === "PAUSE_PERSON") return "log";
    if (S.state === "DONE") return "map";
    return null;
  }
  let shownKey = "";
  const shown = new Set();
  function sync() {
    const want = pick();
    if (want !== slot && now() - slotAt > DWELL / SPEED) { slot = want; slotAt = now(); }
    shown.clear();
    if (!presenter) {
      pins.forEach((id) => shown.add(id));
      if (auto && slot) shown.add(slot);
      if (S.helpOpen) shown.add("help");
    }
    const key = [...shown].join() + "|" + [...pins].join();
    if (key === shownKey) return;
    shownKey = key;
    document.querySelectorAll(".card").forEach((c) => {
      const id = c.dataset.panel;
      c.classList.toggle("show", shown.has(id));
      c.classList.toggle("pinned", pins.has(id));
    });
    document.querySelectorAll(".dock").forEach((d) => d.classList.toggle("on", !!d.querySelector(".card.show")));
  }
  function togglePin(id) {
    pins.has(id) ? pins.delete(id) : pins.add(id);
    store.set("pins", [...pins]); sync();
  }
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
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return [g, w, h];
  }
  function spark(c, data, o) {
    const [g, w, h] = fit(c);
    const t1 = now(), t0 = t1 - WINDOW;
    const x = (t) => ((t - t0) / WINDOW) * w, y = (v) => h - 2 - ((Math.min(o.max, v) - o.min) / (o.max - o.min)) * (h - 4);
    if (o.limit != null) {
      g.strokeStyle = "rgba(248,81,73,.55)"; g.setLineDash([3, 3]); g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, Math.round(y(o.limit)) + 0.5); g.lineTo(w, Math.round(y(o.limit)) + 0.5); g.stroke(); g.setLineDash([]);
    }
    const pts = data.filter((p) => p[0] >= t0 - 1);
    if (pts.length < 2) return;
    g.lineWidth = 1.5; g.lineJoin = g.lineCap = "round"; g.strokeStyle = o.color;
    g.beginPath();
    pts.forEach((p, k) => (k ? g.lineTo(x(p[0]), y(p[1])) : g.moveTo(x(p[0]), y(p[1]))));
    g.stroke();
    if (o.fill) { g.lineTo(x(pts[pts.length - 1][0]), h); g.lineTo(x(pts[0][0]), h); g.fillStyle = o.fill; g.fill(); }
  }
  function push(k, v) {
    if (v == null) return;
    const s = series[k], t = S.at || now();
    s.push([t, v]);
    while (s.length && s[0][0] < t - WINDOW - 2) s.shift();
  }
  const forceTone = (f) => (f > FORCE_LIMIT ? "#F85149" : f > 36 ? "#E3B341" : "#D97757");
  function drawMap() {
    const [g, w, h] = fit($("map-canvas"));
    g.fillStyle = "rgba(0,0,0,.28)"; g.fillRect(0, 0, w, h);
    if (!S.grid) return;
    const R = S.grid.length, C = S.grid[0].length, bb = (S.bb = S.bb || [R, C, 0, 0]);
    S.grid.forEach((row, i) => row.forEach((v, j) => { if (v > 0.005 || (S.sprayed && S.sprayed[i][j] === "1")) { bb[0] = Math.min(bb[0], i); bb[1] = Math.min(bb[1], j); bb[2] = Math.max(bb[2], i); bb[3] = Math.max(bb[3], j); } }));
    const cols = Math.min(C, Math.max(bb[3] - bb[1] + 13, 24)), rows = Math.min(R, Math.round((cols * h) / w));
    const i0 = Math.max(0, Math.min(R - rows, Math.round((bb[0] + bb[2] - rows) / 2))), j0 = Math.max(0, Math.min(C - cols, Math.round((bb[1] + bb[3] - cols) / 2)));
    const cw = w / cols, ch = h / rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const r = S.grid[i0 + i][j0 + j], at = (m) => m && m[i0 + i][j0 + j] === "1";
      const col = at(S.painted) ? (r > 0.08 ? `rgba(217,119,87,${0.35 + r * 0.6})` : "rgba(143,168,148,.85)") : r > 0.005 ? `rgba(217,119,87,${0.3 + r * 0.7})` : at(S.sprayed) ? "rgba(235,235,245,.14)" : null;
      if (col) { g.fillStyle = col; g.fillRect(j * cw, i * ch, cw - 1, ch - 1); }
    }
    if (S.brush) {
      g.strokeStyle = "#fff"; g.lineWidth = 1.5;
      g.beginPath(); g.arc((S.brush[1] - j0 + 0.5) * cw, (S.brush[0] - i0 + 0.5) * ch, 5, 0, 7); g.stroke();
    }
  }
  const PLUME_AT = [[0.18, 0.3], [0.42, 0.62], [0.62, 0.42], [0.82, 0.7], [0.88, 0.2]];
  function drawPlume() {
    const c = $("plume"), g = c.getContext("2d"), w = c.width, h = c.height, t = now();
    g.globalCompositeOperation = "source-over";
    g.fillStyle = "#121315"; g.fillRect(0, 0, w, h);
    g.strokeStyle = "rgba(255,255,255,.08)"; g.lineWidth = 1;
    g.strokeRect(w * 0.08 + 0.5, h * 0.12 + 0.5, w * 0.84, h * 0.76);
    g.globalCompositeOperation = "lighter";
    S.gas.forEach((s, k) => {
      const [px, py] = PLUME_AT[k], strength = Math.min(1, s.ppm / 20);
      if (strength < 0.03) return;
      for (let n = 0; n < 4; n++) {
        const d = n * 0.11 + ((t * 0.04 + k * 0.13) % 0.11);
        const x = (px + d * 0.9 + Math.sin(t * 0.7 + n) * 0.01) * w, y = (py - d * 0.6) * h, r = (0.08 + d * 0.9) * w * (0.4 + strength);
        const grd = g.createRadialGradient(x, y, 0, x, y, r);
        grd.addColorStop(0, `rgba(248,81,73,${0.42 * strength * (1 - n / 4)})`);
        grd.addColorStop(1, "rgba(248,81,73,0)");
        g.fillStyle = grd; g.fillRect(x - r, y - r, 2 * r, 2 * r);
      }
    });
    g.globalCompositeOperation = "source-over";
    S.gas.forEach((s, k) => {
      const [px, py] = PLUME_AT[k];
      g.fillStyle = s.ppm > 10 ? "#F85149" : s.ppm > 5 ? "#E3B341" : "rgba(255,255,255,.7)";
      g.beginPath(); g.arc(px * w, py * h, 2.5, 0, 7); g.fill();
      g.fillStyle = "rgba(235,235,245,.55)"; g.font = "10px system-ui"; g.fillText(s.id, px * w + 6, py * h + 3);
    });
  }
  function renderGas() {
    $("gas-list").innerHTML = S.gas.map((s) => {
      const tone = s.ppm > 10 ? "stop" : s.ppm > 5 ? "warn" : "";
      return `<li class="${tone}"><span>${s.at}</span><span class="bar"><i style="width:${Math.min(100, s.ppm * 5)}%"></i></span><b class="num">${s.ppm.toFixed(1)}</b></li>`;
    }).join("");
  }
  function renderControl() {
    const c = S.control;
    $("cc-sum").textContent = c ? c.summary : "No incident.";
    $("cc-acts").innerHTML = c ? c.actions.map(([a, s], k) => `<li class="${k === c.chosen ? "chosen" : ""}"><span>${esc(a)}</span><b class="num">${s.toFixed(2)}</b></li>`).join("") : "";
  }
  function robotTask(r) {
    if (r.id === "atlas") return S.gasOn ? "Retreating upwind" : NAME[S.state] || "Standby";
    if (r.id === "kestrel") return S.gasOn ? (S.control ? "Localising leak" : "Re-tasking") : "Gas sweep";
    return "Overwatch";
  }
  function renderFleet() {
    $("fleet-list").innerHTML = ROBOTS.map((r) => `<li><span>${r.name}</span><em>${robotTask(r)}</em></li>`).join("");
    const mods = [["Perception", "Edge", ""], ["Gemini Robotics-ER", "Cloud", ""], ["Planner", "Edge", ""], ["Force sensor", S.state === "BACK_OFF" ? "Over limit" : "OK", S.state === "BACK_OFF" ? "warn" : ""],
      ["Gas array", S.gasOn ? "Alarm" : "OK", S.gasOn ? "stop" : ""]];
    $("mod-list").innerHTML = mods.map(([n, v, t]) => `<li><span class="st ${t}"></span><span>${n}</span><em>${v}</em></li>`).join("");
  }
  function renderLabels() {
    const el = $("labels");
    const list = S.robots || (MOCK ? ROBOTS.filter((r) => r.at[S.shot]).map((r) => ({ id: r.id, name: r.name, x: r.at[S.shot][0], y: r.at[S.shot][1] })) : []);
    el.innerHTML = list.map((r) => `<div class="rl${r.id === "atlas" ? " lead" : ""}${r.id === S.focus ? " focus" : ""}" style="left:${r.x * 100}%;top:${r.y * 100}%"><b>${esc(r.name)}</b><i></i><s></s></div>`).join("");
  }
  let subKey = "";
  function renderStatus() {
    const gasMax = Math.max(...S.gas.map((g) => g.ppm));
    const alarm = S.gasOn && gasMax > 10;
    const tone = alarm ? "stop" : TONE[S.state] || "";
    $("capsule").className = "glass " + tone;
    $("c-state").textContent = alarm ? "Gas alarm" : NAME[S.state] || "Standby";
    let key = "";
    if (alarm) key = `${gasMax.toFixed(1)} ppm`;
    else if (S.state === "SCRUB" || S.state === "BACK_OFF") key = `${S.force.toFixed(1)} N`;
    else if (S.soak) key = `${Math.ceil(S.soak.left)} s left`;
    else if (S.rust != null) key = `${(S.rust * 100).toFixed(1)}% rust`;
    $("c-key").textContent = key;
    B.classList.toggle("alarm", alarm || tone === "stop");
    B.classList.toggle("caution", !alarm && tone === "warn");
    let k = S.started ? `Rust repair · step ${STEP_OF[S.state] || 1} of 7` : "Offshore autonomy · live sim";
    let line = S.subtitle || (S.started ? LINE[S.state] || "" : "Press Space to begin");
    if (S.gasOn) { k = "Gas alarm · Manifold B"; line = S.control ? "Atlas retreating upwind, Kestrel localising the leak" : "H₂S detected, control centre assessing"; }
    if (k + line !== subKey) {
      subKey = k + line;
      $("sub-t").classList.add("swap");
      setTimeout(() => { $("sub-k").textContent = k; $("sub-t").textContent = line; $("sub-t").classList.remove("swap"); }, RM ? 0 : 200);
    }
  }
  function setState(st) {
    if (st === S.state) return;
    S.state = st;
    if (st === "DONE" && MOCK) setTimeout(() => shot(1), 3500);
    renderLabels();
  }
  function addEvents(events) {
    if (!events.length) return;
    S.log = [...events.slice().reverse(), ...S.log].slice(0, 5);
    $("log-list").innerHTML = S.log.map((e) => `<li><span class="num">${Number(e.t).toFixed(1)}</span><span><strong>${esc(NAME[e.to] || e.to)}</strong> ${esc(e.reason || "")}</span></li>`).join("");
  }
  function showGemini(gm) {
    const key = JSON.stringify(gm);
    if (!gm || key === S.gemKey) return;
    S.gemKey = key;
    S.gemAt = now();
    const v = gm.verdict || "cannot_tell";
    $("g-verdict").className = "chip " + v;
    $("g-verdict").textContent = { rust: "Rust", clean: "Clean", cannot_tell: "Can't tell" }[v] || v;
    $("g-lat").textContent = `Gemini Robotics-ER · ${Math.round(gm.latency_ms || 0)} ms`;
    num($("g-conf"), gm.confidence || 0, 2);
    $("g-reason").textContent = gm.reason || "";
    if (S.insp) {
      $("g-img").src = "data:image/jpeg;base64," + S.insp.jpg;
      const p = $("g-pts"), r = S.insp.w / 40;
      p.setAttribute("viewBox", `0 0 ${S.insp.w} ${S.insp.h}`);
      p.setAttribute("preserveAspectRatio", "xMidYMid slice");
      p.innerHTML = (gm.points || []).map(([x, y]) => `<circle cx="${x}" cy="${y}" r="${r}" stroke-width="${r / 4}"/>`).join("");
    }
  }

  function apply(h) {
    if (h.state) setState(h.state);
    if (h.t != null) { S.t = h.t; num($("c-time"), h.t, 1); }
    if (h.events) addEvents(h.events);
    if (h.rust_grid) S.grid = h.rust_grid;
    ["sprayed", "painted", "brush"].forEach((k) => { if (h[k]) S[k] = h[k]; });
    if (h.force_n != null) {
      const f = S.spike ? Math.max(h.force_n, S.spike) : h.force_n;
      S.force = f; S.peak = Math.max(S.peak, f);
      push("force", f);
      num($("f-now"), f, 1); num($("f-max"), S.peak, 1); num($("v-force"), f, 1);
      $("f-now").style.color = forceTone(f);
      $("f-gauge").style.width = Math.min(100, (f / 50) * 100) + "%";
      $("f-gauge").style.background = forceTone(f);
    }
    if (h.rust_fraction != null) { S.rust = h.rust_fraction; push("rust", h.rust_fraction * 100); num($("v-rust"), h.rust_fraction * 100, 1); }
    if (h.detector_conf != null) { S.conf = h.detector_conf; push("conf", h.detector_conf); num($("v-conf"), h.detector_conf, 2); }
    if (h.inspection) { S.insp = h.inspection; $("pip-img").src = "data:image/jpeg;base64," + h.inspection.jpg; }
    if (h.gemini) showGemini(h.gemini);
    if ("soak" in h) S.soak = h.soak;
    if (S.soak) { num($("soak-left"), S.soak.left, 0); $("soak-ring").style.strokeDashoffset = String(119.4 * (1 - S.soak.left / S.soak.total)); }
    if (h.gas) { S.gas = h.gas.sensors.map((s) => ({ ...s, to: s.ppm })); S.gasOn = !!h.gas.alarm; }
    if ("control" in h) { S.control = h.control; renderControl(); }
    if (h.robots) { S.robots = h.robots; renderLabels(); }
    if ("subtitle" in h) S.subtitle = h.subtitle;
  }
  function post(ev) {
    return fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ev) }).then((r) => r.json()).catch(() => null);
  }
  const gasTimers = [];
  function gasLeak(on) {
    gasTimers.splice(0).forEach(clearTimeout);
    S.gasOn = S.ev.gas = on;
    S.gas.forEach((s, k) => { s.to = on ? GAS_LEAK[k] : GAS_BASE[k][2]; });
    S.control = null; renderControl();
    if (on) {
      gasTimers.push(setTimeout(() => { S.control = CONTROL; renderControl(); }, 2600 / SPEED));
      gasTimers.push(setTimeout(() => gasLeak(false), 18000 / SPEED));
    }
  }
  function fire(id) {
    const on = (S.ev[id] = !S.ev[id]);
    const ev = { rust: { name: "spawn_patch", args: {} }, glare: { name: "set_glare", args: { on } }, mist: { name: "occlude_camera", args: { on } },
      bead: { name: "force_spike", args: {} }, worker: { name: "person_at", args: { distance: on ? 2.4 : null } }, gas: { name: "gas_leak", args: { on } } }[id];
    if (!MOCK) { post(ev); return; }
    const judge = (to, reason) => addEvents([{ t: S.t, from: "JUDGE", to, reason }]);
    if (id === "rust") { judge("SPRAY", "RE-PLAN: new rust injected, spray it and extend soak"); setState("SPRAY"); }
    if (id === "glare") { $("flash").classList.toggle("on", on); judge("DETECT", on ? "glare on inspection view" : "glare cleared"); }
    if (id === "mist") { $("media").classList.toggle("mist", on); judge(on ? "HOLD" : "VERIFY", on ? "perception lost, hold position" : "camera view restored"); if (on) setState("HOLD"); }
    if (id === "bead") { S.spike = 64.8; setTimeout(() => { S.spike = 0; }, 400); judge("BACK_OFF", "contact force 65 N over 40 N limit, retract"); setState("BACK_OFF"); }
    if (id === "worker") { judge(on ? "PAUSE_PERSON" : "SCRUB", on ? "person at 2.4 m, pause" : "person clear, resume"); if (on) setState("PAUSE_PERSON"); }
    if (id === "gas") { gasLeak(on); judge("HOLD", on ? "H₂S 18 ppm at manifold B, fleet re-tasked" : "gas cleared"); }
  }
  let shotSide = 0;
  function shot(n) {
    S.shot = n;
    if (!MOCK) { post({ name: "camera", args: { shot: n } }); return; }
    const a = $(shotSide ? "shotA" : "shotB"), b = $(shotSide ? "shotB" : "shotA");
    shotSide ^= 1;
    a.classList.remove("orbit");
    Object.assign(a.style, { transition: "none", backgroundImage: `url(/api/mock/shot/${n})`, transform: n === 1 ? "scale(1.18)" : "scale(1.35) translateY(-4%)" });
    void a.offsetWidth;
    Object.assign(a.style, { transition: "", transform: "scale(1.04)" });
    a.classList.add("on");
    b.classList.remove("on");
    if (n === 1) setTimeout(() => { a.style.transform = ""; a.classList.add("orbit"); }, 4000);
    renderLabels();
  }
  let session = null, playT0 = null, cursor = 0, offset = parseFloat(Q.get("t") || "0");
  function playMock() {
    if (!session || playT0 == null) return;
    const el = (now() - playT0) * SPEED + offset, snaps = session.snapshots;
    while (cursor < snaps.length && snaps[cursor].pt <= el) apply(snaps[cursor++]);
  }
  function seek(pt) {
    const snaps = session.snapshots, back = now();
    while (cursor < snaps.length && snaps[cursor].pt <= pt) { const s = snaps[cursor++]; S.at = back - (pt - s.pt) / SPEED; apply(s); }
    S.at = 0;
  }
  function begin() {
    if (S.started) return;
    S.started = true;
    B.classList.remove("intro");
    shot(2);
    if (MOCK) { playT0 = now(); if (offset && session) seek(offset); }
  }
  function setPresenter(on) { presenter = on; B.classList.toggle("presenter", on); shownKey = ""; sync(); }
  function focusRobot(id) {
    S.focus = id; B.classList.remove("no-labels"); renderLabels();
    setTimeout(() => { if (S.focus === id) { S.focus = null; renderLabels(); } }, 3500);
  }
  const scrubbing = () => !S.gasOn && (S.state === "SCRUB" || S.state === "BACK_OFF");
  const CMDS = [
    ...PANELS.map(([id, label, icon, keys]) => ({ id: "p." + id, group: "Panels", label, icon, keys, panel: id, run: () => togglePin(id) })),
    { id: "o.sub", group: "Panels", label: "Phase subtitle", icon: "text", keys: "caption", on: () => !B.classList.contains("no-sub"), run: () => B.classList.toggle("no-sub") },
    { id: "o.labels", group: "Panels", label: "Robot labels", icon: "tag", keys: "names floating", on: () => !B.classList.contains("no-labels"), run: () => B.classList.toggle("no-labels") },
    { id: "o.auto", group: "Panels", label: "Context panels", icon: "wand", keys: "auto surface automatic", on: () => auto, run: () => { auto = !auto; store.set("auto", auto); shownKey = ""; sync(); } },
    { id: "o.unpin", group: "Panels", label: "Unpin all panels", icon: "pin", keys: "clear reset", ctx: () => (pins.size > 2 ? 0.3 : 0), run: () => { pins.clear(); store.set("pins", []); sync(); } },
    { id: "c.1", group: "Cameras", label: "Wide rig", icon: "video", hint: "1", keys: "aerial shot", ctx: () => (S.state === "DONE" ? 0.6 : 0), run: () => shot(1) },
    { id: "c.2", group: "Cameras", label: "Hero deck", icon: "video", hint: "2", keys: "shot", run: () => shot(2) },
    { id: "c.3", group: "Cameras", label: "Repair close-up", icon: "video", hint: "3", keys: "shot zoom", ctx: () => (scrubbing() || S.state === "SPRAY" ? 0.55 : 0), run: () => shot(3) },
    { id: "e.rust", group: "Scenario events", label: "Inject new rust", icon: "plus", hint: "⇧1", keys: "patch spawn replan", ctx: () => (S.state === "SOAK" ? 0.5 : 0), run: () => fire("rust") },
    { id: "e.glare", group: "Scenario events", label: "Sun glare", icon: "sun", hint: "⇧2", keys: "flash light", on: () => !!S.ev.glare, run: () => fire("glare") },
    { id: "e.mist", group: "Scenario events", label: "Lens mist", icon: "cloud", hint: "⇧3", keys: "occlude camera fog", on: () => !!S.ev.mist, ctx: () => (S.state === "VERIFY" ? 0.6 : 0), run: () => fire("mist") },
    { id: "e.bead", group: "Scenario events", label: "Weld bead force spike", icon: "bolt", hint: "⇧4", keys: "back off newton", ctx: () => (scrubbing() ? 0.8 : 0), run: () => fire("bead") },
    { id: "e.worker", group: "Scenario events", label: "Worker walks up", icon: "person", hint: "⇧5", keys: "person pause human", on: () => !!S.ev.worker, ctx: () => (scrubbing() ? 0.4 : 0), run: () => fire("worker") },
    { id: "e.gas", group: "Scenario events", label: "Gas leak at manifold B", icon: "wave", hint: "⇧6", keys: "h2s alarm plume", on: () => S.gasOn, ctx: () => (S.started && !S.gasOn ? 0.35 : 0), run: () => fire("gas") },
    { id: "d.start", group: "Demo", label: "Start demo", icon: "play", hint: "Space", keys: "begin play", ctx: () => (S.started ? 0 : 1), run: begin },
    { id: "d.presenter", group: "Demo", label: "Presenter mode", icon: "eye", hint: "P", keys: "clean video capture record hide", on: () => presenter, ctx: () => (presenter ? 1 : 0.2), run: () => setPresenter(!presenter) },
    ...ROBOTS.map((r) => ({ id: "r." + r.id, group: "Robots", label: r.name, icon: "robot", keys: r.kind, hintFn: () => robotTask(r), run: () => focusRobot(r.id) })),
    { id: "h.keys", group: "Help", label: "Keyboard shortcuts", icon: "keys", keys: "help controls", hint: "?", on: () => !!S.helpOpen, run: () => { S.helpOpen = !S.helpOpen; sync(); } },
  ];
  const GROUPS = ["Panels", "Cameras", "Scenario events", "Demo", "Robots", "Help"];
  const PANEL_CTX = { plume: () => (S.gasOn ? 0.8 : 0), gas: () => (S.gasOn ? 0.5 : 0), map: () => (S.state === "PAINT" ? 0.4 : 0) };
  const ctxScore = (c) => (c.panel ? (pins.has(c.panel) ? 0 : slot === c.panel && auto ? 0.9 : PANEL_CTX[c.panel] ? PANEL_CTX[c.panel]() : 0) : c.ctx ? c.ctx() : 0);
  function fuzzy(q, s) {
    s = s.toLowerCase();
    let i = 0, score = 0, last = -2;
    for (let k = 0; k < s.length && i < q.length; k++) {
      if (s[k] !== q[i]) continue;
      score += (k === last + 1 ? 3 : 1) + (k === 0 || s[k - 1] === " " ? 3 : 0);
      last = k; i++;
    }
    return i === q.length ? score / (q.length * 7) + (s.startsWith(q) ? 0.6 : 0) : 0;
  }
  function results(q) {
    if (!q) {
      const sug = CMDS.map((c) => [c, ctxScore(c)]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 4).map((x) => x[0]);
      const rec = recent.map((id) => CMDS.find((c) => c.id === id)).filter((c) => c && !sug.includes(c)).slice(0, 3);
      return [["Suggested now", sug], ["Recent", rec], ...GROUPS.map((g) => [g, CMDS.filter((c) => c.group === g && !sug.includes(c) && !rec.includes(c))])];
    }
    const hits = CMDS.map((c) => [c, Math.max(fuzzy(q, c.label) * 1.4, fuzzy(q, c.keys || "") * 0.7, fuzzy(q, c.group) * 0.5)]).filter((x) => x[1] > 0.25)
      .map(([c, s]) => [c, s + ctxScore(c) * 0.6]).sort((a, b) => b[1] - a[1]).map((x) => x[0]);
    const sug = hits.filter((c) => ctxScore(c) > 0);
    const out = sug.length ? [["Suggested now", sug]] : hits.length ? [["Top hit", [hits[0]]]] : [];
    const used = new Set(out[0] ? out[0][1] : []);
    const rest = hits.filter((c) => !used.has(c)), order = [...new Set(rest.map((c) => c.group))];
    return [...out, ...order.map((g) => [g, rest.filter((c) => c.group === g)])];
  }
  let flat = [], sel = 0;
  function hintOf(c) {
    if (c.panel) return slot === c.panel && auto && !presenter ? ["now", "Showing now"] : ["", ""];
    return ["", c.hintFn ? c.hintFn() : c.hint || ""];
  }
  function mark(label, q) {
    const k = q ? label.toLowerCase().indexOf(q) : -1;
    return k < 0 ? esc(label) : esc(label.slice(0, k)) + "<u>" + esc(label.slice(k, k + q.length)) + "</u>" + esc(label.slice(k + q.length));
  }
  function renderRes() {
    const q = $("q").value.trim().toLowerCase();
    flat = [];
    $("res").innerHTML = results(q).filter((g) => g[1].length).map(([name, list]) => `<div class="grp">${name}</div>` + list.map((c) => {
      const i = flat.push(c) - 1, [cls, hint] = hintOf(c);
      const on = c.panel ? pins.has(c.panel) : c.on ? c.on() : null;
      return `<div class="row" role="option" data-i="${i}"><span class="ic">${svg(c.icon)}</span><span class="lb">${mark(c.label, q)}</span><span class="ht ${cls}">${esc(hint)}</span>${on == null ? "" : `<span class="sw${on ? " on" : ""}"></span>`}</div>`;
    }).join("")).join("");
    sel = Math.min(sel, flat.length - 1);
    highlight();
  }
  function highlight() {
    $("res").querySelectorAll(".row").forEach((r) => r.classList.toggle("sel", +r.dataset.i === sel));
    const r = $("res").querySelector(".row.sel");
    if (r) r.scrollIntoView({ block: "nearest" });
  }
  function run(c, keep) {
    c.run();
    recent = [c.id, ...recent.filter((x) => x !== c.id)].slice(0, 5);
    store.set("recent", recent);
    keep ? renderRes() : closeSpot();
  }
  function openSpot() { $("spot").hidden = false; $("q").value = ""; sel = 0; renderRes(); $("q").focus(); }
  function closeSpot() { $("spot").hidden = true; $("q").blur(); }
  $("q").addEventListener("input", () => { sel = 0; renderRes(); });
  $("q").addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); sel = (sel + (e.key === "ArrowDown" ? 1 : -1) + flat.length) % flat.length; highlight(); }
    else if (e.key === "Enter" && flat[sel]) { e.preventDefault(); run(flat[sel], e.metaKey || e.ctrlKey); }
    else if (e.key === "Escape") { e.preventDefault(); closeSpot(); }
  });
  $("res").addEventListener("mousemove", (e) => { const r = e.target.closest(".row"); if (r && +r.dataset.i !== sel) { sel = +r.dataset.i; highlight(); } });
  $("res").addEventListener("click", (e) => { const r = e.target.closest(".row"); if (r) run(flat[+r.dataset.i], !!e.target.closest(".sw")); });
  $("spot").addEventListener("mousedown", (e) => { if (e.target === $("spot")) closeSpot(); });
  $("capsule").onclick = openSpot;
  document.querySelectorAll(".card .pin").forEach((b) => { b.onclick = () => togglePin(b.closest(".card").dataset.panel); });
  $("help-list").innerHTML = [["⌘K  /", "Commands"], ["Space", "Start demo"], ["1 2 3", "Cameras"], ["⇧1–6", "Scenario events"], ["P", "Presenter mode"], ["Esc", "Close"]]
    .map(([k, v]) => `<li><span>${v}</span><kbd>${k}</kbd></li>`).join("");

  addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); $("spot").hidden ? openSpot() : closeSpot(); return; }
    if (e.target.tagName === "INPUT") return;
    const k = e.key.toLowerCase();
    if (k === "/") { e.preventDefault(); openSpot(); return; }
    if (k === "escape") { S.helpOpen = false; sync(); return; }
    if (k === "?") { S.helpOpen = !S.helpOpen; sync(); return; }
    if (e.code === "Space") { e.preventDefault(); begin(); return; }
    if (k === "p") { setPresenter(!presenter); return; }
    const d = e.code.replace("Digit", "");
    if (e.shiftKey && "123456".includes(d) && d.length === 1) { fire(["rust", "glare", "mist", "bead", "worker", "gas"][+d - 1]); return; }
    if (["1", "2", "3"].includes(k)) shot(+k);
  });
  let lastSlow = 0;
  function loop() {
    playMock();
    stepTweens();
    if (MOCK) S.gas.forEach((s) => { s.ppm += (s.to - s.ppm) * 0.025 + (Math.random() - 0.5) * 0.02 * s.to; s.ppm = Math.max(0, s.ppm); });
    sync();
    if (shown.has("force")) spark($("f-chart"), series.force, { min: 0, max: 70, color: "#D97757", limit: FORCE_LIMIT, fill: "rgba(217,119,87,.08)" });
    if (shown.has("vitals")) {
      spark($("sp-force"), series.force, { min: 0, max: 70, color: "#D97757" });
      spark($("sp-rust"), series.rust, { min: 0, max: 100, color: "rgba(235,235,245,.6)" });
      spark($("sp-conf"), series.conf, { min: 0, max: 1, color: "rgba(235,235,245,.6)" });
    }
    if (shown.has("plume")) drawPlume();
    const t = now();
    if (t - lastSlow > 0.25) {
      lastSlow = t;
      renderStatus();
      if (shown.has("gas")) renderGas();
      if (shown.has("fleet") || shown.has("modules")) renderFleet();
      if (shown.has("map")) drawMap();
    }
    requestAnimationFrame(loop);
  }
  if (!MOCK) S.started = true;
  else B.classList.add("intro");
  renderControl();
  renderLabels();
  if (MOCK) {
    $("status").textContent = "mock replay";
    shot(1);
    fetch("/api/mock").then((r) => r.json()).then((s) => { session = s; if (playT0 != null && offset) seek(offset); });
  } else {
    const ws = () => {
      const sock = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/state");
      sock.onmessage = (ev) => { try { apply(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
      sock.onclose = () => setTimeout(ws, 1500);
    };
    ws();
  }
  if (Q.has("autostart")) begin();
  if (Q.has("pin")) Q.get("pin").split(",").forEach((id) => pins.add(id));
  if (Q.has("presenter")) setPresenter(true);
  window.ui = { S, openSpot, fire, togglePin, setPresenter, shot };
  requestAnimationFrame(loop);
})();
