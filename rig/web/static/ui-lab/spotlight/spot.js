(() => {
  const Q = new URLSearchParams(location.search), MOCK = Q.has("mock"), SPEED = +(Q.get("speed") || 1);
  const $ = (id) => document.getElementById(id);
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const MAC = /Mac|iP/.test(navigator.platform);
  const now = () => performance.now() / 1000;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const set = (el, v) => { if (el.textContent !== v) el.textContent = v; };
  const fmt = (v, d, u) => (v == null ? "—" : v.toFixed(d) + u);
  const store = {
    get(k, d) { try { const v = localStorage.getItem("spot." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("spot." + k, JSON.stringify(v)); } catch {} },
  };
  const STEPS = ["Patrol", "Detect", "Spray", "Soak", "Scrub", "Verify", "Paint", "Done"];
  const STEP_OF = { PATROL: 0, DETECT: 1, SECOND_VIEW: 1, APPROACH: 1, SPRAY: 2, SOAK: 3, SCRUB: 4, VERIFY: 5, PAINT: 6, VERIFY_PAINT: 6, DONE: 7 };
  const SAFETY = { BACK_OFF: "warn", HOLD: "warn", PAUSE_PERSON: "stop", ESCALATE_TO_HUMAN: "stop" };
  const NAME = { PATROL: "Patrol", DETECT: "Detect", SECOND_VIEW: "Second view", APPROACH: "Approach", SPRAY: "Spray", SOAK: "Soak", SCRUB: "Scrub", BACK_OFF: "Back off", VERIFY: "Verify", HOLD: "Hold", PAUSE_PERSON: "Paused", PAINT: "Paint", VERIFY_PAINT: "Verify paint", DONE: "Done", ESCALATE_TO_HUMAN: "Escalated", JUDGE: "Judge", ACTION: "Action", GAS: "Gas" };
  const PHRASE = { PATROL: "Patrolling the lower deck", DETECT: "Looking for corrosion", SECOND_VIEW: "Taking a second look", APPROACH: "Moving in on the patch", SPRAY: "Spraying rust converter", SOAK: "Letting the converter soak", SCRUB: "Scrubbing the patch", BACK_OFF: "Backing off, contact force over the limit", VERIFY: "Checking the steel is clean", HOLD: "Holding, camera view lost", PAUSE_PERSON: "Paused, a person is close by", PAINT: "Painting the bare steel", VERIFY_PAINT: "Checking the paint coat", DONE: "Repair complete", ESCALATE_TO_HUMAN: "Handing over to a human" };
  const SUGGEST = { SOAK: ["mission"], SCRUB: ["force", "map"], DETECT: ["gemini", "inspect"], SECOND_VIEW: ["inspect"], VERIFY: ["gemini", "inspect"], DONE: ["map", "log"] };
  const GROUPS = ["Top hit", "Suggested", "Recent", "Panels", "Robots", "Cameras", "Scenario events", "Demo", "Help"];
  const WIN = 30;
  const S = { state: null, step: -1, t: 0, force: 0, peak: 0, rust: null, conf: null, spike: 0, log: [], logN: 0, why: "", soak: null, shot: 1, leak: 0, leakOn: false, mist: false, labels: false, started: false, cycle: null, source: "EDGE", acted: new Set() };
  const series = { force: [], rust: [], conf: [], gas: [] };
  const SENSORS = [
    ["GD‑1", "Separator", 1.1, 0.04, 0.2, 0.32], ["GD‑2", "Flange B2", 0.8, 0.92, 0.3, 0.7], ["GD‑3", "Compressor", 1.6, 0.36, 0.58, 0.5],
    ["GD‑4", "Manifold", 0.6, 0.14, 0.8, 0.3], ["GD‑5", "Wellhead", 0.9, 0.02, 0.78, 0.76], ["GD‑6", "Quarters", 0.3, 0, 0.46, 0.22],
  ].map(([id, at, b, a, x, y]) => ({ id, at, b, a, x, y, v: b }));
  const ROBOTS = [
    { id: "quad1", t: "Quad‑1", role: "Rust repair", bat: 78, at: { 1: [47, 42], 2: [57, 47], 3: [50, 56] }, st: () => NAME[S.state] || "Standby" },
    { id: "quad2", t: "Quad‑2", role: "Gas survey", bat: 64, at: { 1: [61, 31], 2: [21, 63] }, st: () => (S.leak > 0.3 ? "To Flange B2" : "East deck sweep") },
    { id: "drone1", t: "Drone‑1", role: "Aerial watch", bat: 41, at: { 1: [72, 14], 2: [68, 20] }, st: () => "Overwatch, 40 m" },
  ];
  const gasTop = () => SENSORS.reduce((a, b) => (b.v > a.v ? b : a));
  const lvl = (v) => (v >= 20 ? "stop" : v >= 10 ? "warn" : "");
  const forceColor = (f) => (f > 40 ? "#F85149" : f >= 36 ? "#E3B341" : f >= 18 ? "#3FB950" : "rgba(255,255,255,.72)");
  const ICON = Object.fromEntries(Object.entries({
    flag: '<path d="M6 21V4h11l-2.5 4L17 12H6"/>', wave: '<path d="M3 12h4l2.5-6 5 12 2.5-6h4"/>', gauge: '<path d="M4 17a8 8 0 1 1 16 0"/><path d="m12 17 4-5"/>',
    grid: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M3 12h18M9 5v14M15 5v14"/>', eye: '<path d="M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12Z"/><circle cx="12" cy="12" r="3"/>',
    spark: '<path d="M12 3c.8 5 4 8.2 9 9-5 .8-8.2 4-9 9-.8-5-4-8.2-9-9 5-.8 8.2-4 9-9Z"/>', list: '<path d="M9 6h11M9 12h11M9 18h11M4.5 6h0M4.5 12h0M4.5 18h0"/>',
    wind: '<path d="M3 8h10a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h7"/>', cloud: '<path d="M7 18a4 4 0 0 1-.4-8A6 6 0 0 1 18 9.5 4.3 4.3 0 0 1 17.5 18H7Z"/>',
    target: '<circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/>', cam: '<path d="M4 8h3l2-3h6l2 3h3v11H4Z"/><circle cx="12" cy="13" r="3.5"/>',
    bolt: '<path d="M13 3 5 14h6l-1 7 8-11h-6l1-7Z"/>', play: '<path d="M8 5v14l11-7Z"/>', robot: '<rect x="5" y="8" width="14" height="10" rx="3"/><path d="M12 4v4M9.5 13h0M14.5 13h0"/>',
    tag: '<path d="M3.5 12V4.5H11L20 13.5 13.5 20Z"/><circle cx="7.5" cy="8.5" r="1"/>', keys: '<rect x="3" y="6" width="18" height="12" rx="3"/><path d="M7 10h0M11 10h0M15 10h0M8 14h8"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h0"/>', film: '<rect x="3" y="5" width="18" height="14" rx="3"/><path d="M3 9h18M3 15h18"/>', x: '<path d="M6 6l12 12M18 6 6 18"/>',
    back: '<path d="M4 12a8 8 0 1 0 2.4-5.7M4 4.5v4h4"/>',
  }).map(([k, v]) => [k, `<svg viewBox="0 0 24 24">${v}</svg>`]));
  const tweens = new Map();
  function num(el, v, d = 1) {
    if (v == null || Number.isNaN(v)) { tweens.delete(el); set(el, "—"); el._v = null; return; }
    if (el._v == null || RM) { el._v = v; set(el, v.toFixed(d)); }
    tweens.set(el, { to: v, d });
  }
  function stepTweens() {
    tweens.forEach((tw, el) => {
      el._v += (tw.to - el._v) * 0.2;
      if (Math.abs(tw.to - el._v) < 10 ** -tw.d / 2) { el._v = tw.to; tweens.delete(el); }
      set(el, el._v.toFixed(tw.d));
    });
  }
  function fit(c) {
    const dpr = Math.min(devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
    if (!w) return [];
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return [g, w, h];
  }
  function spark(c, data, o) {
    const [g, w, h] = fit(c);
    if (!g) return;
    const win = o.win || WIN, t0 = now() - win;
    const x = (t) => ((t - t0) / win) * w, y = (v) => h - 2 - ((Math.min(o.max, v) - o.min) / (o.max - o.min)) * (h - 4);
    (o.refs || []).forEach(([v, col]) => { g.strokeStyle = col; g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, Math.round(y(v)) + 0.5); g.lineTo(w, Math.round(y(v)) + 0.5); g.stroke(); });
    g.setLineDash([]);
    const pts = data.filter((p) => p[0] >= t0 - 1);
    if (pts.length < 2) return;
    g.lineWidth = o.w || 1.4;
    g.lineJoin = g.lineCap = "round";
    if (o.ramp) {
      for (let k = 1; k < pts.length; k++) { g.strokeStyle = forceColor(Math.max(pts[k - 1][1], pts[k][1])); g.beginPath(); g.moveTo(x(pts[k - 1][0]), y(pts[k - 1][1])); g.lineTo(x(pts[k][0]), y(pts[k][1])); g.stroke(); }
    } else {
      g.strokeStyle = o.color;
      g.beginPath();
      pts.forEach((p, k) => g[k ? "lineTo" : "moveTo"](x(p[0]), y(p[1])));
      g.stroke();
      if (o.fill) { g.lineTo(x(pts.at(-1)[0]), h); g.lineTo(x(pts[0][0]), h); g.fillStyle = o.fill; g.fill(); }
    }
    const l = pts.at(-1);
    g.fillStyle = o.ramp ? forceColor(l[1]) : o.color;
    g.beginPath(); g.arc(Math.min(w - 2.5, x(l[0])), y(l[1]), 2.2, 0, 7); g.fill();
  }
  function push(k, v) {
    if (v == null) return;
    const s = series[k], t = S.at || now();
    s.push([t, v]);
    while (s.length && s[0][0] < t - WIN - 2) s.shift();
  }
  function drawMap(c) {
    const [g, w, h] = fit(c);
    if (!g || !S.grid) return;
    const rows = S.grid.length, cols = S.grid[0].length, cw = w / cols, ch = h / rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const r = S.grid[i][j], painted = S.painted?.[i][j] === "1", sprayed = S.sprayed?.[i][j] === "1";
      let col = painted ? (r > 0.08 ? `rgba(196,98,58,${0.35 + r * 0.6})` : "rgba(143,168,148,.85)") : r > 0.02 ? `rgba(196,98,58,${0.2 + r * 0.8})` : null;
      if (sprayed && !painted) col = "rgba(122,164,222,.45)";
      if (col) { g.fillStyle = col; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }
    }
    if (S.brush && ["SCRUB", "SPRAY", "PAINT", "BACK_OFF"].includes(S.state)) {
      g.strokeStyle = "#fff"; g.lineWidth = 1.5;
      g.beginPath(); g.arc((S.brush[1] + 0.5) * cw, (S.brush[0] + 0.5) * ch, 4.5, 0, 7); g.stroke();
    }
  }
  function drawPlume(c) {
    const [g, w, h] = fit(c);
    if (!g) return;
    g.strokeStyle = "rgba(255,255,255,.12)";
    g.beginPath(); g.roundRect(w * 0.06, h * 0.1, w * 0.88, h * 0.8, 10); g.stroke();
    g.fillStyle = "rgba(255,255,255,.045)";
    [[0.12, 0.18, 0.2, 0.26], [0.4, 0.16, 0.16, 0.3], [0.64, 0.56, 0.22, 0.26], [0.14, 0.56, 0.3, 0.24]].forEach(([a, b, cw, ch]) => { g.beginPath(); g.roundRect(a * w, b * h, cw * w, ch * h, 5); g.fill(); });
    const src = SENSORS[1], t = now(), L = S.leak;
    for (let k = 0; L > 0.02 && k < 7; k++) {
      const d = k / 6, px = (src.x + d * 0.46) * w + Math.sin(t * 0.8 + k) * 3, py = (src.y - d * 0.36) * h + Math.cos(t * 0.6 + k) * 3, r = (10 + d * 44) * (0.4 + L * 0.6);
      const gr = g.createRadialGradient(px, py, 0, px, py, r);
      gr.addColorStop(0, `rgba(248,81,73,${(0.7 * (1 - d * 0.6) * L).toFixed(3)})`);
      gr.addColorStop(1, "rgba(248,81,73,0)");
      g.fillStyle = gr; g.beginPath(); g.arc(px, py, r, 0, 7); g.fill();
    }
    g.font = "500 10px system-ui, -apple-system, sans-serif";
    SENSORS.forEach((s) => {
      g.fillStyle = { stop: "#F85149", warn: "#E3B341", "": "rgba(255,255,255,.55)" }[lvl(s.v)];
      g.beginPath(); g.arc(s.x * w, s.y * h, 3, 0, 7); g.fill();
      g.fillStyle = "rgba(255,255,255,.4)"; g.fillText(s.id.slice(3), s.x * w + 6, s.y * h + 3.5);
    });
  }
  function report() {
    const g = gasTop();
    if (g.v >= 10) return { key: "gas" + lvl(g.v), sum: `Gas at ${g.at} is above background and rising. Wind is carrying it north‑east, toward the compressor. Quad‑1 is 14 m downwind.`,
      acts: [["Pause Quad‑1 and move it upwind", 0.94], ["Send Quad‑2 to confirm at " + g.at, 0.88], ["Isolate valve V‑12 upstream", 0.81], ["Show the gas plume", 0.63, "plume"]] };
    if (S.state === "BACK_OFF") return { key: "bo", sum: "The brush hit a weld bead at 65 N. The arm retracted 5 cm and will retry at 70 % press.", acts: [["Retry softer", 0.93], ["Show the force sensor", 0.71, "force"], ["Mark the bead on the pipe map", 0.55, "map"]] };
    if (S.state === "PAUSE_PERSON") return { key: "pp", sum: "A person is inside 3 m of Quad‑1. All motion is paused until they move away.", acts: [["Wait for clear", 0.96], ["Show robot labels", 0.6, "labels"], ["Call the deck supervisor", 0.42]] };
    if (S.state === "HOLD") return { key: "ho", sum: "The inspection lens is fogged. Quad‑1 is holding until it can see the patch again.", acts: [["Wait for the view", 0.9], ["Show the inspection camera", 0.74, "inspect"], ["Send Drone‑1 for a view", 0.51]] };
    if (!S.started) return { key: "idle", sum: "Fleet idle on the lower deck. No gas above background.", acts: [["Start the demo", 0.99, "start"]] };
    return { key: "ok" + Math.round(S.rust ?? 100), sum: `Repair on track. ${S.rust == null ? "Survey under way" : `Rust is at ${S.rust.toFixed(0)} %`}. No gas above background. Fleet healthy.`,
      acts: [["Continue the current plan", 0.91], ["Show the pipe map", 0.64, "map"], ["Queue a survey of the next patch", 0.48]] };
  }
  const VIT = [["force", "Brush force", "N", 1, { min: 0, max: 70, ramp: 1, refs: [[40, "rgba(248,81,73,.55)"]] }],
    ["rust", "Rust left", "%", 1, { min: 0, max: 100, color: "#D9825C", fill: "rgba(217,130,92,.12)" }],
    ["conf", "Detector", "", 2, { min: 0, max: 1, color: "rgba(255,255,255,.7)" }]];
  const P = {
    mission: { t: "Mission", ic: "flag", k: "h", dock: "right", kw: "stepper phase progress soak steps", d: "Where the repair is in its eight steps, with the soak timer.",
      html: () => `<div class="track">${STEPS.map(() => "<i></i>").join("")}</div><div class="m-row"><b data-k="step"></b><span data-k="of"></span></div><div class="soak" data-k="soak" hidden><svg viewBox="0 0 36 36"><circle cx="18" cy="18" r="15"/><circle data-k="ring" cx="18" cy="18" r="15"/></svg><b class="tnum" data-k="left"></b><span>s of soak left</span></div>`,
      peek: () => NAME[S.state] || "Standby",
      draw(q, r) {
        if (r._k !== S.step) { r._k = S.step; r.querySelectorAll(".track i").forEach((e, n) => (e.className = n < S.step ? "done" : n === S.step ? (n === 7 ? "cur ok" : "cur") : "")); }
        set(q.step, S.step < 0 ? "Standby" : STEPS[S.step]); set(q.of, S.step < 0 ? "" : `${S.step + 1} of 8`);
        q.soak.hidden = !S.soak;
        if (S.soak) { num(q.left, S.soak.left, 0); q.ring.style.strokeDashoffset = 94.25 * (1 - S.soak.left / S.soak.total); }
      } },
    vitals: { t: "Vitals", ic: "wave", k: "v", dock: "left", kw: "sparkline rust confidence detector", d: "Brush force, rust left and detector confidence over the last 30 s.",
      html: () => VIT.map(([k, l, u]) => `<div class="vit"><span>${l}</span><b class="tnum" data-k="${k}"></b><em>${u}</em><canvas data-k="${k}C"></canvas></div>`).join(""),
      peek: () => fmt(S.rust, 1, " % rust"),
      draw(q) { VIT.forEach(([k, , , d, o]) => { num(q[k], S[k], d); spark(q[k + "C"], series[k], o); }); q.force.style.color = forceColor(S.force); } },
    force: { t: "Force sensor", ic: "gauge", k: "f", dock: "left", kw: "brush contact newton limit", d: "Contact force at the brush, against the 40 N limit.",
      html: () => `<div class="f-now"><b class="tnum" data-k="now"></b><span>N</span><em data-k="max"></em></div><div class="gauge"><span data-k="g"></span><s></s></div><canvas class="f-chart" data-k="c"></canvas>`,
      peek: () => fmt(S.force, 1, " N"),
      draw(q) {
        num(q.now, S.force, 1); q.now.style.color = forceColor(S.force); set(q.max, `peak ${S.peak.toFixed(1)}`);
        q.g.style.width = Math.min(100, S.force * 2) + "%"; q.g.style.background = forceColor(S.force);
        spark(q.c, series.force, { win: 20, min: 0, max: Math.max(50, S.peak * 1.15), ramp: 1, w: 1.6, refs: [[40, "rgba(248,81,73,.8)"]] });
      } },
    map: { t: "Pipe map", ic: "grid", k: "m", dock: "right", kw: "heatmap rust grid sprayed painted", d: "Rust across the unrolled pipe surface, with sprayed and painted cells.",
      html: () => `<canvas class="map" data-k="c"></canvas><div class="legend"><i class="l-rust"></i>Rust<i class="l-spray"></i>Sprayed<i class="l-paint"></i>Painted</div>`,
      peek: () => fmt(S.rust, 0, " %"), draw: (q) => drawMap(q.c) },
    inspect: { t: "Inspection camera", ic: "eye", k: "i", dock: "right", kw: "pip picture camera detector mask", d: "What the detector sees, with the rust mask outlined.",
      html: () => `<div class="pip"><img data-k="img" alt=""><span class="pip-none">Waiting for the first look</span></div>`,
      peek: () => (S.insp ? "Live" : ""),
      draw(q) { if (S.insp && q.img._j !== S.insp.jpg) { q.img._j = S.insp.jpg; q.img.src = "data:image/jpeg;base64," + S.insp.jpg; q.img.parentNode.classList.add("on"); } q.img.classList.toggle("mist", S.mist); } },
    gemini: { t: "Second opinion", ic: "spark", k: "o", dock: "right", meta: "Robotics‑ER", kw: "gemini verdict vlm verify", d: "Gemini Robotics‑ER checks the detector before the robot moves on.",
      html: () => `<div class="g-top"><span class="chip" data-k="v">No call yet</span><span class="tnum dim" data-k="lat"></span></div><div class="bar"><span data-k="bar"></span></div><div class="thumb"><img data-k="img" alt=""><svg data-k="pts"></svg></div><p class="why" data-k="why"></p>`,
      peek: () => (S.gemini ? S.gemini.verdict.replace("_", " ") : ""),
      draw(q, r) {
        const g = S.gemini;
        if (!g || r._g === S.gkey) return;
        r._g = S.gkey;
        q.v.className = "chip " + g.verdict;
        set(q.v, ({ rust: "Rust", clean: "Clean", cannot_tell: "Can't tell" }[g.verdict] || g.verdict) + "  " + g.confidence.toFixed(2));
        set(q.lat, Math.round(g.latency_ms) + " ms"); q.bar.style.width = g.confidence * 100 + "%"; set(q.why, g.reason);
        if (!S.insp) return;
        q.img.src = "data:image/jpeg;base64," + S.insp.jpg;
        q.pts.setAttribute("viewBox", `0 0 ${S.insp.w} ${S.insp.h}`); q.pts.setAttribute("preserveAspectRatio", "xMidYMid slice");
        const rr = S.insp.w / 40;
        q.pts.innerHTML = (g.points || []).map(([x, y]) => `<circle cx="${x}" cy="${y}" r="${rr}" stroke-width="${rr / 4}"/>`).join("");
      } },
    log: { t: "Decisions", ic: "list", k: "l", dock: "bottom", kw: "log history events reasons", d: "Every state change and the reason the planner gave.",
      html: () => `<ol class="log" data-k="l"></ol>`, peek: () => (S.log.length ? S.log.length + " recent" : ""),
      draw(q, r) {
        if (r._n === S.logN) return;
        r._n = S.logN;
        q.l.innerHTML = S.log.slice(0, 4).map((e) => {
          const c = SAFETY[e.to] || (e.to === "GAS" ? "stop" : e.to === "DONE" ? "ok" : /^RE-PLAN/.test(e.reason || "") ? "warn" : "");
          return `<li><span class="tnum">${Number(e.t).toFixed(1)}</span><b class="${c}">${esc(NAME[e.to] || e.to)}</b><span>${esc((e.reason || "").replace(/->/g, "→"))}</span></li>`;
        }).join("");
      } },
    gas: { t: "Gas sensors", ic: "wind", k: "g", dock: "left", meta: "% LEL", kw: "leak lel h2s detector methane", d: "Six fixed detectors around the process deck. Alarm at 20 % LEL.",
      html: () => SENSORS.map((s, n) => `<div class="gas-row" data-k="r${n}"><span>${s.id}</span><span class="dim">${s.at}</span><i><s data-k="b${n}"></s></i><b class="tnum" data-k="v${n}"></b></div>`).join("") + `<canvas class="gas-chart" data-k="c"></canvas>`,
      peek: () => gasTop().v.toFixed(1) + " % LEL",
      draw(q) {
        SENSORS.forEach((s, n) => { num(q["v" + n], s.v, 1); q["b" + n].style.width = Math.min(100, s.v * 2) + "%"; const c = lvl(s.v); if (q["r" + n]._c !== c) { q["r" + n]._c = c; q["r" + n].className = "gas-row " + c; } });
        spark(q.c, series.gas, { min: 0, max: 50, color: gasTop().v >= 20 ? "#F85149" : "rgba(255,255,255,.7)", refs: [[20, "rgba(248,81,73,.5)"]] });
      } },
    plume: { t: "Gas plume", ic: "cloud", k: "p", dock: "right", meta: "Wind NE 6 m/s", kw: "leak dispersion wind map", d: "Top‑down plume estimate from the detectors and the wind.",
      html: () => `<canvas class="plume" data-k="c"></canvas>`, peek: () => (S.leak > 0.05 ? "Spreading" : "Clear"), draw: (q) => drawPlume(q.c) },
    control: { t: "Control centre", ic: "target", k: "a", dock: "right", meta: "Gemini 4 Argon", kw: "situation report gemini argon actions plan", d: "A situation report with ranked actions, rebuilt whenever the picture changes.",
      html: () => `<p class="cc-sum" data-k="sum"></p><ol class="cc-acts" data-k="acts"></ol>`, peek: () => (gasTop().v >= 10 || SAFETY[S.state] ? "New plan" : "On track"),
      draw(q, r) {
        const rep = report(), key = rep.key + S.acted.size;
        if (r._k === key) return;
        r._k = key;
        set(q.sum, rep.sum);
        q.acts.innerHTML = rep.acts.map(([l, c], n) => `<li data-act="${n}" class="${S.acted.has(l) ? "done" : ""}"><span class="tnum">${n + 1}</span><span>${esc(l)}</span><b class="tnum">${Math.round(c * 100)}</b></li>`).join("");
        q.acts.onclick = (e) => {
          const li = e.target.closest("[data-act]");
          if (!li) return;
          const [l, , id] = rep.acts[+li.dataset.act];
          S.acted.add(l); addLog({ t: S.t, to: "ACTION", reason: l });
          if (id) CMD[id].run();
        };
      } },
    fleet: { t: "Fleet", ic: "robot", k: "b", dock: "left", g: "Robots", kw: "robots quadruped drone battery", d: "Every robot on the rig, what it is doing and its battery.",
      html: () => ROBOTS.map((r) => `<div class="bot"><b>${r.t}</b><span class="dim" data-k="${r.id}"></span><i><s style="width:${r.bat}%"></s></i><em class="tnum">${r.bat}</em></div>`).join(""),
      peek: () => ROBOTS.length + " robots", draw: (q) => ROBOTS.forEach((r) => set(q[r.id], r.st())) },
  };
  const EVENTS = [
    { id: "rust", t: "New rust patch", k: "1", d: "Injects a fresh patch next to the brush. The planner re-plans: spray it, then extend the soak.", ev: () => ({ name: "spawn_patch", args: {} }) },
    { id: "glare", t: "Sun glare", k: "2", tog: 1, d: "Floods the inspection view with glare, so the detector loses confidence.", ev: (on) => ({ name: "set_glare", args: { on } }) },
    { id: "mist", t: "Lens mist", k: "3", tog: 1, d: "Fogs the inspection lens. The robot holds until it can see again.", ev: (on) => ({ name: "occlude_camera", args: { on } }) },
    { id: "bead", t: "Weld bead", k: "4", d: "The brush hits a weld bead at 65 N. The arm backs off and retries softer.", ev: () => ({ name: "force_spike", args: {} }) },
    { id: "worker", t: "Worker walks up", k: "5", tog: 1, d: "A person steps within 3 m. Everything pauses until they leave.", ev: (on) => ({ name: "person_at", args: { distance: on ? 2.4 : null } }) },
    { id: "leak", t: "Gas leak", k: "6", tog: 1, d: "Flange B2 starts leaking. The detectors climb, the plume grows and Control centre ranks a response.", ev: (on) => ({ name: "gas_leak", args: { on } }) },
    { id: "can", t: "Drop a can", k: "7", d: "A can lands on R1's walkway. The nearest robot with a gripper is dispatched to pick it up.", ev: () => ({ name: "drop_can", args: { at: "R1" } }) },
  ];
  const CMDS = [];
  const cmd = (o) => (CMDS.push(o), o);
  Object.entries(P).forEach(([id, p]) => cmd({ id, g: p.g || "Panels", t: p.t, ic: p.ic, key: p.k, kbd: p.k.toUpperCase(), kw: p.kw, d: p.d, panel: 1, peek: p.peek, on: () => opened.has(id), run: (f) => toggle(id, f) }));
  cmd({ id: "labels", g: "Robots", t: "Robot labels", ic: "tag", key: "r", kbd: "R", kw: "names tags overlay", d: "Floating name tags over each robot in the shot.", on: () => S.labels, run: () => { S.labels = !S.labels; } });
  ROBOTS.forEach((r) => cmd({ id: r.id, g: "Robots", t: r.t, ic: "robot", kw: r.role.toLowerCase(), d: `${r.role}. Battery ${r.bat} %.`, peek: r.st, run: () => { S.labels = true; ping(r.id); } }));
  ["Wide rig|Aerial orbit around the whole platform.", "Hero deck|The main deck at eye level, facing the repair.", "Close‑up|Tight on the brush and the patch."].forEach((s, n) => {
    const [t, d] = s.split("|");
    cmd({ id: "cam" + (n + 1), g: "Cameras", t, ic: "cam", key: String(n + 1), kbd: String(n + 1), kw: "camera shot view", d, media: `<img class="pv-img" src="/api/mock/shot/${n + 1}" alt="">`, on: () => S.shot === n + 1, run: () => shot(n + 1) });
  });
  EVENTS.forEach((e) => cmd({ id: "ev-" + e.id, g: "Scenario events", t: e.t, ic: "bolt", kbd: "⇧" + e.k, kw: "unscripted inject judge", d: e.d, on: e.tog ? () => !!e.on : null, run: () => fire(e) }));
  cmd({ id: "start", g: "Demo", t: "Start demo", ic: "play", kbd: "Space", kw: "begin play run", d: "Starts the mission replay from the patrol.", run: begin });
  cmd({ id: "cine", g: "Demo", t: "Cinematic mode", ic: "film", key: "c", kbd: "C", kw: "hide clean video", d: "Hides everything but the video. ⌘K still works.", on: () => document.body.classList.contains("cine"), run: () => document.body.classList.toggle("cine") });
  cmd({ id: "closeall", g: "Demo", t: "Close all panels", ic: "x", key: "x", kbd: "X", kw: "clear hide reset", d: "Sends every docked card away.", run: () => [...opened.keys()].forEach((id) => hide(id)) });
  if (MOCK) cmd({ id: "restart", g: "Demo", t: "Restart replay", ic: "back", kw: "reset reload again", d: "Plays the mock mission again from the start.", run: () => { location.search = "?mock=1&autostart=1"; } });
  cmd({ id: "keys", g: "Help", t: "Keyboard shortcuts", ic: "keys", kbd: "?", kw: "help controls keys", d: "Every command also has a key.",
    extra: () => `<ul class="pv-keys">${[[MAC ? "⌘K" : "Ctrl K", "Search"], ["/", "Search"], ["esc", "Close"], ...CMDS.filter((c) => c.kbd && c.kbd !== "?").map((c) => [c.kbd, c.t])].map(([k, t]) => `<li><kbd>${esc(k)}</kbd>${esc(t)}</li>`).join("")}</ul>` });
  cmd({ id: "about", g: "Help", t: "About this demo", ic: "info", kw: "help info", d: "A quadruped finds corrosion on offshore pipework, treats it and proves it is gone. Gemini checks its work; the planner explains every decision." });
  const CMD = Object.fromEntries(CMDS.map((c) => [c.id, c]));
  const views = new Set();
  function mount(id, root) {
    const q = {};
    root.querySelectorAll("[data-k]").forEach((e) => (q[e.dataset.k] = e));
    const v = { p: P[id], q, root };
    views.add(v);
    v.p.draw(q, root);
    return v;
  }
  const DOCKS = { left: $("dock-left"), right: $("dock-right"), bottom: $("dock-bottom") };
  const opened = new Map();
  const toggle = (id, from) => (opened.has(id) ? hide(id) : show(id, from));
  function show(id, from) {
    const p = P[id], d = p.dock, dock = DOCKS[d], el = document.createElement("section");
    el.className = "card flying";
    el.innerHTML = `<header><span>${p.t}</span>${p.meta ? `<em>${p.meta}</em>` : ""}<button class="x" aria-label="Close ${p.t}">${ICON.x}</button></header>${p.html()}`;
    el.querySelector(".x").onclick = () => hide(id);
    dock.appendChild(el);
    opened.set(id, { el, v: mount(id, el) });
    for (const [k, o] of opened) if (dock.scrollHeight > dock.clientHeight + 1 && o.el.parentNode === dock && k !== id) hide(k, true);
    if (RM) { el.classList.remove("flying"); return; }
    const r = el.getBoundingClientRect();
    el.style.transition = "none";
    el.style.transform = from
      ? `translate(${from.left + from.width / 2 - r.left - r.width / 2}px,${from.top + from.height / 2 - r.top - r.height / 2}px) scale(${Math.min(1, from.width / r.width).toFixed(3)})`
      : { left: "translateX(-16px)", right: "translateX(16px)", bottom: "translateY(16px)" }[d];
    el.style.opacity = from ? "0.7" : "0";
    el.offsetWidth;
    el.style.transition = el.style.transform = el.style.opacity = "";
    setTimeout(() => el.classList.remove("flying"), 380);
  }
  function hide(id, instant) {
    const o = opened.get(id);
    if (!o) return;
    opened.delete(id); views.delete(o.v);
    if (instant || RM) { o.el.remove(); return; }
    o.el.style.height = o.el.offsetHeight + "px";
    o.el.classList.add("leaving");
    o.el.offsetWidth;
    o.el.style.height = "0px";
    setTimeout(() => o.el.remove(), 280);
  }
  function clip(dock) {
    let p = "";
    for (const el of dock.children) {
      if (el.tagName !== "SECTION" || el.classList.contains("flying") || el.offsetHeight < 1) continue;
      const x = el.offsetLeft, y = el.offsetTop, w = el.offsetWidth, h = el.offsetHeight, k = Math.min(16, h / 2);
      p += `M${x + k} ${y}H${x + w - k}A${k} ${k} 0 0 1 ${x + w} ${y + k}V${y + h - k}A${k} ${k} 0 0 1 ${x + w - k} ${y + h}H${x + k}A${k} ${k} 0 0 1 ${x} ${y + h - k}V${y + k}A${k} ${k} 0 0 1 ${x + k} ${y}Z`;
    }
    if (dock._p === p) return;
    dock._p = p;
    dock.firstElementChild.hidden = !p;
    dock.firstElementChild.style.clipPath = p ? `path("${p}")` : "";
  }
  const pal = $("palette"), inp = $("p-input"), list = $("p-list"), pv = $("p-preview");
  let palOpen = false, items = [], sel = 0, pvCmd = null, pvView = null, recent = store.get("recent", []);
  function openPal(pick) { palOpen = true; pal.classList.add("open"); inp.value = ""; filter(pick); inp.focus(); }
  function closePal() { palOpen = false; pal.classList.remove("open"); inp.blur(); setPreview(null); }
  function suggested() {
    const a = alarm(), ids = !S.started ? ["start", "control", "cam1"] : [...(a ? a.panels : []), ...(SUGGEST[S.state] || []), "control", "mission", "force"];
    return [...new Set(ids)].slice(0, 4);
  }
  function fuzzy(q, s) {
    s = s.toLowerCase();
    let i = 0, sc = 0, prev = -2;
    const idx = [];
    for (let j = 0; j < s.length && i < q.length; j++) if (s[j] === q[i]) { sc += 1 + (j === prev + 1 ? 2 : 0) + (j === 0 || s[j - 1] === " " ? 3 : 0); prev = j; idx.push(j); i++; }
    return i === q.length ? { sc: sc - s.length * 0.02, idx } : null;
  }
  const match = (q, c) => fuzzy(q, c.t) || ((c.kw || "") + " " + c.g.toLowerCase()).split(" ").some((w) => w.startsWith(q)) && { sc: 0.5, idx: [] };
  function filter(pick) {
    const q = inp.value.trim().toLowerCase();
    if (!q) {
      const sug = suggested();
      items = [...sug.map((id) => ({ c: CMD[id], g: "Suggested" })), ...recent.filter((id) => CMD[id] && !sug.includes(id)).slice(0, 4).map((id) => ({ c: CMD[id], g: "Recent" }))];
    } else {
      const rows = CMDS.map((c) => ({ c, m: match(q, c), g: c.g })).filter((r) => r.m).sort((a, b) => b.m.sc - a.m.sc);
      if (rows[0]) rows[0].g = "Top hit";
      items = rows.sort((a, b) => GROUPS.indexOf(a.g) - GROUPS.indexOf(b.g) || b.m.sc - a.m.sc);
    }
    sel = Math.max(0, items.findIndex((r) => r.c.id === pick));
    renderList();
  }
  const hl = (t, m) => (m?.idx.length ? [...t].map((ch, i) => (m.idx.includes(i) ? `<mark>${esc(ch)}</mark>` : esc(ch))).join("") : esc(t));
  function renderList() {
    let g0 = "";
    list.innerHTML = items.map((r, i) => {
      const c = r.c, head = r.g !== g0 ? `<li class="gh" role="presentation">${(g0 = r.g)}</li>` : "";
      return head + `<li role="option" data-i="${i}" class="${c.on?.() ? "on" : ""}"><span class="ic">${ICON[c.ic]}</span><span class="lb">${hl(c.t, r.m)}</span><span class="pk">${c.peek ? esc(c.peek()) : ""}</span><i class="st"></i>${c.kbd ? `<kbd>${c.kbd}</kbd>` : ""}</li>`;
    }).join("") || `<li class="empty">No results</li>`;
    pvCmd = undefined;
    select(sel, true);
  }
  function select(i, force) {
    if (!items.length) { setPreview(null); return; }
    i = (i + items.length) % items.length;
    if (i === sel && !force) return;
    sel = i;
    list.querySelector("li.sel")?.classList.remove("sel");
    const li = list.querySelector(`li[data-i="${i}"]`);
    li.classList.add("sel");
    li.scrollIntoView({ block: "nearest" });
    setPreview(items[i].c);
  }
  function setPreview(c) {
    if (c === pvCmd) return;
    pvCmd = c;
    if (pvView) views.delete(pvView);
    pvView = null;
    if (!c) { pv.innerHTML = palOpen && !items.length ? `<p class="pv-none">Try “force”, “gas” or “camera”</p>` : ""; return; }
    pv.innerHTML = `<div class="pv-head"><span class="ic">${ICON[c.ic]}</span><div><b>${esc(c.t)}</b><span>${esc(c.g)}</span></div></div>`
      + (c.panel ? `<div class="pv-live">${P[c.id].html()}</div>` : c.media || "") + (c.d ? `<p class="pv-d">${esc(c.d)}</p>` : "") + (c.extra ? c.extra() : "");
    if (c.panel) pvView = mount(c.id, pv.querySelector(".pv-live"));
  }
  function run(c, keep) {
    const from = palOpen ? pv.querySelector(".pv-live")?.getBoundingClientRect() || pv.getBoundingClientRect() : null;
    if (!c.run) return;
    recent = [c.id, ...recent.filter((x) => x !== c.id)].slice(0, 6);
    store.set("recent", recent);
    if (keep) { c.run(from); renderList(); return; }
    closePal();
    c.run(from);
  }
  function peeks() {
    list.querySelectorAll("li[data-i]").forEach((li) => {
      const c = items[+li.dataset.i]?.c;
      if (c?.peek) set(li.children[2], c.peek());
      li.classList.toggle("on", !!c?.on?.());
    });
  }
  inp.addEventListener("input", () => filter());
  list.addEventListener("pointermove", (e) => { const li = e.target.closest("li[data-i]"); if (li && (e.movementX || e.movementY)) select(+li.dataset.i); });
  list.addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) run(items[+li.dataset.i].c, e.metaKey || e.ctrlKey); });
  addEventListener("pointerdown", (e) => { if (palOpen && !pal.contains(e.target) && !e.target.closest("#capsule")) closePal(); });
  $("capsule").onclick = () => (palOpen ? closePal() : openPal());
  function alarm() {
    const g = gasTop();
    if (g.v >= 20) return { lvl: "stop", text: "Gas at " + g.at, num: [g.v, 0, " % LEL"], panels: ["gas", "plume", "control"] };
    const s = SAFETY[S.state];
    if (!s) return null;
    const peak = Math.max(S.force, ...series.force.filter((p) => p[0] > now() - 3).map((p) => p[1]));
    return { lvl: s, text: { BACK_OFF: "Back off", HOLD: "View lost", PAUSE_PERSON: "Person within 3 m", ESCALATE_TO_HUMAN: "Needs a human" }[S.state],
      num: S.state === "BACK_OFF" ? [peak, 0, " N"] : S.state === "PAUSE_PERSON" ? [2.4, 1, " m"] : null,
      panels: { BACK_OFF: ["force", "control"], HOLD: ["inspect", "control"], PAUSE_PERSON: ["labels", "control"] }[S.state] || ["control"] };
  }
  function subtitle(a) {
    if (!S.started) return ["Offshore rig, rust repair", `Press Space to begin, or ${MAC ? "⌘K" : "Ctrl K"} for everything`];
    if (a?.lvl === "stop" && gasTop().v >= 20) return [`Gas detected near ${gasTop().at}`, "Control centre has ranked a response"];
    let m = PHRASE[S.state] || "";
    if (S.state === "SCRUB" && S.cycle) m += `, pass ${S.cycle[0]} of ${S.cycle[1]}`;
    return [m, (S.why || "").replace(/->/g, "→")];
  }
  const labels = $("labels");
  labels.innerHTML = ROBOTS.map((r) => `<div class="rl" data-r="${r.id}"><b>${r.t}</b><span></span></div>`).join("");
  function ping(id) { const el = labels.querySelector(`[data-r="${id}"]`); el.classList.remove("ping"); el.offsetWidth; el.classList.add("ping"); }
  function hud() {
    const a = alarm(), cap = $("capsule"), cls = a ? a.lvl : S.state === "DONE" ? "ok" : S.started ? "run" : "";
    if (cap._c !== cls) { cap._c = cls; cap.className = "glass hl " + cls; $("vignette").className = a ? a.lvl : ""; }
    set($("c-state"), a ? a.text : S.started ? NAME[S.state] || "Starting" : "Ready");
    num($("c-time"), S.t, 1);
    const k = a ? a.num : S.soak ? [S.soak.left, 0, " s soak"] : S.state === "SCRUB" ? [S.force, 0, " N"] : S.rust == null ? null : [S.rust, 1, " % rust"];
    $("c-key").hidden = !k;
    if (k) { num($("c-num"), k[0], k[1]); set($("c-unit"), k[2]); }
    const [m, s] = subtitle(a);
    if ($("s-main").textContent !== m) { set($("s-main"), m); if (!RM) $("subtitle").animate([{ opacity: 0.2 }, { opacity: 1 }], 240); }
    set($("s-sub"), s);
    labels.classList.toggle("on", S.labels);
    if (labels._s !== S.shot) { labels._s = S.shot; ROBOTS.forEach((r) => { const el = labels.querySelector(`[data-r="${r.id}"]`), p = r.at[S.shot]; el.hidden = !p; if (p) { el.style.left = p[0] + "%"; el.style.top = p[1] + "%"; } }); }
    if (S.labels) ROBOTS.forEach((r) => set(labels.querySelector(`[data-r="${r.id}"] span`), r.st()));
    set($("p-src"), S.source);
  }
  function gasStep(dt) {
    S.leak += ((S.leakOn ? 1 : 0) - S.leak) * Math.min(1, dt * (S.leakOn ? 0.22 : 0.5));
    SENSORS.forEach((s) => { s.v = Math.max(0, s.v + (s.b + S.leak * s.a * 50 - s.v) * Math.min(1, dt * 2) + (Math.random() - 0.5) * 0.05); });
    if (now() - (S.gasT || 0) > 0.25) { S.gasT = now(); push("gas", gasTop().v); }
  }
  function addLog(e) { S.log.unshift(e); S.log = S.log.slice(0, 8); S.logN++; S.why = e.reason; }
  function setState(st) {
    S.state = st;
    if (st in STEP_OF) S.step = STEP_OF[st];
    if (st === "DONE" && MOCK) setTimeout(() => shot(1), 3500);
  }
  function apply(h) {
    if (h.state && h.state !== S.state) setState(h.state);
    if (h.t != null) S.t = h.t;
    if (h.hud) { const m = /cycle (\d+)\/(\d+)/.exec(h.hud[0] || ""); S.cycle = m ? [+m[1], +m[2]] : null; }
    (h.events || []).forEach(addLog);
    if (h.rust_grid) S.grid = h.rust_grid;
    if (h.sprayed) S.sprayed = h.sprayed;
    if (h.painted) S.painted = h.painted;
    if (h.brush) S.brush = h.brush;
    if (h.force_n != null) { S.force = Math.max(h.force_n, S.spike); push("force", S.force); S.peak = Math.max(S.peak, S.force); }
    if (h.rust_fraction != null) { S.rust = h.rust_fraction * 100; push("rust", S.rust); }
    if (h.detector_conf != null) { S.conf = h.detector_conf; push("conf", S.conf); }
    if (h.source) S.source = h.source;
    if (h.inspection) S.insp = h.inspection;
    if (h.gemini) { const k = JSON.stringify(h.gemini); if (k !== S.gkey) { S.gkey = k; S.gemini = h.gemini; } }
    if ("soak" in h) S.soak = h.soak;
  }
  const post = (ev) => fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ev) }).catch(() => null);
  function fire(a) {
    a.on = a.tog ? !a.on : true;
    const judge = (to, reason) => { addLog({ t: S.t, to, reason }); if (to in NAME && to !== "GAS") setState(to); };
    if (a.id === "leak") { S.leakOn = a.on; addLog({ t: S.t, to: "GAS", reason: a.on ? "Flange B2 leaking, detectors rising" : "leak isolated" }); }
    if (!MOCK) { if (a.id !== "leak") post(a.ev(a.on)); return; }
    if (a.id === "rust" && S.grid) {
      const [bi, bj] = S.brush || [20, 30];
      S.grid = S.grid.map((row, i) => row.map((v, j) => { const d = Math.hypot((i - bi + 6) / 3.5, (((j - bj + 13 + 64) % 64) / 4) - 1.5); return Math.max(v, d < 1.6 ? 0.85 * (1 - d / 1.6) + 0.1 : 0); }));
      judge("SPRAY", "RE-PLAN: new rust injected by judge -> spray it, extend soak");
    }
    if (a.id === "glare") { $("flash").classList.toggle("on", a.on); addLog({ t: S.t, to: "DETECT", reason: a.on ? "glare on inspection view -> detector unsure" : "glare cleared" }); }
    if (a.id === "mist") { S.mist = a.on; judge(a.on ? "HOLD" : "VERIFY", a.on ? "perception lost (lens mist) -> hold position" : "camera view restored"); }
    if (a.id === "bead") { S.spike = 64.8; S.force = 64.8; push("force", 64.8); S.peak = Math.max(S.peak, 64.8); setTimeout(() => (S.spike = 0), 400); judge("BACK_OFF", "contact force 65 N > 40 N limit -> retract"); }
    if (a.id === "worker") judge(a.on ? "PAUSE_PERSON" : "SCRUB", a.on ? "person at 2.4 m < 3.0 m -> pause" : "person clear -> resume");
    if (a.id === "can") addLog({ t: S.t, to: "PICK_CAN", reason: "can on R1's walkway -> nearest gripper robot picks it up" });
  }
  let side = 0;
  function shot(n) {
    S.shot = n;
    if (!MOCK) { post({ name: "camera", args: { shot: n } }); return; }
    const a = $(side ? "shotA" : "shotB"), b = $(side ? "shotB" : "shotA");
    side ^= 1;
    a.classList.remove("orbit");
    a.style.transition = "none";
    a.style.backgroundImage = `url(/api/mock/shot/${n})`;
    a.style.transform = n === 1 ? "scale(1.18)" : "scale(1.3) translateY(-3%)";
    a.offsetWidth;
    a.style.transition = "";
    a.classList.add("on");
    a.style.transform = "scale(1.04)";
    b.classList.remove("on");
    if (n === 1) setTimeout(() => { a.style.transform = ""; a.classList.add("orbit"); }, 4000);
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
    shot(2);
    if (MOCK) { playT0 = now(); if (offset && session) seek(offset); }
  }
  function liveState() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/state");
    ws.onmessage = (ev) => { try { S.started = true; apply(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(liveState, 1500);
  }
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if ((e.metaKey || e.ctrlKey) && k === "k") { e.preventDefault(); palOpen ? closePal() : openPal(); return; }
    if (palOpen) {
      if (k === "escape") { e.preventDefault(); closePal(); }
      else if (k === "arrowdown" || k === "arrowup") { e.preventDefault(); select(sel + (k === "arrowdown" ? 1 : -1)); }
      else if (k === "enter" && items[sel]) { e.preventDefault(); run(items[sel].c, e.metaKey || e.ctrlKey); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey || e.target.tagName === "INPUT") return;
    if (k === "/") { e.preventDefault(); openPal(); return; }
    if (e.key === "?") { e.preventDefault(); openPal("keys"); return; }
    if (e.code === "Space") { e.preventDefault(); begin(); return; }
    if (e.shiftKey && /^Digit[1-7]$/.test(e.code)) { fire(EVENTS[+e.code[5] - 1]); return; }
    if (!e.shiftKey) CMDS.find((c) => c.key === k)?.run();
  });
  let last = now(), frames = 0, fpsT = now();
  function loop() {
    const t = now();
    Object.values(DOCKS).forEach(clip);
    playMock();
    gasStep(Math.min(0.1, t - last));
    last = t;
    stepTweens();
    views.forEach((v) => v.p.draw(v.q, v.root));
    hud();
    if (palOpen) peeks();
    frames++;
    if (MOCK && t - fpsT > 0.5) { $("fps").textContent = String(Math.round(frames / (t - fpsT))); frames = 0; fpsT = t; }
    requestAnimationFrame(loop);
  }
  $("c-kbd").textContent = MAC ? "⌘K" : "Ctrl K";
  $("p-keep").textContent = MAC ? "⌘↵" : "Ctrl ↵";
  if (MOCK) {
    $("status").textContent = "Mock replay";
    shot(1);
    fetch("/api/mock").then((r) => r.json()).then((s) => { session = s; if (playT0 != null && offset) seek(offset); });
  } else liveState();
  if (Q.has("autostart")) begin();
  requestAnimationFrame(loop);
})();
