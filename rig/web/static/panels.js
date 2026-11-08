(() => {
  const H = HUD, { q, num, set, esc, spark, fit, now } = H;
  const fc = (f) => (f > 40 ? "#F85149" : f >= 36 ? "#E3B341" : f >= 18 ? "#3FB950" : "rgba(255,255,255,.72)");
  const COL = { stop: "#F85149", warn: "#E3B341", ok: "#3FB950", act: "#2EC4B6", idle: "rgba(235,235,245,.6)" };
  const tone = (st = "") => (/PAUSE|RETREAT|FAIL|ESCALATE/.test(st) ? "stop" : /WAIT|BACK|HOLD|DOCK|GAS|BLOCK/.test(st) ? "warn" : /DONE|CHARG/.test(st) ? "ok" : /IDLE/.test(st) ? "idle" : "act");
  const NAMES = { R1: "Unit 1 · rust", R2: "Unit 2 · gas", R3: "Unit 3", Q1: "Unit 1 · rust", Q2: "Unit 2 · gas", Q3: "Unit 3", "unit-1": "Unit 1 · rust", "unit-2": "Unit 2 · gas", "unit-3": "Unit 3", unit1: "Unit 1 · Scrubber", unit2: "Unit 2 · Fixer" };
  const rname = (r) => r.name || NAMES[r.id] || r.id;
  const pct = (v, d = 0) => (v == null ? "—" : (100 * v).toFixed(d) + " %");
  const STEPS = ["Patrol", "Detect", "Spray", "Soak", "Scrub", "Verify", "Paint", "Done"];
  const STEP_OF = { PATROL: 0, DETECT: 1, SECOND_VIEW: 1, APPROACH: 1, SPRAY: 2, SOAK: 3, SCRUB: 4, BACK_OFF: 4, VERIFY: 5, PAINT: 6, VERIFY_PAINT: 6, DONE: 7 };
  const SEE = ["DETECT", "SECOND_VIEW", "APPROACH", "VERIFY", "VERIFY_PAINT", "HOLD"];
  Object.assign(H, { fc, COL, tone, rname });
  const MISSION = `<div class="track">${STEPS.map(() => "<i></i>").join("")}</div><div class="m-row"><b data-k="step"></b><span data-k="of"></span></div><div class="soak" data-k="soak"><svg viewBox="0 0 36 36"><circle cx="18" cy="18" r="15"/><circle data-k="ring" cx="18" cy="18" r="15"/></svg><b class="tnum" data-k="left"></b><span>s of soak left</span></div>`;
  H.panel({ id: "mission", title: "Mission", icon: "flag", key: "n", phases: ["SOAK"], priority: 50, keys: "stepper soak progress steps timer",
    desc: "Where the repair is in its eight steps, with the soak timer.", hint: (S) => H.nice(S.state),
    render(el, S) {
      const x = q(el, MISSION), k = STEP_OF[S.state] ?? -1;
      if (el._k !== k) { el._k = k; [...el.firstChild.children].forEach((e, n) => (e.className = n < k ? "done" : n === k ? (n === 7 ? "cur ok" : "cur") : "")); }
      set(x.step, k < 0 ? "Standby" : STEPS[k]); set(x.of, k < 0 ? "" : `${k + 1} of 8`);
      x.soak.hidden = !S.soak;
      if (S.soak) { num(x.left, S.soak.left, 0); x.ring.style.strokeDashoffset = 94.25 * (1 - S.soak.left / S.soak.total); }
    } });
  const VIT = [["force", "Brush force", "N", 1, { min: 0, max: 70, ramp: fc, refs: [[40, "rgba(248,81,73,.55)"]] }],
    ["rust", "Rust left", "%", 1, { min: 0, max: 100, color: "#D9825C" }], ["conf", "Detector", "", 2, { min: 0, max: 1, color: "rgba(255,255,255,.7)" }]];
  const VITALS = VIT.map(([k, l, u]) => `<div class="vit"><span>${l}</span><b class="tnum" data-k="${k}"></b><em>${u}</em><canvas data-k="${k}C"></canvas></div>`).join("");
  H.panel({ id: "vitals", title: "Vitals", icon: "wave", key: "v", edge: "left", keys: "sparkline rust confidence detector",
    desc: "Brush force, rust left and detector confidence over the last 30 s.", hint: (S) => (S.rust == null ? "" : S.rust.toFixed(1) + " % rust"),
    render(el, S) { const x = q(el, VITALS); VIT.forEach(([k, , , d, o]) => { num(x[k], S[k], d); spark(x[k + "C"], S.series[k], o); }); x.force.style.color = fc(S.force); } });
  function drawPipe(c, S) {
    const [g, w, h] = fit(c), G = S.rust_grid;
    if (!g || !G) return;
    const rows = G.length, cols = G[0].length, cw = w / cols, ch = h / rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const r = G[i][j], painted = S.painted?.[i][j] === "1", sprayed = S.sprayed?.[i][j] === "1";
      let col = painted ? (r > 0.08 ? `rgba(196,98,58,${0.35 + r * 0.6})` : "rgba(143,168,148,.85)") : r > 0.02 ? `rgba(196,98,58,${0.2 + r * 0.8})` : null;
      if (sprayed && !painted) col = "rgba(122,164,222,.45)";
      if (col) { g.fillStyle = col; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }
    }
    if (S.brush && ["SCRUB", "SPRAY", "PAINT", "BACK_OFF"].includes(S.state)) { g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.arc((S.brush[1] + 0.5) * cw, (S.brush[0] + 0.5) * ch, 4.5, 0, 7); g.stroke(); }
  }
  const PIPE = `<canvas class="map" data-k="c"></canvas><div class="legend"><i class="l-rust"></i>Rust<i class="l-spray"></i>Sprayed<i class="l-paint"></i>Painted</div>`;
  H.panel({ id: "pipe", title: "Pipe map", icon: "grid", key: "m", phases: ["DONE", "PAINT"], priority: 30, keys: "heatmap rust grid sprayed painted",
    desc: "Rust across the unrolled pipe surface, with sprayed and painted cells.", hint: (S) => (S.rust == null ? "" : S.rust.toFixed(0) + " %"),
    render(el, S) { drawPipe(q(el, PIPE).c, S); } });
  const PIP = `<div class="pip"><img data-k="img" alt=""><canvas data-k="cv"></canvas><span class="pip-none" data-k="none">Waiting for the first look</span><em class="pip-l" data-k="lab"></em></div>`;
  H.panel({ id: "inspect", title: "Inspection camera", icon: "eye", key: "i", phases: (S) => !!S.inspection && SEE.includes(S.state), priority: 40, keys: "pip picture camera detector mask",
    desc: "What the on-board detector sees, with the rust mask outlined. In a replay it shows the recorded Isaac render nearest the replay clock.", hint: (S) => (S.inspection ? "Live" : H.simcam ? H.simcam(S).label || "" : ""),
    render(el, S) {
      const x = q(el, PIP), j = S.inspection?.jpg, sc = !j && H.simcam ? H.simcam(S) : null, src = j ? "data:image/jpeg;base64," + j : sc?.img;
      if (src && x.img._j !== src) { x.img._j = src; x.img.src = src; }
      const vid = sc?.video, pip = x.img.parentNode;
      if (vid && vid.videoWidth) { const c = x.cv, w = c.clientWidth * 2, h = c.clientHeight * 2; if (c.width !== w) { c.width = w; c.height = h; } const k = Math.max(w / vid.videoWidth, h / vid.videoHeight); c.getContext("2d").drawImage(vid, (w - vid.videoWidth * k) / 2, (h - vid.videoHeight * k) / 2, vid.videoWidth * k, vid.videoHeight * k); }
      pip.classList.toggle("on", !!src || !!vid); pip.classList.toggle("vid", !!vid);
      set(x.none, sc?.none || "Waiting for the first look"); set(x.lab, sc?.label || (j ? "On-board detector" : ""));
      x.img.classList.toggle("mist", !!S.mist);
    } });
  const GEM = `<div class="g-top"><span class="chip" data-k="v">No call yet</span><span class="tnum dim" data-k="lat"></span></div><div class="bar"><span data-k="bar"></span></div><div class="thumb"><img data-k="img" alt=""><svg data-k="pts"></svg></div><p class="why" data-k="why"></p>`;
  H.panel({ id: "gemini", title: "Second opinion", meta: "Gemini Robotics-ER", icon: "spark", key: "o", phases: (S) => now() - S.gemAt < 9 / S.speed, priority: 70, keys: "gemini verdict vlm verify",
    desc: "Gemini checks the detector's call before the robot moves on.", hint: (S) => (S.gemini ? S.gemini.verdict.replace("_", " ") : ""),
    render(el, S) {
      const x = q(el, GEM), g = S.gemini;
      if (!g || el._g === S.gkey) return;
      el._g = S.gkey;
      x.v.className = "chip " + g.verdict;
      set(x.v, ({ rust: "Rust", clean: "Clean", cannot_tell: "Can't tell" }[g.verdict] || g.verdict) + "  " + (g.confidence || 0).toFixed(2));
      set(x.lat, Math.round(g.latency_ms || 0) + " ms"); x.bar.style.width = (g.confidence || 0) * 100 + "%"; set(x.why, g.reason);
      if (!S.inspection) return;
      const I = S.inspection, r = I.w / 40;
      x.img.src = "data:image/jpeg;base64," + I.jpg;
      x.pts.setAttribute("viewBox", `0 0 ${I.w} ${I.h}`); x.pts.setAttribute("preserveAspectRatio", "xMidYMid slice");
      x.pts.innerHTML = (g.points || []).map(([px, py]) => `<circle cx="${px}" cy="${py}" r="${r}" stroke-width="${r / 4}"/>`).join("");
    } });
  const logCls = (e) => (tone(e.to) === "stop" || /GAS/.test(e.to) ? "stop" : tone(e.to) === "warn" || /RE-PLAN|REPLAN/.test(e.reason + e.to) ? "warn" : /DONE|APPROVED/.test(e.to) ? "ok" : "");
  H.logRows = (S, n) => S.log.slice(0, n).map((e) => `<li><span class="tnum">${Number(e.t).toFixed(1)}</span><b class="${logCls(e)}">${esc(H.nice(e.to))}</b><span>${e.from && !/^[A-Z_]+$/.test(e.from) ? `<i>${esc(e.from)}</i> ` : ""}${esc(String(e.reason || "").replace(/->/g, "→"))}</span></li>`).join("");
  H.panel({ id: "log", title: "Decisions", icon: "list", key: "l", edge: "bottom", phases: ["PAUSE_PERSON", "PAUSED"], priority: 45, keys: "log history events reasons why",
    desc: "Every state change and the reason the planner gave.", hint: (S) => (S.log.length ? S.log.length + " recent" : ""),
    render(el, S) { const x = q(el, `<ol class="log" data-k="l"></ol>`); if (el._n !== S.logN) { el._n = S.logN; x.l.innerHTML = H.logRows(S, 4); } } });
  const gasRows = (S) => (S.gas?.sensors || []).map((s) => { const v = (s.lel || 0) * 100, c = v >= 20 ? "stop" : v >= 10 ? "warn" : ""; return `<div class="gas-row ${c}"><span>${esc(s.id)}</span><span class="dim">${esc(s.at || "")}${s.ppm != null ? " · " + Math.round(s.ppm) + " ppm" : ""}</span><i><s style="width:${Math.min(100, v * 2)}%"></s></i><b class="tnum">${v.toFixed(1)}</b></div>`; }).join("");
  const gasMore = (g) => {
    if (!g) return "";
    const a = g.acoustic, src = g.source, out = [];
    if (g.h2s_ppm != null) out.push(`H2S ${g.h2s_ppm.toFixed(1)} ppm`);
    if (a) out.push(`US ${Math.round(a.db)} dB${a.snr_db > 0 ? ` (+${Math.round(a.snr_db)})` : ""}${a.bearing_deg != null ? ` → ${Math.round(a.bearing_deg)}°` : ""}`);
    if (src) out.push(`source ${src.x.toFixed(1)}, ${src.y.toFixed(1)} ±${src.sigma_m.toFixed(1)} m`);
    if (g.flow_gs != null) out.push(`flow ${g.flow_gs.toFixed(1)} g/s`);
    if (g.mode) out.push(String(g.mode).toLowerCase());
    return out.join(" · ");
  };
  H.panel({ id: "gas", title: "Gas sensors", meta: "% LEL", icon: "wind", key: "g", edge: "left", priority: 80, keys: "leak lel h2s detector methane",
    phases: (S) => S.gasAlarm && (!S.control || now() - S.gasAt < 6 / S.speed), desc: "Fixed and on-robot gas detectors. Alarm at 10 % LEL, retreat at 20 %.",
    hint: (S) => (S.gas ? (S.gasMax ? pct(S.gasMax, 1) + " LEL" : S.gasAlarm ? "Alarm" : "Clear") : ""),
    render(el, S) {
      const x = q(el, `<div data-k="rows"></div><p class="why" data-k="more"></p><canvas class="gas-chart" data-k="c"></canvas>`), rows = gasRows(S) || `<p class="none">No gas sensors reporting</p>`;
      if (x.rows._h !== rows) { x.rows._h = rows; x.rows.innerHTML = rows; }
      set(x.more, gasMore(S.gas));
      spark(x.c, S.series.gas, { min: 0, max: 30, color: S.gasAlarm ? "#F85149" : "rgba(255,255,255,.7)", refs: [[10, "rgba(227,179,65,.5)"], [20, "rgba(248,81,73,.5)"]] });
    } });
  function drawPlume(c, S) {
    const [g, w, h] = fit(c);
    if (!g) return;
    const L = Math.min(1, S.gasMax / 0.2), a = ((S.gas?.wind?.dir_deg ?? 225) + 180) * Math.PI / 180, t = now(), sx = w * 0.32, sy = h * 0.6;
    g.strokeStyle = "rgba(255,255,255,.1)"; g.beginPath(); g.roundRect(w * 0.05, h * 0.08, w * 0.9, h * 0.84, 10); g.stroke();
    for (let k = 0; L > 0.02 && k < 7; k++) {
      const d = k / 6, px = sx + Math.sin(a) * d * w * 0.45 + Math.sin(t + k) * 3, py = sy - Math.cos(a) * d * h * 0.6 + Math.cos(t * 0.7 + k) * 3, r = (10 + d * 46) * (0.4 + L * 0.6);
      const gr = g.createRadialGradient(px, py, 0, px, py, r);
      gr.addColorStop(0, `rgba(248,81,73,${(0.7 * (1 - d * 0.6) * L).toFixed(3)})`); gr.addColorStop(1, "rgba(248,81,73,0)");
      g.fillStyle = gr; g.beginPath(); g.arc(px, py, r, 0, 7); g.fill();
    }
    g.fillStyle = L > 0.5 ? "#F85149" : "rgba(255,255,255,.6)"; g.beginPath(); g.arc(sx, sy, 3, 0, 7); g.fill();
    g.font = "500 11px -apple-system, system-ui, sans-serif"; g.fillStyle = "rgba(235,235,245,.5)";
    g.fillText(S.gas?.sensors?.[0]?.at || "Source", sx + 7, sy + 4);
  }
  H.panel({ id: "plume", title: "Gas plume", icon: "cloud", key: "u", suggest: (S) => (S.gasAlarm ? 0.8 : 0), keys: "leak dispersion wind pip camera",
    desc: "The plume camera when the sim streams one (header pip); otherwise an estimate from the detectors and the wind.",
    hint: (S) => (S.gasAlarm ? "Spreading" : "Clear"),
    render(el, S) {
      const p = S.pip?.jpg, x = q(el, p ? `<div class="pip on"><img data-k="img" alt=""></div><p class="why" data-k="cap"></p>` : `<canvas class="plume" data-k="c"></canvas><p class="why" data-k="cap"></p>`);
      if (p && x.img._j !== p) { x.img._j = p; x.img.src = "data:image/jpeg;base64," + p; }
      if (!p) drawPlume(x.c, S);
      const wd = S.gas?.wind;
      set(x.cap, S.pip?.caption || (wd ? `Wind from ${wd.dir_deg}° at ${wd.speed_ms} m/s` : "No wind data"));
    } });
  const acted = new Set(), sent = (x) => (x ? x[0] + x.slice(1).toLowerCase() : "");
  const ctl = (S) => {
    const c = S.control, cut = S.linkCut || c?.link === "CUT", ag = c?.agent || S.intent ? { ...(c?.agent || {}), intent: S.intent || c?.agent?.intent } : null;
    return !c ? null : { ...c, agent: ag && (cut ? { ...ag, link: "CUT" } : ag), ...(cut ? { mode: "EDGE", banner: "Gemini link cut — edge only", link: "CUT" } : {}) };
  };
  H.ctl = ctl;
  window.hudControl = () => ctl(H.S);
  function act(a) {
    H.event("control_action", { id: a.id, robot: a.robot }, () => { acted.add(a.id); H.addLog({ t: H.S.t, from: "Control centre", to: "ACTION", reason: a.label }); });
  }
  H.on((h) => {
    if (!h.control) return;
    if (h.control.banner && h.control.banner !== H._cb) H.banner("warn", sent(h.control.banner), "Robots keep their last plan on the edge", "link");
    H._cb = h.control.banner;
    const key = (h.control.actions || []).map((a) => a.id).join();
    if (key === H._ck) return;
    H._ck = key;
    [...Array(9).keys()].forEach((n) => H.uncommand("cc." + n));
    (h.control.actions || []).slice(0, 5).forEach((a, n) => H.command({ id: "cc." + n, group: "Control centre", label: a.label, icon: a.source === "edge" ? "warn" : "target", keys: "action plan argon " + (a.robot || ""),
      desc: `${a.source === "edge" ? "Edge safety rule" : "Ranked " + (a.rank || n + 1) + " by Gemini 4 Argon"}. ${a.rationale || ""}`, hint: `${a.source === "edge" ? "Edge" : "Argon"} · ${a.source === "edge" ? "hard" : Math.round(a.confidence * 100)}`,
      score: () => (n === 0 && (H.S.gasAlarm || h.control.banner) ? 0.95 : 0), active: () => acted.has(a.id), run: () => act(a) }));
  });
  const hot = (S) => {
    const c = S.control, an = c?.anomalies || [];
    const fh = H.fusionHot ? H.fusionHot(S) : 0;
    if (!c && !fh && S.state !== "ESCALATE_TO_HUMAN") return 0;
    if (!fh && S.gasAlarm && now() - S.gasAt < 6 / S.speed) return 0;
    if (!c) return fh || 90;
    return S.gasAlarm || S.linkCut || c?.banner || S.state === "ESCALATE_TO_HUMAN" || an.some((a) => a.severity === "critical") ? 90 : an.some((a) => /warn/.test(a.severity)) ? 50 : 0;
  };
  const ccRow = (a, n) => `<li data-n="${n}" class="${acted.has(a.id) ? "done" : ""}"><span class="tnum">${a.rank || n + 1}</span><span>${esc(a.label)}<small>${a.source === "edge" ? "Edge rule" : "Argon"}${a.rationale ? " · " + esc(a.rationale) : ""}</small></span><b class="tnum">${a.source === "edge" ? "hard" : Math.round((a.confidence || 0) * 100)}</b></li>`;
  const healthWhy = (f) => {
    const e = f.evidence || {}, u = e.unit ? " " + e.unit : "";
    if (f.hazard) return "beside live gas: treated as gas";
    if (f.status === "STUCK") return `flat at ${e.value}${u}, σ ${e.std} vs ${e.nbr_std ?? "–"}`;
    if (f.status === "DEAD") return `silent ${e.silent_s} s`;
    if (f.status === "DRIFT") return `bias ${e.bias > 0 ? "+" : ""}${e.bias}${u} > ${e.tol}`;
    if (f.status === "NOISY") return `noise ${e.ratio}× its peers`;
    return `reads ${e.value}${u}`;
  };
  const health = (h) => (!h || !h.total ? "" : `<div class="cc-health"><p class="hh"><b>Sensor health</b><span class="tnum ${h.faults.length ? "warn" : ""}">${h.ok}/${h.total} OK</span></p>`
    + h.faults.slice(0, 3).map((f) => `<p class="hf ${f.hazard || f.status === "DEAD" ? "stop" : ""}"><span class="hs">${f.status === "OUT_OF_RANGE" ? "RANGE" : f.status}</span><span class="id">${esc(f.id)}</span><span class="ev">${esc(healthWhy(f))}</span><span class="jb tnum">${esc((f.job || "–").replace("sensor_maintenance-", "maint ").replace("verify_sensor-", "verify "))}</span></p>`).join("") + `</div>`);
  H.planner = (ag, n) => (ag && (ag.intent || ag.calls?.length) ? `<div class="cc-agent">${ag.intent ? `<p class="intent">“${esc(ag.intent)}”</p>` : ""}${(ag.calls || []).slice(-n).reverse().map((x) =>
    `<p class="call ${x.ok ? "" : "no"}"><b>${x.ok ? "✓" : "✗"}</b>${esc(x.label)}<small>${esc(x.ok ? x.why : "refused: " + x.refused)}</small></p>`).join("")}</div>` : "");
  H.vetoes = (c, n) => (c?.vetoed || []).slice(0, n).map((v) => `<p class="veto"><b>Edge veto</b> <s>${esc(v.label)}</s><small>${esc(v.why)}</small></p>`).join("");
  H.panel({ id: "control", title: "Control centre", meta: "Gemini 4 Argon", icon: "target", key: "a", priority: hot, phases: (S) => hot(S) > 0, keys: "situation report gemini argon actions plan edge link",
    desc: "Gemini 4 Argon's situation report and ranked actions, the edge rules that override it, and the planner's last calls. Each action is also a palette command.",
    hint: (S) => { const c = ctl(S); return !c ? (H.fusionHint ? H.fusionHint(S) : "") : c.link === "CUT" ? "Edge only" : c.banner ? sent(c.banner) : c.key ? sent(c.key.replace(/[_+]/g, " ")) : "On track"; },
    render(el, S) {
      const x = q(el, `<form class="cc-order" data-k="f"><input data-k="i" placeholder="Give Argon an order…" autocomplete="off" aria-label="Order for Argon"><button type="submit">Send</button></form><div data-k="u"></div><div data-k="b"></div>`);
      if (!x.f._on) {
        x.f._on = true;
        x.i.addEventListener("keydown", (e) => e.stopPropagation());
        x.f.addEventListener("submit", (e) => {
          e.preventDefault();
          const text = x.i.value.trim();
          if (!text) return;
          H.post("intent", { text });
          H.addLog({ t: S.t, from: "Operator", to: "ORDER", reason: text });
          S.intent = text;
          x.i.value = "";
          x.f.classList.add("sent");
          setTimeout(() => x.f.classList.remove("sent"), 1200);
        });
      }
      const fu = H.fusion ? H.fusion(S) : "";
      if (x.u._h !== fu) { x.u._h = fu; x.u.innerHTML = fu; }
      const c = ctl(S), key = JSON.stringify(c) + acted.size + !fu;
      if (x.b._k === key) return;
      x.b._k = key;
      x.b.innerHTML = !c ? (fu ? "" : `<p class="none">Waiting for a situation report</p>`) : `<div class="cc-meta"><span class="badge ${c.mode === "CLOUD" ? "cloud" : ""}">${c.mode === "CLOUD" ? "Cloud" : "Edge"}</span>${c.latency_ms != null ? `<span class="tnum">${(c.latency_ms / 1000).toFixed(1)} s</span>` : ""}${c.cost_today_usd != null ? `<span class="tnum">$${c.cost_today_usd.toFixed(3)} today · ${c.calls_today || 0} calls</span>` : ""}</div>`
        + (c.banner ? `<p class="edge">${esc(sent(c.banner))}</p>` : "") + health(c.health)
        + `<div class="cc-sum">${(c.situation?.length ? c.situation : [c.summary || "Edge safety rules only; no cloud plan yet."]).slice(0, 3).map((l) => `<p>${esc(l)}</p>`).join("")}</div>`
        + `<ol class="cc-acts">${(c.actions || []).slice(0, 3).map(ccRow).join("")}</ol>` + H.vetoes(c, 1) + H.planner(c.agent, 1);
      x.b.onclick = (e) => { const li = e.target.closest("[data-n]"); if (li && c) act(c.actions[+li.dataset.n]); };
    } });
  function bounds(m, rs) {
    if (m?.bounds) return m.bounds;
    const xs = rs.map((r) => r.pose.x), ys = rs.map((r) => r.pose.y);
    return { xmin: Math.min(...xs, -10) - 3, xmax: Math.max(...xs, 10) + 3, ymin: Math.min(...ys, -10) - 3, ymax: Math.max(...ys, 10) + 3 };
  }
  H.drawRig = function (c, S, o = {}) {
    const [g, w, h] = fit(c), m = S.fleetMap, f = S.fleet, rs = (f?.robots || []).filter((r) => r.pose);
    if (!g) return [];
    const b = bounds(m, rs), k = Math.min((w - 24) / (b.xmax - b.xmin), (h - 24) / (b.ymax - b.ymin)), ox = (w - k * (b.xmax - b.xmin)) / 2, oy = (h - k * (b.ymax - b.ymin)) / 2;
    const X = (x) => ox + (x - b.xmin) * k, Y = (y) => h - oy - (y - b.ymin) * k, big = !!o.big;
    const count = {};
    rs.forEach((r) => r.deck !== "air" && (count[r.deck] = (count[r.deck] || 0) + 1));
    const deck = o.deck && o.deck !== "auto" ? o.deck : Object.keys(count).sort((a, z) => count[z] - count[a])[0] || "top_deck";
    (m?.walkways || []).forEach((s) => {
      const on = deck === "all" || s.deck === deck;
      g.strokeStyle = on ? (s.kind === "stairs" ? "rgba(227,179,65,.4)" : "rgba(235,235,245,.28)") : "rgba(235,235,245,.06)";
      g.lineWidth = on ? (big ? 2.5 : 1.6) : 1;
      g.beginPath(); g.moveTo(X(s.a[0]), Y(s.a[1])); g.lineTo(X(s.b[0]), Y(s.b[1])); g.stroke();
    });
    if (m?.dock) { const [dx, dy] = m.dock.xyz; g.strokeStyle = COL.act; g.lineWidth = 1.2; g.strokeRect(X(dx) - 5, Y(dy) - 5, 10, 10); if (big) { g.fillStyle = "rgba(235,235,245,.5)"; g.font = "500 11px -apple-system, system-ui, sans-serif"; g.fillText("Dock", X(dx) + 9, Y(dy) + 4); } }
    if (big) (f?.tasks || []).filter((t) => t.status !== "done" && t.xyz).forEach((t) => {
      g.fillStyle = t.kind === "gas" ? COL.stop : t.kind === "rust" ? "#C4623A" : "rgba(235,235,245,.35)";
      const x = X(t.xyz[0]), y = Y(t.xyz[1]);
      g.beginPath(); g.moveTo(x, y - 4); g.lineTo(x + 4, y); g.lineTo(x, y + 4); g.lineTo(x - 4, y); g.fill();
    });
    (f?.zones || []).forEach((z) => {
      const x = X(z.xyz[0]), y = Y(z.xyz[1]), r = Math.max(6, z.radius * k);
      if (z.kind === "gas") { const gr = g.createRadialGradient(x, y, 0, x, y, r * 1.4); gr.addColorStop(0, "rgba(248,81,73,.55)"); gr.addColorStop(1, "rgba(248,81,73,0)"); g.fillStyle = gr; g.beginPath(); g.arc(x, y, r * 1.4, 0, 7); g.fill(); }
      else { g.strokeStyle = z.kind === "person" ? COL.warn : "rgba(235,235,245,.4)"; g.setLineDash(z.kind === "no_fly" ? [4, 4] : []); g.lineWidth = 1.2; g.beginPath(); g.arc(x, y, r, 0, 7); g.stroke(); g.setLineDash([]); }
      if (z.kind === "person") { g.fillStyle = COL.warn; g.beginPath(); g.arc(x, y, 2.5, 0, 7); g.fill(); }
    });
    if (S.crate?.xyz) {
      const x = X(S.crate.xyz[0]), y = Y(S.crate.xyz[1]), sz = Math.max(big ? 8 : 6, (S.crate.size?.[0] || 0.5) * k);
      g.fillStyle = COL.warn; g.fillRect(x - sz / 2, y - sz / 2, sz, sz);
      g.fillStyle = "rgba(255,255,255,.85)"; g.font = `500 ${big ? 11 : 10}px -apple-system, system-ui, sans-serif`; g.textAlign = "center"; g.fillText(S.crate.kind || "object", x, y + sz / 2 + (big ? 13 : 11)); g.textAlign = "start";
    }
    const hits = [];
    rs.forEach((r) => {
      if (deck !== "all" && r.deck !== deck && r.deck !== "air") return;
      const x = X(r.pose.x), y = Y(r.pose.y), col = COL[tone(r.job_state || r.state)], rr = big ? 6 : 4, a = -(r.pose.yaw || 0);
      if (S.focus === r.id) { g.strokeStyle = "rgba(255,255,255,.85)"; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, rr + 5, 0, 7); g.stroke(); }
      g.fillStyle = col; g.beginPath();
      if (r.deck === "air" || r.kind === "A") { g.moveTo(x, y - rr - 1); g.lineTo(x + rr, y + rr); g.lineTo(x - rr, y + rr); } else g.arc(x, y, rr, 0, 7);
      g.fill();
      g.strokeStyle = col; g.lineWidth = 1.6; g.beginPath(); g.moveTo(x + Math.cos(a) * rr, y + Math.sin(a) * rr); g.lineTo(x + Math.cos(a) * (rr + (big ? 9 : 6)), y + Math.sin(a) * (rr + (big ? 9 : 6))); g.stroke();
      g.fillStyle = "rgba(255,255,255,.85)"; g.font = `${big ? 600 : 500} ${big ? 12 : 10}px -apple-system, system-ui, sans-serif`;
      g.fillText(big ? rname(r) : rname(r).split(" ·")[0].replace("Unit ", "U"), x + rr + 4, y - rr);
      hits.push({ id: r.id, x, y });
    });
    return Object.assign(hits, { deck });
  };
  const fleetRows = (S) => (S.fleet?.robots || []).map((r) => `<div class="bot"><b>${esc(rname(r).split(" ·")[0])}</b><span class="dim">${esc(H.nice(r.job_state || r.state))}</span><i><s style="width:${pct(r.battery?.soc).replace(" ", "")};background:${(r.battery?.soc ?? 1) < 0.25 ? COL.warn : ""}"></s></i><em class="tnum">${r.battery?.soc == null ? "" : Math.round(r.battery.soc * 100)}</em></div>`).join("");
  H.panel({ id: "fleet", title: "Fleet", icon: "robot", key: "b", edge: "left", keys: "robots map battery deck units", suggest: (S) => ((S.fleet?.robots?.length || 0) > 1 ? 0.2 : 0),
    desc: "Every robot on the rig: where it is on its deck, what it is doing and its battery. ⇧M opens the full mission map.", hint: (S) => (S.fleet ? S.fleet.robots.length + " robots" : ""),
    render(el, S) {
      const x = q(el, `<canvas class="rig" data-k="c"></canvas><div data-k="rows"></div>`), rows = fleetRows(S);
      if (now() - (el._t || 0) > 0.1) { el._t = now(); H.drawRig(x.c, S); }
      if (x.rows._h !== rows) { x.rows._h = rows; x.rows.innerHTML = rows || `<p class="none">No fleet snapshot yet</p>`; }
    } });
})();
