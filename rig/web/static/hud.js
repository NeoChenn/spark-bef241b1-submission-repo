(() => {
  const Q = new URLSearchParams(location.search);
  const MOCK = Q.has("mock");
  const SPEED = parseFloat(Q.get("speed") || "1");
  const $ = (id) => document.getElementById(id);
  const RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const STEPS = ["PATROL", "DETECT", "SPRAY", "SOAK", "SCRUB", "VERIFY", "PAINT", "DONE"];
  const STEP_OF = { PATROL: 0, DETECT: 1, SECOND_VIEW: 1, APPROACH: 1, SPRAY: 2, SOAK: 3, SCRUB: 4, VERIFY: 5, PAINT: 6, VERIFY_PAINT: 6, DONE: 7 };
  const SAFETY = { BACK_OFF: "warn", HOLD: "warn", PAUSE_PERSON: "stop", ESCALATE_TO_HUMAN: "stop" };
  const LABEL = { ESCALATE_TO_HUMAN: "ESCALATE", PAUSE_PERSON: "PERSON", BACK_OFF: "BACK OFF", SECOND_VIEW: "SECOND VIEW", VERIFY_PAINT: "VERIFY PAINT" };
  const PERCEIVE = new Set(["DETECT", "SECOND_VIEW", "APPROACH", "SOAK", "VERIFY", "VERIFY_PAINT", "HOLD"]);
  const MODS = [
    { id: "hud", name: "HUD", key: "h" }, { id: "map", name: "Pipe map", key: "m" },
    { id: "gemini", name: "Gemini", key: "g" }, { id: "log", name: "Decision log", key: "l" },
    { id: "force", name: "Force sensor", key: "f" }, { id: "pip", name: "Inspection PiP", key: "p" },
    { id: "plume", name: "Gas plume", key: "v" },
    { id: "control", name: "Control centre", key: "o" },
    { id: "captions", name: "Captions", key: "t" },
  ];
  const ACTIONS = [
    { id: "rust", label: "New rust", key: "1", ev: () => ({ name: "spawn_patch", args: {} }) },
    { id: "glare", label: "Glare", key: "2", toggle: true, ev: (on) => ({ name: "set_glare", args: { on } }) },
    { id: "mist", label: "Lens mist", key: "3", toggle: true, ev: (on) => ({ name: "occlude_camera", args: { on } }) },
    { id: "bead", label: "Weld bead", key: "4", ev: () => ({ name: "force_spike", args: {} }) },
    { id: "worker", label: "Worker walks up", key: "5", toggle: true, ev: (on) => ({ name: "person_at", args: { distance: on ? 2.4 : null } }) },
    { id: "link", label: "Cut Gemini link", key: "6", toggle: true, ev: (on) => ({ name: "cut_link", args: { on } }) },
  ];
  const FORCE_LIMIT = 40, FORCE_CAP = 38, VITAL_WINDOW = 30, FORCE_WINDOW = 20;

  const S = { robots: [], camera: null, shot: 0, shotEl: null, state: null, t: 0, grid: null, sprayed: null, painted: null, target: null, brush: null, gemini: null, insp: null, peak: 0, log: [], started: false };
  const series = { force: [], rust: [], conf: [] };
  const now = () => performance.now() / 1000;
  const RAMP = [[0, [184, 182, 174]], [10, [184, 182, 174]], [18, [181, 200, 74]], [25, [63, 185, 80]], [35, [63, 185, 80]], [38, [227, 179, 65]], [40, [227, 179, 65]], [40.01, [248, 81, 73]]];
  function forceColor(f) {
    let k = 1;
    while (k < RAMP.length - 1 && f > RAMP[k][0]) k++;
    const [a, ca] = RAMP[k - 1], [b, cb] = RAMP[k];
    const u = Math.max(0, Math.min(1, (f - a) / (b - a || 1)));
    return `rgb(${ca.map((c, n) => Math.round(c + (cb[n] - c) * u)).join(",")})`;
  }

  const store = {
    get(k, d) { try { const v = localStorage.getItem("hud." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("hud." + k, JSON.stringify(v)); } catch {} },
  };
  const modsOn = store.get("mods", {});

  function setMod(id, on) {
    modsOn[id] = on;
    store.set("mods", modsOn);
    document.querySelectorAll(`.mod[data-mod="${id}"]`).forEach((el) => el.classList.toggle("off", !on));
    const li = document.querySelector(`#mods-list li[data-id="${id}"]`);
    if (li) li.classList.toggle("on", on);
  }

  function buildMods() {
    const ul = $("mods-list");
    MODS.forEach((m) => {
      const li = document.createElement("li");
      li.dataset.id = m.id;
      li.innerHTML = `<span class="sw"></span><span>${m.name}</span><kbd>${m.key.toUpperCase()}</kbd>`;
      li.onclick = () => setMod(m.id, !modsOn[m.id]);
      ul.appendChild(li);
      setMod(m.id, modsOn[m.id] !== false);
    });
    $("mods-toggle").onclick = () => $("mods").classList.toggle("open");
  }

  function buildSteps() {
    $("steps").innerHTML = STEPS.map((s) => `<li><span></span>${s}</li>`).join("");
  }

  function buildActions() {
    const body = $("u-body");
    ACTIONS.forEach((a) => {
      const b = document.createElement("button");
      b.innerHTML = `<span>${a.label}</span><kbd>⇧${a.key}</kbd>`;
      b.onclick = () => fire(a, b);
      a.btn = b;
      body.appendChild(b);
    });
    $("u-toggle").onclick = () => $("unscripted").classList.toggle("open");
  }

  const tweens = new Map();
  function num(el, v, digits = 1) {
    if (v == null || Number.isNaN(v)) { tweens.delete(el); el.textContent = "—"; el._v = null; return; }
    if (el._v == null || RM) { el._v = v; el.textContent = v.toFixed(digits); }
    tweens.set(el, { to: v, digits });
  }
  function stepTweens() {
    tweens.forEach((tw, el) => {
      el._v += (tw.to - el._v) * 0.2;
      if (Math.abs(tw.to - el._v) < Math.pow(10, -tw.digits) / 2) { el._v = tw.to; tweens.delete(el); }
      el.textContent = el._v.toFixed(tw.digits);
    });
  }

  function fitCanvas(c) {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = c.clientWidth || c.width, h = c.clientHeight || c.height;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [g, w, h];
  }

  function spark(c, data, o) {
    const [g, w, h] = fitCanvas(c);
    g.clearRect(0, 0, w, h);
    const t1 = now(), t0 = t1 - o.window, pad = 2;
    const x = (t) => ((t - t0) / o.window) * w;
    const y = (v) => h - pad - ((v - o.min) / (o.max - o.min)) * (h - 2 * pad);
    if (o.grid) {
      g.strokeStyle = "rgba(255,255,255,0.05)";
      g.lineWidth = 1;
      g.beginPath(); g.moveTo(0, h - 0.5); g.lineTo(w, h - 0.5); g.stroke();
    }
    (o.refs || []).forEach((r) => {
      g.strokeStyle = r.color; g.lineWidth = 1; g.setLineDash(r.dash || []);
      g.beginPath(); g.moveTo(0, Math.round(y(r.v)) + 0.5); g.lineTo(w, Math.round(y(r.v)) + 0.5); g.stroke();
    });
    g.setLineDash([]);
    const pts = data.filter((p) => p[0] >= t0 - 1);
    if (pts.length < 2) return;
    g.lineWidth = o.width || 1.25; g.lineJoin = g.lineCap = "round";
    const px = (p) => [x(p[0]), y(Math.min(o.max, p[1]))];
    if (o.ramp) {
      for (let k = 1; k < pts.length; k++) {
        g.strokeStyle = o.ramp(Math.max(pts[k - 1][1], pts[k][1]));
        g.beginPath(); g.moveTo(...px(pts[k - 1])); g.lineTo(...px(pts[k])); g.stroke();
      }
    } else {
      g.strokeStyle = o.color;
      g.beginPath();
      pts.forEach((p, k) => (k ? g.lineTo(...px(p)) : g.moveTo(...px(p))));
      g.stroke();
    }
    if (o.fill) {
      g.lineTo(x(pts[pts.length - 1][0]), h); g.lineTo(x(pts[0][0]), h); g.closePath();
      g.fillStyle = o.fill; g.fill();
    }
    const last = pts[pts.length - 1];
    g.fillStyle = o.ramp ? o.ramp(last[1]) : o.color;
    g.beginPath(); g.arc(Math.min(w - 2, x(last[0])), y(Math.min(o.max, last[1])), 2, 0, 7); g.fill();
  }

  function push(k, v) {
    if (v == null) return;
    const s = series[k], t = S.at || now();
    s.push([t, v]);
    while (s.length && s[0][0] < t - VITAL_WINDOW - 2) s.shift();
  }

  function drawCharts() {
    const css = getComputedStyle(document.documentElement);
    const accent = css.getPropertyValue("--accent").trim();
    spark($("sp-force"), series.force, { window: VITAL_WINDOW, min: 0, max: 70, ramp: forceColor, grid: 1,
      refs: [{ v: FORCE_LIMIT, color: "rgba(248,81,73,0.6)", dash: [3, 3] }] });
    spark($("sp-rust"), series.rust, { window: VITAL_WINDOW, min: 0, max: 100, color: "#D9825C", grid: 1, fill: "rgba(181,83,42,0.12)" });
    spark($("sp-conf"), series.conf, { window: VITAL_WINDOW, min: 0, max: 1, color: accent, grid: 1 });
    const fmax = Math.max(50, S.peak * 1.15);
    spark($("fm-chart"), series.force, { window: FORCE_WINDOW, min: 0, max: fmax, ramp: forceColor, width: 1.6,
      refs: [{ v: FORCE_CAP, color: "rgba(245,244,238,0.18)" }, { v: FORCE_LIMIT, color: "rgba(248,81,73,0.9)", dash: [4, 3] }] });
  }

  function drawMap() {
    const c = $("map-canvas");
    const [g, w, h] = fitCanvas(c);
    g.fillStyle = "#1c1c1b";
    g.fillRect(0, 0, w, h);
    if (!S.grid) return;
    const rows = S.grid.length, cols = S.grid[0].length, cw = w / cols, ch = h / rows;
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const r = S.grid[i][j];
        const painted = S.painted && S.painted[i][j] === "1";
        const sprayed = S.sprayed && S.sprayed[i][j] === "1";
        let col = null;
        if (painted) col = r > 0.08 ? `rgba(181,83,42,${0.35 + r * 0.6})` : "rgba(143,168,148,0.92)";
        else if (r > 0.02) col = `rgba(181,83,42,${0.25 + r * 0.75})`;
        if (col) { g.fillStyle = col; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }
        if (sprayed && !painted) { g.fillStyle = "rgba(122,164,222,0.38)"; g.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5); }
      }
    }
    g.strokeStyle = "rgba(245,244,238,0.05)";
    g.lineWidth = 1;
    g.beginPath();
    for (let j = 8; j < cols; j += 8) { g.moveTo(Math.round(j * cw) + 0.5, 0); g.lineTo(Math.round(j * cw) + 0.5, h); }
    for (let i = 10; i < rows; i += 10) { g.moveTo(0, Math.round(i * ch) + 0.5); g.lineTo(w, Math.round(i * ch) + 0.5); }
    g.stroke();
    if (S.brush && ["SCRUB", "SPRAY", "PAINT", "BACK_OFF"].includes(S.state)) {
      const bx = (S.brush[1] + 0.5) * cw, by = (S.brush[0] + 0.5) * ch;
      g.strokeStyle = "rgba(245,244,238,0.3)";
      g.beginPath(); g.moveTo(0, by); g.lineTo(bx - 7, by); g.moveTo(bx + 7, by); g.lineTo(w, by);
      g.moveTo(bx, 0); g.lineTo(bx, by - 7); g.moveTo(bx, by + 7); g.lineTo(bx, h); g.stroke();
      g.strokeStyle = "#F5F4EE"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(bx, by, 4.5, 0, 7); g.stroke();
    }
  }

  const tagTimers = new Map();
  function tag(text, cls, ms) {
    let el = [...$("tags").children].find((e) => e.dataset.k === text);
    if (!el) {
      el = document.createElement("span");
      el.dataset.k = text;
      el.className = "tag " + (cls || "");
      el.textContent = text;
      $("tags").appendChild(el);
    }
    clearTimeout(tagTimers.get(text));
    if (ms) tagTimers.set(text, setTimeout(() => el.remove(), ms));
    return el;
  }
  function untag(text) {
    const el = [...$("tags").children].find((e) => e.dataset.k === text);
    if (el) el.remove();
  }

  function setState(st) {
    if (st === S.state) return;
    const prev = S.state;
    S.state = st;
    $("st-label").textContent = LABEL[st] || st;
    $("st-name").className = "state chip-state " + (SAFETY[st] || (st === "DONE" ? "ok" : ""));
    $("steps").classList.toggle("ok", st === "DONE");
    if (SAFETY[prev]) untag(LABEL[prev] || prev);
    if (SAFETY[st]) tag(LABEL[st] || st, SAFETY[st] === "stop" ? "stop" : "");
    if (st in STEP_OF) {
      const k = STEP_OF[st];
      [...$("steps").children].forEach((li, n) => { li.className = n < k ? "done" : n === k ? "cur" : ""; });
    }
    if (st === "DONE") untag("RE-PLAN");
    if (st === "DONE" && MOCK) setTimeout(() => { if (S.state === "DONE") shot(1); }, 3500);
  }

  function addEvents(events) {
    events.forEach((e) => {
      S.log.unshift(e);
      if (/^RE-PLAN/.test(e.reason || "")) tag("RE-PLAN", "re", 6000);
    });
    S.log = S.log.slice(0, 6);
    if (!events.length) return;
    $("log-list").innerHTML = S.log.map((e) => {
      const cls = SAFETY[e.to] === "stop" ? "s" : SAFETY[e.to] || /^RE-PLAN/.test(e.reason || "") ? "w" : e.to === "DONE" ? "k" : "";
      const t = Number(e.t).toFixed(1).padStart(5, " ");
      return `<li class="${cls}"><span>${t}</span><span class="tr">${short(e.from)} → <b>${short(e.to)}</b></span><span class="why">${esc(e.reason || "")}</span></li>`;
    }).join("");
  }
  const short = (s) => LABEL[s] || s;
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  let geminiTimer = null;
  function showGemini(gm) {
    const key = JSON.stringify(gm);
    if (!gm || key === S.geminiKey) return;
    S.geminiKey = key;
    const v = gm.verdict || "cannot_tell";
    const chip = $("g-verdict");
    chip.className = "chip " + v;
    chip.textContent = { rust: "Rust", clean: "Clean", cannot_tell: "Can't tell" }[v] || v;
    $("g-lat").textContent = Math.round(gm.latency_ms || 0) + " ms";
    num($("g-conf"), gm.confidence || 0, 2);
    $("g-bar").style.width = Math.round((gm.confidence || 0) * 100) + "%";
    $("g-reason").textContent = gm.reason || "";
    if (S.insp) {
      $("g-img").src = "data:image/jpeg;base64," + S.insp.jpg;
      const svg = $("g-pts");
      svg.setAttribute("viewBox", `0 0 ${S.insp.w} ${S.insp.h}`);
      svg.setAttribute("preserveAspectRatio", "xMidYMid slice");
      const r = S.insp.w / 40;
      svg.innerHTML = (gm.points || []).map(([x, y]) => `<circle cx="${x}" cy="${y}" r="${r}" stroke-width="${r / 4}"/>`).join("");
    }
    $("gemini").classList.remove("hide");
    clearTimeout(geminiTimer);
    geminiTimer = setTimeout(() => $("gemini").classList.add("hide"), 11000 / (MOCK ? SPEED : 1));
  }

  function showPlume(p) {
    $("plume").classList.remove("hide");
    if (p.image && p.seq !== S.plumeSeq) {
      S.plumeSeq = p.seq;
      $("plume-img").src = "/api/plume.jpg?s=" + p.seq;
    }
    const lel = (p.stats && p.stats.max_lel_pct) || 0;
    const tag = $("pl-tag");
    tag.className = "pl-tag " + (lel >= 50 ? "stop" : lel >= 10 ? "warn" : "");
    tag.textContent = `${p.flange || ""} ${Math.round(lel)}% LEL`;
  }

  const CC_TOP = 3;
  const CUT = { mode: "EDGE", banner: "GEMINI LINK CUT — EDGE ONLY", pending: false, link: "CUT" };
  function showControl(raw) {
    S.control = raw;
    const c = S.linkCut ? { ...raw, ...CUT, agent: raw.agent && { ...raw.agent, link: "CUT" } } : raw;
    S.controlView = c;
    const mode = $("cc-mode");
    mode.textContent = c.mode;
    mode.className = "badge" + (c.mode === "CLOUD" ? " cloud" : "") + (c.pending ? " pending" : "");
    $("cc-lat").textContent = c.latency_ms != null ? `${(c.latency_ms / 1000).toFixed(1)} s` : "—";
    $("cc-cost").textContent = `$${(c.cost_today_usd || 0).toFixed(3)} today · ${c.calls_today || 0} calls`;
    $("cc-model").textContent = c.model ? "Gemini 4 Argon" + (c.replay ? " · replay" : "") : "edge only";
    $("cc-banner").textContent = c.banner || "";
    $("cc-banner").classList.toggle("on", !!c.banner);
    const sit = c.situation && c.situation.length ? c.situation : ["Edge safety rules only; no cloud plan yet."];
    const sitKey = sit.join("|");
    if (sitKey !== S.ccSit) { S.ccSit = sitKey; $("cc-sit").innerHTML = sit.map((l) => `<p>${esc(l)}</p>`).join(""); }
    $("cc-acts").innerHTML = (c.actions || []).slice(0, CC_TOP).map((a) => `<li class="${a.source}">
      <span class="rk">${a.rank}</span><span class="lb"><span class="src">${a.source === "edge" ? "Edge rule" : "Argon"}</span>${esc(a.label)}</span>
      <span class="cf">${a.source === "edge" ? "HARD" : a.confidence.toFixed(2)}</span>
      <div class="bar"><span style="width:${Math.round(a.confidence * 100)}%"></span></div>
      <span class="why">${esc(a.rationale || "")}</span></li>`).join("");
    const gas = (c.gas || []).filter((g) => g.lel >= 5).map((g) =>
      `<span class="${g.lel >= 50 ? "critical" : g.lel >= 10 ? "warn" : ""}">${esc(g.id)} ${g.lel.toFixed(0)}% LEL${g.trend > 3 ? " ▲" : g.trend < -3 ? " ▼" : ""}${g.db != null ? " · " + Math.round(g.db) + " dB" : ""}</span>`);
    const chips = (c.anomalies || []).map((a) => `<span class="${a.severity}">${esc(a.kind.replace(/_/g, " "))} · ${esc(a.zone)}</span>`);
    $("cc-chips").innerHTML = chips.concat(gas).join("");
    const veto = (c.vetoed || []).slice(0, 2).map((v) => `<span class="veto"><b>Edge veto</b> <s>${esc(v.label)}</s><em>${esc(v.why)}</em></span>`);
    $("cc-foot").innerHTML = veto.join("");
    const ag = c.agent;
    $("cc-agent").innerHTML = !ag ? "" : `<div class="ag-head"><i>Mission planner</i><span class="${ag.link === "CUT" ? "cut" : ""}">link ${ag.link}</span></div>` +
      (ag.intent ? `<p class="ag-intent">“${esc(ag.intent)}”</p>` : "") +
      (ag.calls || []).slice(-2).reverse().map((x) => `<div class="ag-call ${x.ok ? "" : "no"}"><b>${x.ok ? "✓" : "✗"}</b><span>${esc(x.label)}</span><em>${esc(x.ok ? x.why : "refused: " + x.refused)}</em></div>`).join("");
  }
  window.hudControl = () => S.controlView || null;

  const NUM = /\d+(?:\.\d+)?/g, LIFT = 0.85, EDGE = 28;
  const CAP = { key: null, c: null, nums: [], layer: 0 };
  function setCaption(c) {
    const box = $("subtitle");
    box.classList.toggle("hide", !c);
    if (!c) { CAP.key = null; return; }
    const whole = c.lead + "\u0001" + c.rest;
    const key = c.tone + whole.replace(NUM, "#");
    const nums = whole.match(NUM) || [];
    if (key === CAP.key) { CAP.nums.forEach((n, k) => { n.to = +nums[k]; if (RM) n.v = n.to; }); return; }
    CAP.key = key; CAP.c = c;
    CAP.nums = nums.map((x) => ({ v: +x, to: +x, d: (x.split(".")[1] || "").length }));
    const [a, b] = box.children;
    CAP.layer ^= 1;
    const on = CAP.layer ? b : a, off = CAP.layer ? a : b;
    on.className = "sub on " + c.tone;
    renderCaption(on);
    off.classList.remove("on");
    setTimeout(() => { if (!off.classList.contains("on")) off.innerHTML = ""; }, 170);
  }
  function renderCaption(el) {
    let k = 0;
    const fill = (t) => esc(t.replace(NUM, () => { const n = CAP.nums[k++]; return n.v.toFixed(n.d); }));
    el.innerHTML = `<b>${fill(CAP.c.lead)}</b>${fill(CAP.c.rest)}`;
  }
  function stepCaption() {
    if (!CAP.c || !CAP.nums.some((n) => n.v !== n.to)) return;
    CAP.nums.forEach((n) => { n.v += (n.to - n.v) * 0.16; if (Math.abs(n.to - n.v) < Math.pow(10, -n.d) / 2) n.v = n.to; });
    renderCaption($("subtitle").children[CAP.layer]);
  }

  function placeSubtitle() {
    const box = $("subtitle"), pill = $("status-pill").getBoundingClientRect(), st = $("stepper").getBoundingClientRect();
    const cx = innerWidth / 2, room = 2 * Math.min(cx - pill.right, st.left - cx) - 32;
    const low = box.scrollWidth > room;
    box.style.top = low ? `${Math.max(pill.bottom, st.bottom) + 12}px` : "";
  }

  function project(cam, p) {
    const T = cam.T, K = cam.K, d = [p[0] - T[0][3], p[1] - T[1][3], p[2] + LIFT - T[2][3]];
    const c = [0, 1, 2].map((j) => d[0] * T[0][j] + d[1] * T[1][j] + d[2] * T[2][j]);
    return [K[0][0] * c[0] / c[2] + K[0][2], K[1][1] * c[1] / c[2] + K[1][2], c[2]];
  }
  function frameRect(cam) {
    const el = MOCK ? S.shotEl : $("view");
    const r = el ? el.getBoundingClientRect() : { left: 0, top: 0, width: innerWidth, height: innerHeight };
    const k = Math.max(r.width / cam.w, r.height / cam.h);
    return { k, x: r.left + (r.width - cam.w * k) / 2, y: r.top + (r.height - cam.h * k) / 2 };
  }
  function labelEl(r) {
    let el = document.querySelector(`.lbl[data-id="${r.id}"]`);
    if (!el) {
      el = document.createElement("div");
      el.className = "lbl";
      el.dataset.id = r.id;
      el.innerHTML = `<span class="l-stem"></span><span class="l-pin"></span><span class="l-arrow"></span><div class="l-card"><b><i class="l-dot"></i><span class="l-name"></span></b><span class="l-txt"></span></div>`;
      $("labels").appendChild(el);
    }
    return el;
  }
  function placeLabels() {
    const multi = S.robots.length > 1 && S.camera;
    const keep = new Set(multi ? S.robots.map((r) => r.id) : []);
    [...$("labels").children].forEach((el) => { if (!keep.has(el.dataset.id)) el.remove(); });
    if (!multi) return;
    const cam = S.camera, f = frameRect(cam), W = innerWidth, H = innerHeight;
    S.robots.forEach((r) => {
      if (!r.world_pos) return;
      const el = labelEl(r), card = el.lastChild;
      const [u, v, z] = project(cam, r.world_pos);
      let x = f.x + u * f.k, y = f.y + v * f.k, mode = z <= 0.1 ? "behind" : r.occluded ? "occ" : "on";
      if (mode !== "behind" && (x < EDGE || x > W - EDGE || y < EDGE || y > H - EDGE)) {
        const dx = x - W / 2, dy = y - H / 2, s = Math.min((W / 2 - EDGE) / Math.abs(dx || 1e-9), (H / 2 - EDGE) / Math.abs(dy || 1e-9));
        x = W / 2 + dx * s; y = H / 2 + dy * s; mode = "edge";
        const a = Math.atan2(dy, dx), cw = card.offsetWidth, ch = card.offsetHeight;
        el.querySelector(".l-arrow").style.transform = `rotate(${a}rad)`;
        const cx = Math.min(Math.max(x - Math.cos(a) * (cw / 2 + 16), cw / 2 + 8), W - cw / 2 - 8);
        const cy = Math.min(Math.max(y - Math.sin(a) * (ch / 2 + 16), ch / 2 + 8), H - ch / 2 - 8);
        card.style.left = `${cx - x - cw / 2}px`; card.style.top = `${cy - y - ch / 2}px`;
      } else {
        card.style.left = card.style.top = "";
      }
      el.className = `lbl ${r.caption.tone} ${mode === "on" ? "" : mode}`;
      el.style.transform = `translate(${x}px, ${y}px)`;
      el.querySelector(".l-name").textContent = r.name;
      el.querySelector(".l-txt").textContent = r.caption.text;
    });
  }

  function apply(h) {
    if (h.control) showControl(h.control);
    if (h.robots) S.robots = h.robots;
    if (h.camera) S.camera = h.camera;
    if (h.robots || h.caption) {
      const one = S.robots.length <= 1 ? S.robots[0] : !S.camera ? S.robots[0] : null;
      const c = one ? one.caption : S.robots.length ? null : h.caption;
      setCaption(c && S.robots.length > 1 ? { ...c, lead: `${one.name} · ${c.lead}` } : c);
    }
    if (MOCK && h.shot && h.shot !== S.shot) shot(h.shot);
    if (h.state) setState(h.state);
    if (h.t != null) num($("st-time"), h.t, 1);
    if (h.events) addEvents(h.events);
    ["grid", "sprayed", "painted", "target"].forEach((k) => {
      const v = h[k === "grid" ? "rust_grid" : k];
      if (v) S[k] = v;
    });
    if (h.brush) S.brush = h.brush;
    if (h.force_n != null) {
      const f = S.spike ? Math.max(h.force_n, S.spike) : h.force_n;
      push("force", f);
      S.peak = Math.max(S.peak, f);
      num($("v-force"), f, 1);
      num($("fm-now"), f, 1);
      num($("fm-max"), S.peak, 1);
      $("fm-now").style.color = $("v-force").style.color = forceColor(f);
      $("fm-gauge").style.width = Math.min(100, (f / 50) * 100) + "%";
      $("fm-gauge").style.background = forceColor(f);
    }
    if (h.rust_fraction != null) { push("rust", h.rust_fraction * 100); num($("v-rust"), h.rust_fraction * 100, 1); $("v-rust-bar").style.width = Math.min(100, h.rust_fraction * 100) + "%"; }
    if (h.detector_conf != null) { push("conf", h.detector_conf); num($("v-conf"), h.detector_conf, 2); }
    if (h.source) $("st-src").className = "badge" + (h.source === "CLOUD" ? " cloud" : "");
    if (h.source) $("st-src").textContent = h.source;
    if (h.inspection) {
      S.insp = h.inspection;
      $("pip-img").src = "data:image/jpeg;base64," + h.inspection.jpg;
    }
    if (h.gemini) showGemini(h.gemini);
    if (h.plume) showPlume(h.plume);
    $("pip").classList.toggle("hide", !S.insp || !PERCEIVE.has(S.state));
    const soak = h.soak;
    $("soak").classList.toggle("hide", !soak);
    if (soak) {
      num($("soak-left"), soak.left, 0);
      $("soak-ring").style.strokeDashoffset = String(119.4 * (1 - soak.left / soak.total));
    }
  }

  function post(ev) {
    return fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(ev) })
      .then((r) => r.json()).catch(() => null);
  }

  function fire(a, btn) {
    a.on = a.toggle ? !a.on : true;
    btn.classList.toggle("fired", !!(a.toggle && a.on));
    if (!MOCK) { post(a.ev(a.on)); return; }
    const judge = (to, reason) => addEvents([{ t: lastSnapT(), from: "JUDGE", to, reason }]);
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
    if (a.id === "mist") { $("pip").classList.toggle("mist", a.on); judge(a.on ? "HOLD" : "VERIFY", a.on ? "perception lost (lens mist) -> hold position" : "camera view restored"); if (a.on) setState("HOLD"); }
    if (a.id === "bead") {
      S.spike = 64.8;
      setTimeout(() => { S.spike = 0; }, 400);
      judge("BACK_OFF", "contact force 65 N > 40 N limit -> retract");
      setState("BACK_OFF");
    }
    if (a.id === "link") { S.linkCut = a.on; judge(a.on ? "EDGE ONLY" : "CLOUD", a.on ? "Gemini link cut -> coordinator and robots keep the last plan" : "Gemini link restored"); if (S.control) showControl(S.control); }
    if (a.id === "worker") { judge(a.on ? "PAUSE_PERSON" : "SCRUB", a.on ? "person at 2.4 m < 3.0 m -> pause" : "person clear -> resume"); if (a.on) setState("PAUSE_PERSON"); }
  }

  let shotSide = 0;
  const SHOT_URL = (n) => `/api/mock/shot/${n}`;
  function shot(n) {
    if (!MOCK) { post({ name: "camera", args: { shot: n } }); return; }
    const a = $(shotSide ? "shotA" : "shotB"), b = $(shotSide ? "shotB" : "shotA");
    S.shot = n; S.shotEl = a;
    shotSide ^= 1;
    a.classList.remove("orbit");
    a.style.transition = "none";
    a.style.backgroundImage = `url(${SHOT_URL(n)})`;
    a.style.transform = n === 1 ? "scale(1.18)" : "scale(1.35) translateY(-4%)";
    void a.offsetWidth;
    a.style.transition = "";
    a.classList.add("on");
    a.style.transform = "scale(1.04)";
    b.classList.remove("on");
    if (n === 1) setTimeout(() => { a.style.transform = ""; a.classList.add("orbit"); }, 4000);
  }

  let session = null, playT0 = null, cursor = 0, offset = parseFloat(Q.get("t") || "0");
  const lastSnapT = () => (cursor > 0 ? session.snapshots[cursor - 1].t : 0);
  function playMock() {
    if (!session || playT0 == null) return;
    const el = (now() - playT0) * SPEED + offset;
    const snaps = session.snapshots;
    while (cursor < snaps.length && snaps[cursor].pt <= el) apply(snaps[cursor++]);
  }

  function seek(pt) {
    const snaps = session.snapshots;
    const back = now();
    while (cursor < snaps.length && snaps[cursor].pt <= pt) {
      const s = snaps[cursor++];
      S.at = back - (pt - s.pt) / SPEED;
      apply(s);
    }
    S.at = 0;
  }

  function begin() {
    if (S.started) return;
    S.started = true;
    $("title").classList.add("gone");
    shot(2);
    const reveal = Q.has("autostart") ? 0 : 3600;
    setTimeout(() => {
      const mods = [...document.querySelectorAll(".mod, .mods, .unscripted")];
      mods.forEach((el, k) => { el.style.transitionDelay = RM ? "0ms" : `${k * 110}ms`; });
      document.body.classList.remove("intro");
      setTimeout(() => mods.forEach((el) => { el.style.transitionDelay = ""; }), mods.length * 110 + 300);
      if (MOCK) {
        playT0 = now();
        if (offset && session) seek(offset);
      }
    }, reveal);
  }

  function liveState() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + "/ws/state");
    ws.onmessage = (ev) => { try { apply(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(liveState, 1500);
  }

  function buildHelp() {
    const rows = [["Space", "Start demo"], ["?  /  K", "Show / hide controls"], ["Tab", "Module list"], ["C", "Cinematic mode (hide all)"],
      ["U", "Unscripted change panel"], ["1  2  3", "Wide rig / hero deck / close-up"],
      ...MODS.map((m) => [m.key.toUpperCase(), m.name]), ...ACTIONS.map((a) => ["Shift+" + a.key, a.label])];
    $("help-list").innerHTML = rows.map(([k, v]) => `<li><kbd>${k}</kbd><span>${v}</span></li>`).join("");
  }

  function keys(e) {
    if (e.target.tagName === "INPUT") return;
    const k = e.key.toLowerCase();
    if (k === "?" || k === "k") { $("help").classList.toggle("open"); return; }
    if (k === "escape") { $("help").classList.remove("open"); return; }
    if (e.code === "Space") { e.preventDefault(); begin(); return; }
    if (k === "tab") { e.preventDefault(); $("mods").classList.toggle("open"); return; }
    if (k === "c") { document.body.classList.toggle("cine"); return; }
    if (k === "u") { $("unscripted").classList.toggle("open"); return; }
    const m = MODS.find((x) => x.key === k);
    if (m) { setMod(m.id, !modsOn[m.id]); return; }
    if (["1", "2", "3"].includes(k) && !e.shiftKey) { shot(+k); return; }
    const a = ACTIONS.find((x) => e.shiftKey && x.key === e.code.replace("Digit", ""));
    if (a) fire(a, a.btn);
  }

  let frames = 0, fpsT = now();
  function loop() {
    playMock();
    stepTweens();
    stepCaption();
    placeSubtitle();
    placeLabels();
    drawCharts();
    drawMap();
    frames++;
    if (MOCK && now() - fpsT > 0.5) { $("fps").textContent = String(Math.round(frames / (now() - fpsT))); frames = 0; fpsT = now(); }
    requestAnimationFrame(loop);
  }

  buildMods();
  buildSteps();
  buildHelp();
  buildActions();
  document.body.classList.add("intro");
  if (Q.get("title")) $("t-name").textContent = Q.get("title");
  $("soak").classList.add("hide");
  $("pip").classList.add("hide");
  $("gemini").classList.add("hide");
  addEventListener("keydown", keys);
  if (MOCK) {
    $("status").textContent = "mock replay";
    shot(1);
    fetch("/api/mock").then((r) => r.json()).then((s) => { s.snapshots = s.snapshots.concat(s.fleet || []); session = s; if (playT0 != null && offset) seek(offset); });
  } else {
    liveState();
  }
  if (Q.has("autostart")) begin();
  if (Q.has("open")) { $("mods").classList.add("open"); $("unscripted").classList.add("open"); }
  requestAnimationFrame(loop);
})();
