(() => {
  const Q = new URLSearchParams(location.search), MOCK = Q.has("mock"), REPLAY = Q.get("replay"), SIM = MOCK || !!REPLAY;
  const SPEED = +(Q.get("speed") || 1), RM = matchMedia("(prefers-reduced-motion: reduce)").matches, MAC = /Mac|iP/.test(navigator.platform);
  const $ = (id) => document.getElementById(id), now = () => performance.now() / 1000, B = document.body;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const set = (el, v) => { v = String(v ?? ""); if (el.textContent !== v) el.textContent = v; };
  const store = {
    get(k, d) { try { const v = localStorage.getItem("hud." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
    set(k, v) { try { localStorage.setItem("hud." + k, JSON.stringify(v)); } catch {} },
  };
  const ICONS = {
    flag: "M6 21V4h11l-2.5 4L17 12H6", wave: "M3 12h4l2.5-6 5 12 2.5-6h4", gauge: "M4 17a8 8 0 1 1 16 0M12 17l4-5",
    grid: "M3 5h18v14H3zM3 12h18M9 5v14M15 5v14", eye: "M2.5 12S6 5.5 12 5.5 21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12ZM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6",
    spark: "M12 3c.8 5 4 8.2 9 9-5 .8-8.2 4-9 9-.8-5-4-8.2-9-9 5-.8 8.2-4 9-9Z", list: "M9 6h11M9 12h11M9 18h11M4.5 6h0M4.5 12h0M4.5 18h0",
    wind: "M3 8h10a3 3 0 1 0-3-3M3 12h15a3 3 0 1 1-3 3M3 16h7", cloud: "M7 18a4 4 0 0 1-.4-8A6 6 0 0 1 18 9.5 4.3 4.3 0 0 1 17.5 18H7Z",
    target: "M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16M12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6M12 2v3M12 19v3M2 12h3M19 12h3",
    cam: "M4 8h3l2-3h6l2 3h3v11H4ZM12 9.5a3.5 3.5 0 1 0 0 7 3.5 3.5 0 0 0 0-7", bolt: "M13 3 5 14h6l-1 7 8-11h-6l1-7Z", play: "M8 5v14l11-7Z",
    robot: "M5 8h14v10H5zM12 4v4M9.5 13h0M14.5 13h0", tag: "M3.5 12V4.5H11L20 13.5 13.5 20ZM7.5 8.5h0", x: "M6 6l12 12M18 6 6 18",
    keys: "M3 6h18v12H3zM7 10h0M11 10h0M15 10h0M8 14h8", info: "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18M12 11v5M12 8h0",
    film: "M3 5h18v14H3zM3 9h18M3 15h18", back: "M4 12a8 8 0 1 0 2.4-5.7M4 4.5v4h4", pin: "M9 3h6l-1 6 4 4H6l4-4ZM12 13v8",
    warn: "M12 3 2 20h20ZM12 10v4M12 17h0", hand: "M8 13V5.5a1.5 1.5 0 0 1 3 0V11M11 10V4.5a1.5 1.5 0 0 1 3 0V11M14 10.5V6a1.5 1.5 0 0 1 3 0v7a7 7 0 0 1-7 7c-3 0-4.5-2-6-5l-1.4-2.6a1.5 1.5 0 0 1 2.6-1.4L8 13",
    check: "M5 12.5 10 17 19 7", layers: "M12 3l9 5-9 5-9-5zM3 13l9 5 9-5", map: "M9 4 3 6v14l6-2 6 2 6-2V4l-6 2zM9 4v14M15 6v14",
    chart: "M4 20V10M10 20V4M16 20v-7M22 20H2", inbox: "M3 13h5l1.5 3h5L16 13h5M5 5h14l2 8v6H3v-6Z", link: "M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1M14 10a4 4 0 0 0-5.7 0l-3 3a4 4 0 0 0 5.7 5.7l1-1",
    sun: "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5",
    block: "M4 18h16M6 18l3-9h6l3 9M8 13h8", person: "M12 3a3 3 0 1 0 0 6 3 3 0 0 0 0-6M6 21v-5a6 6 0 0 1 12 0v5", dot: "M12 11a1 1 0 1 0 0 2 1 1 0 0 0 0-2",
    battery: "M3 8h15v8H3zM21 11v2M6 11v2",
  };
  const icon = (k) => `<svg viewBox="0 0 24 24"><path d="${ICONS[k] || ICONS.dot}"/></svg>`;
  const NAME = { BACK_OFF: "Back off", PAUSE_PERSON: "Paused", ESCALATE_TO_HUMAN: "Escalated", SECOND_VIEW: "Second view", VERIFY_PAINT: "Verify paint", TRANSIT: "Walking", TO_DOCK: "To dock", GAS_CHECK: "Gas check", LEAK_APPROACH: "To leak", LEAK_INSPECT: "Inspect flange", LEAK_VERIFY: "Verify leak", RETORQUE: "Re-torque" };
  const nice = (s) => NAME[s] || (s ? s[0] + s.slice(1).toLowerCase().replace(/_/g, " ") : "");
  const TONE = { BACK_OFF: "warn", HOLD: "warn", WAIT: "warn", PAUSE_PERSON: "stop", PAUSED: "stop", ESCALATE_TO_HUMAN: "stop", GAS_RETREAT: "stop", DONE: "ok" };
  const ALARM = {
    BACK_OFF: ["warn", "Back off", "Contact force over the 40 N limit · retracting", "warn"],
    PAUSE_PERSON: ["stop", "Person in work zone", "Worker within 3 m · arm stopped", "hand"],
    PAUSED: ["stop", "Person in work zone", "A robot paused within 3 m of a person", "hand"],
    HOLD: ["warn", "Holding", "Inspection camera obscured · waiting for a clear view", "eye"],
    ESCALATE_TO_HUMAN: ["stop", "Escalated", "An operator is needed in the control room", "warn"],
    DONE: ["ok", "Repair complete", "Steel verified clean and sealed", "check"],
  };
  const FORCE = new Set(["SCRUB", "BACK_OFF"]), RANK = { stop: 3, warn: 2, act: 1, ok: 0.5 };
  const GAS = 0.1, DWELL = 1.6, WIN = 34;
  const S = { started: false, t: 0, state: null, caption: null, robots: [], camera: null, log: [], logN: 0, force: 0, peak: 0, spike: 0, rust: null, conf: null, soak: null, source: "EDGE", gas: null, gasMax: 0, gasAlarm: false, gemAt: -99, shot: 1, focus: null, labels: store.get("labels", true), series: {}, sim: SIM, mock: MOCK, replay: REPLAY, speed: SPEED };
  function push(k, v) {
    if (v == null || Number.isNaN(+v)) return;
    const s = (S.series[k] ||= []), t = S.at || now();
    s.push([t, +v]);
    while (s[0][0] < t - WIN) s.shift();
  }
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
  function q(el, html) {
    if (el._q && el._h === html) return el._q;
    el.innerHTML = html; el._h = html; el._q = {};
    el.querySelectorAll("[data-k]").forEach((e) => (el._q[e.dataset.k] = e));
    return el._q;
  }
  function fit(c) {
    const dpr = Math.min(devicePixelRatio || 1, 2), w = c.clientWidth, h = c.clientHeight;
    if (!w) return [];
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    return [g, w, h];
  }
  function spark(c, data = [], o) {
    const [g, w, h] = fit(c);
    if (!g) return;
    const win = o.win || 30, t0 = now() - win, x = (t) => ((t - t0) / win) * w;
    const y = (v) => h - 2 - ((Math.min(o.max, v) - o.min) / (o.max - o.min)) * (h - 4);
    (o.refs || []).forEach(([v, col]) => { g.strokeStyle = col; g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, Math.round(y(v)) + 0.5); g.lineTo(w, Math.round(y(v)) + 0.5); g.stroke(); });
    g.setLineDash([]);
    const pts = data.filter((p) => p[0] >= t0 - 1);
    if (pts.length < 2) return;
    g.lineWidth = o.w || 1.4; g.lineJoin = g.lineCap = "round";
    for (let k = 1; k < pts.length; k++) {
      if (k === 1 || o.ramp) { if (k > 1) g.stroke(); g.strokeStyle = o.ramp ? o.ramp(Math.max(pts[k - 1][1], pts[k][1])) : o.color; g.beginPath(); g.moveTo(x(pts[k - 1][0]), y(pts[k - 1][1])); }
      g.lineTo(x(pts[k][0]), y(pts[k][1]));
    }
    g.stroke();
    const l = pts.at(-1);
    g.fillStyle = o.ramp ? o.ramp(l[1]) : o.color;
    g.beginPath(); g.arc(Math.min(w - 2.5, x(l[0])), y(l[1]), 2.2, 0, 7); g.fill();
  }
  const P = new Map(), C = new Map(), V = new Map(), pins = new Set(store.get("pins", [])), hooks = [];
  let booted = false, auto = store.get("auto", true), presenter = Q.has("presenter"), slot = null, slotAt = -99, dismissed = null;
  const command = (c) => (C.set(c.id, { group: "Demo", icon: "bolt", ...c }), c);
  const uncommand = (id) => C.delete(id);
  function panel(p) {
    p = { edge: "right", priority: 0, icon: "grid", ...p };
    P.set(p.id, p);
    command({ id: p.id, group: "Panels", label: p.title, icon: p.icon, key: p.key, keys: p.keys, desc: p.desc, hint: p.hint, panel: p, active: () => pins.has(p.id), run: (from) => togglePin(p.id, from) });
    if (booted && pins.has(p.id)) show(p.id);
    return p;
  }
  function view(v) {
    V.set(v.id, v);
    command({ id: "view." + v.id, group: "Views", label: v.title, icon: v.icon, key: v.key, keys: v.keys, desc: v.desc, hint: v.hint, preview: v.preview, active: () => sheet.id === v.id, run: () => openView(sheet.id === v.id ? null : v.id) });
    return v;
  }
  const DOCK = { left: $("dock-left"), right: $("dock-right"), bottom: $("dock-bottom") }, open = new Map(), views = new Set();
  const mount = (fn, el, prev) => { const v = { fn, el, prev }; views.add(v); fn(el, S); return v; };
  function show(id, from) {
    const p = P.get(id);
    if (!p || open.has(id)) return;
    const dock = DOCK[p.edge] || DOCK.right, el = document.createElement("section");
    el.className = "card flying";
    el.innerHTML = `<header>${icon(p.icon)}<span>${esc(p.title)}</span><em>${esc(p.meta || "")}</em><i class="pin">${icon("pin")}</i><button class="x" aria-label="Close ${esc(p.title)}">${icon("x")}</button></header><div class="cb"></div>`;
    el.querySelector(".x").onclick = () => (pins.has(id) ? togglePin(id) : ((dismissed = id), hide(id)));
    dock.appendChild(el);
    open.set(id, { el, v: mount(p.render, el.lastChild) });
    el.classList.toggle("pinned", pins.has(id));
    for (const [k, o] of open) if (k !== id && o.el.parentNode === dock && dock.scrollHeight > dock.clientHeight + 1) { pins.delete(k); hide(k, true); }
    if (RM) { el.classList.remove("flying"); return; }
    const r = el.getBoundingClientRect();
    el.style.transition = "none";
    el.style.transform = from ? `translate(${from.left + from.width / 2 - r.left - r.width / 2}px,${from.top + from.height / 2 - r.top - r.height / 2}px) scale(${Math.min(1, from.width / r.width).toFixed(3)})`
      : { left: "translateX(-16px)", right: "translateX(16px)", bottom: "translateY(16px)" }[p.edge];
    el.style.opacity = from ? "0.7" : "0";
    el.offsetWidth;
    el.style.transition = el.style.transform = el.style.opacity = "";
    setTimeout(() => el.classList.remove("flying"), 380);
  }
  function hide(id, instant) {
    const o = open.get(id);
    if (!o) return;
    open.delete(id); views.delete(o.v);
    if (instant || RM) { o.el.remove(); return; }
    o.el.style.height = o.el.offsetHeight + "px";
    o.el.classList.add("leaving");
    o.el.offsetWidth;
    o.el.style.height = "0px";
    setTimeout(() => o.el.remove(), 280);
  }
  function togglePin(id, from) {
    if (pins.has(id)) { pins.delete(id); if (slot !== id) hide(id); } else { pins.add(id); dismissed = null; show(id, from); }
    open.get(id)?.el.classList.toggle("pinned", pins.has(id));
    store.set("pins", [...pins]);
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
  const prio = (p) => (typeof p.priority === "function" ? p.priority(S) : p.priority);
  const live = (p) => (typeof p.phases === "function" ? !!p.phases(S) : !!p.phases && p.phases.includes(S.state));
  function surface() {
    let want = null;
    if (dismissed && !live(P.get(dismissed))) dismissed = null;
    if (auto && !presenter && !S.guide && S.started && !sheet.id) P.forEach((p) => { if (!pins.has(p.id) && p.id !== dismissed && live(p) && (!want || prio(p) > prio(want))) want = p; });
    const id = want ? want.id : null;
    if (id === slot || now() - slotAt < DWELL / SPEED) return;
    if (slot && !pins.has(slot)) hide(slot);
    slot = id; slotAt = now();
    if (id) show(id);
  }
  const sheet = { id: null, el: $("sheet"), v: null };
  function openView(id) {
    if (sheet.v) views.delete(sheet.v);
    sheet.id = id; sheet.v = null;
    B.classList.toggle("sheet", !!id);
    if (!id) return;
    const v = V.get(id);
    sheet.el.innerHTML = `<header>${[...V.values()].map((x) => `<button data-v="${x.id}" class="${x.id === id ? "on" : ""}">${icon(x.icon)}${esc(x.title)}</button>`).join("")}<button class="x" aria-label="Close view">${icon("x")}</button></header><div class="sb"></div>`;
    sheet.el.querySelector("header").onclick = (e) => { const b = e.target.closest("button"); if (b) openView(b.dataset.v || null); };
    sheet.v = mount(v.render, sheet.el.lastChild, true);
  }
  const pal = $("palette"), inp = $("p-input"), list = $("p-list"), pv = $("p-preview");
  let palOpen = false, items = [], sel = 0, pvCmd, pvView = null, recent = store.get("recent", []);
  const GROUPS = ["Suggested now", "Top hit", "Recent", "Control centre", "Views", "Panels", "Unscripted change", "Robots", "Cameras", "Demo", "Help"];
  const score = (c) => (c.panel ? (pins.has(c.id) ? 0 : slot === c.id ? 0.9 : live(c.panel) ? prio(c.panel) / 100 : c.panel.suggest ? c.panel.suggest(S) : 0) : c.score ? c.score(S) : 0);
  const val = (h) => (typeof h === "function" ? h(S) : h) ?? "";
  const kbd = (k) => (k ? (k.length === 1 ? k.toUpperCase() : k) : "");
  let asking = null;
  const PH = inp.placeholder;
  function openPal(text = "") { palOpen = true; pal.classList.add("open"); inp.value = text; filter(); inp.focus(); }
  function closePal() { palOpen = false; asking = null; inp.placeholder = PH; pal.classList.remove("open"); inp.blur(); setPreview(null); }
  function ask(label, placeholder, cb) { openPal(); asking = { label, cb }; inp.placeholder = placeholder; filter(); }
  function fuzzy(qq, s) {
    s = s.toLowerCase();
    let i = 0, sc = 0, prev = -2;
    const idx = [];
    for (let j = 0; j < s.length && i < qq.length; j++) if (s[j] === qq[i]) { sc += 1 + (j === prev + 1 ? 2 : 0) + (j === 0 || s[j - 1] === " " ? 3 : 0); prev = j; idx.push(j); i++; }
    return i === qq.length ? { sc: sc - s.length * 0.02, idx } : null;
  }
  const match = (qq, c) => fuzzy(qq, c.label) || (((c.keys || "") + " " + c.group).toLowerCase().split(/\s+/).some((w) => w.startsWith(qq)) && { sc: 0.5, idx: [] });
  function filter() {
    const qq = inp.value.trim().toLowerCase(), all = [...C.values()];
    if (asking) {
      const text = inp.value.trim(), a = asking;
      items = [{ c: { id: "ask", group: a.label, icon: "spark", label: text ? `Send “${text}”` : "Type, then press return", desc: "Gemini turns this into jobs through the coordinator API. It cannot command motors or approve its own SUPV jobs.", run: () => text && a.cb(text) }, g: a.label }];
      sel = 0; renderList(); return;
    }
    if (!qq) {
      const sug = all.map((c) => [c, score(c)]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1]).slice(0, 4).map((x) => x[0]);
      const rec = recent.map((id) => C.get(id)).filter((c) => c && !sug.includes(c)).slice(0, 3);
      items = [...sug.map((c) => ({ c, g: "Suggested now" })), ...rec.map((c) => ({ c, g: "Recent" })),
        ...all.filter((c) => !sug.includes(c) && !rec.includes(c)).map((c) => ({ c, g: c.group })).sort((a, b) => GROUPS.indexOf(a.g) - GROUPS.indexOf(b.g))];
    } else {
      const rows = all.map((c) => ({ c, m: match(qq, c) })).filter((r) => r.m).map((r) => ({ ...r, s: r.m.sc + score(r.c) * 4, g: r.c.group })).sort((a, b) => b.s - a.s);
      if (rows[0]) rows[0].g = score(rows[0].c) > 0 ? "Suggested now" : "Top hit";
      items = rows.sort((a, b) => GROUPS.indexOf(a.g) - GROUPS.indexOf(b.g) || b.s - a.s);
    }
    sel = 0;
    renderList();
  }
  const hl = (t, m) => (m?.idx.length ? [...t].map((ch, i) => (m.idx.includes(i) ? `<mark>${esc(ch)}</mark>` : esc(ch))).join("") : esc(t));
  function renderList() {
    let g0 = "";
    list.innerHTML = items.map((r, i) => (r.g !== g0 ? `<li class="gh">${esc((g0 = r.g))}</li>` : "") +
      `<li role="option" data-i="${i}"><span class="ic">${icon(r.c.icon)}</span><span class="lb">${hl(r.c.label, r.m)}</span><span class="pk"></span><i class="st"></i>${r.c.key ? `<kbd>${kbd(r.c.key)}</kbd>` : ""}</li>`).join("") || `<li class="empty">No results</li>`;
    pvCmd = undefined;
    peeks();
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
    if (!c) { pv.innerHTML = palOpen && !items.length ? `<p class="pv-none">Try “force”, “gas”, “map” or “storm”</p>` : ""; return; }
    const fn = c.panel ? c.panel.preview || c.panel.render : c.preview;
    pv.innerHTML = `<div class="pv-head"><span class="ic">${icon(c.icon)}</span><div><b>${esc(c.label)}</b><span>${esc(c.group)}</span></div></div>${fn ? `<div class="pv-live"></div>` : ""}${c.desc ? `<p class="pv-d">${esc(c.desc)}</p>` : ""}`;
    if (fn) pvView = mount(fn, pv.querySelector(".pv-live"), true);
  }
  function run(c, keep) {
    if (!c?.run) return;
    const from = palOpen ? pv.querySelector(".pv-live")?.getBoundingClientRect() || pv.getBoundingClientRect() : null;
    recent = [c.id, ...recent.filter((x) => x !== c.id)].slice(0, 6);
    store.set("recent", recent);
    if (!keep) closePal();
    c.run(from);
    if (keep) { const s = sel; filter(); select(s, true); }
  }
  function peeks() {
    list.querySelectorAll("li[data-i]").forEach((li) => {
      const c = items[+li.dataset.i]?.c;
      if (!c) return;
      set(li.children[2], c.panel && slot === c.id && !pins.has(c.id) ? "Showing now" : val(c.hint));
      li.children[2].classList.toggle("now", !!(c.panel && slot === c.id && !pins.has(c.id)));
      li.classList.toggle("on", !!c.active?.());
    });
  }
  inp.addEventListener("input", filter);
  list.addEventListener("pointermove", (e) => { const li = e.target.closest("li[data-i]"); if (li && (e.movementX || e.movementY)) select(+li.dataset.i); });
  list.addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) run(items[+li.dataset.i].c, e.metaKey || e.ctrlKey); });
  addEventListener("pointerdown", (e) => { if (palOpen && !pal.contains(e.target) && !e.target.closest("#island")) closePal(); });
  const isl = $("island");
  let banT = 0;
  function fitIsland() { const l = isl.querySelector(isl.classList.contains("big") ? ".ban" : ".cmp"); isl.style.width = l.offsetWidth + "px"; isl.style.height = l.offsetHeight + "px"; }
  const ro = new ResizeObserver(fitIsland);
  isl.querySelectorAll(":scope > div").forEach((d) => ro.observe(d));
  function banner(tone, title, sub, ic = "warn", ms = 4000) {
    if (S.seeking) return;
    set($("b-title"), title); set($("b-sub"), sub); $("b-icon").innerHTML = icon(ic);
    isl.dataset.tone = tone; isl.classList.add("big"); B.classList.add("alarm"); fitIsland();
    clearTimeout(banT);
    banT = setTimeout(() => { isl.classList.remove("big"); B.classList.remove("alarm"); fitIsland(); }, RM ? ms + 1000 : ms);
  }
  function island() {
    if (!isl.classList.contains("big")) { const tone = S.gasAlarm ? "stop" : TONE[S.state] || (S.linkCut ? "warn" : S.started ? "run" : ""); if (isl.dataset.tone !== tone) isl.dataset.tone = tone; }
    set($("i-state"), S.state ? nice(S.state) : S.started ? "Live" : "Ready");
    num($("i-time"), S.t, 1);
    const k = S.gasAlarm && S.gasMax ? [S.gasMax * 100, 0, "% LEL"] : S.soak ? [S.soak.left, 0, "s left"] : FORCE.has(S.state) ? [S.force, 1, "N"] : S.rust != null ? [S.rust, 1, "% rust"] : null;
    isl.classList.toggle("nokey", !k);
    if (k) { if ($("i-unit").textContent !== k[2]) { tweens.delete($("i-key")); $("i-key")._v = null; } num($("i-key"), k[0], k[1]); set($("i-unit"), k[2]); }
    set($("i-edge"), S.linkCut || S.control?.link === "CUT" ? "edge only" : "");
  }
  const sub = $("subtitle");
  let subKey = null, layer = 0;
  function caption() {
    if (!S.started) return { lead: "Offshore rig autonomy", rest: " · press Space to begin", tone: "act" };
    if (S.robots.length > 1) {
      const rank = (r) => (RANK[r.caption?.tone] || 0) + (S.gasAlarm && /GAS|LEAK|RETORQUE/.test(r.state) ? 2.5 : 0) - (/DOCK|CHARG/.test(r.state) ? 1.5 : 0);
      const r = S.robots.reduce((a, b) => (rank(b) > rank(a) ? b : a));
      return r.caption && { ...r.caption, lead: `${r.name.split(" ·")[0]} · ${r.caption.lead}` };
    }
    return S.caption || S.robots[0]?.caption;
  }
  function subtitle() {
    const c = caption(), html = c ? `<b>${esc(c.lead)}</b>${esc(c.rest)}` : "", key = (c?.tone || "") + html.replace(/\d+(\.\d+)?/g, "#");
    const cur = sub.children[layer];
    if (key === subKey) { if (cur.innerHTML !== html) cur.innerHTML = html; return; }
    subKey = key; layer ^= 1;
    const on = sub.children[layer];
    on.className = "sub on " + (c?.tone || ""); on.innerHTML = html;
    cur.classList.remove("on");
  }
  function project(cam, p) {
    const T = cam.T, K = cam.K, d = [p[0] - T[0][3], p[1] - T[1][3], p[2] + 0.85 - T[2][3]];
    const c = [0, 1, 2].map((j) => d[0] * T[0][j] + d[1] * T[1][j] + d[2] * T[2][j]);
    return [K[0][0] * c[0] / c[2] + K[0][2], K[1][1] * c[1] / c[2] + K[1][2], c[2]];
  }
  const LBL = $("labels");
  function labels() {
    const on = S.labels && !presenter && !sheet.id && S.robots.length > 1 && S.camera;
    LBL.classList.toggle("on", !!on);
    if (!on) return;
    const cam = S.camera, el0 = SIM ? S.shotEl : $("view"), r = el0 ? el0.getBoundingClientRect() : { left: 0, top: 0, width: innerWidth, height: innerHeight };
    const k = Math.max(r.width / cam.w, r.height / cam.h), ox = r.left + (r.width - cam.w * k) / 2, oy = r.top + (r.height - cam.h * k) / 2, M = 30;
    const keep = new Set(), lo = DOCK.left._p ? DOCK.left.getBoundingClientRect().right + 8 : 8, hi = DOCK.right._p ? DOCK.right.getBoundingClientRect().left - 8 : innerWidth - 8;
    S.robots.forEach((rb) => {
      if (!rb.world_pos) return;
      keep.add(rb.id);
      let el = LBL.querySelector(`[data-id="${rb.id}"]`);
      if (!el) { el = document.createElement("div"); el.dataset.id = rb.id; el.innerHTML = `<div class="lc"><b><i></i><span></span></b><small></small></div>`; LBL.appendChild(el); }
      const [u, v, z] = project(cam, rb.world_pos);
      let x = ox + u * k, y = oy + v * k, mode = z <= 0.1 ? "behind" : rb.occluded ? "occ" : "";
      if (!mode && (x < M || x > innerWidth - M || y < M + 60 || y > innerHeight - M)) { x = Math.min(innerWidth - M, Math.max(M, x)); y = Math.min(innerHeight - M, Math.max(M + 60, y)); mode = "edge"; }
      el.className = `lbl ${rb.caption?.tone || ""} ${mode} ${S.ping === rb.id ? "ping" : ""} ${S.focus === rb.id ? "focus" : ""}`;
      el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
      set(el.querySelector("span"), rb.name); set(el.querySelector("small"), rb.caption?.text);
      const cw = el.firstChild.offsetWidth, dx = Math.min(Math.max(x - cw / 2, lo), hi - cw) - (x - cw / 2);
      el.firstChild.style.marginLeft = dx.toFixed(0) + "px";
    });
    [...LBL.children].forEach((el) => keep.has(el.dataset.id) || el.remove());
  }
  function addLog(e) { S.log.unshift(e); if (S.log.length > 40) S.log.length = 40; S.logN++; }
  function setState(st) {
    S.state = st;
    if (ALARM[st]) banner(...ALARM[st]);
    if (st === "DONE" && MOCK) setTimeout(() => S.state === "DONE" && shot(1), 3500);
  }
  function gasIn(g) {
    const m = g.max ?? Math.max(0, ...(g.sensors || []).map((s) => s.lel || 0)), al = g.alarm ?? m >= GAS;
    if (al && !S.gasAlarm) {
      S.gasAt = now();
      const top = (g.sensors || []).reduce((a, b) => ((b.lel || 0) > (a?.lel || 0) ? b : a), null);
      banner("stop", m ? `Gas · ${Math.round(m * 100)} % LEL` : "Gas detected", [top?.at || g.source, "work nearby paused"].filter(Boolean).join(" · "), "wind", 4200);
    }
    S.gas = g; S.gasMax = m || 0; S.gasAlarm = !!al;
    if (m) push("gas", m * 100);
  }
  const KEEP = ["rust_grid", "sprayed", "painted", "target", "brush", "inspection", "pip", "control", "fleet", "source", "orchestrator", "fpv", "weather", "hud"];
  function apply(h) {
    KEEP.forEach((k) => { if (h[k] != null) S[k] = h[k]; });
    if (h.fleet) S.fleetLive = true;
    if (h.robots) { const ids = h.robots.map((r) => r.id).join(); if (ids !== S.ids) { S.ids = ids; robotCommands(h.robots); } S.robots = h.robots; }
    if (h.camera) S.camera = h.camera;
    if ("caption" in h) S.caption = h.caption;
    if (h.t != null) S.t = h.t;
    if (SIM && h.shot && h.shot !== S.shot) shot(h.shot);
    if (h.state && h.state !== S.state) setState(h.state);
    (h.events || []).forEach(addLog);
    if (h.force_n != null) { S.force = Math.max(h.force_n, S.spike); S.peak = Math.max(S.peak, S.force); push("force", S.force); }
    if (h.rust_fraction != null) push("rust", (S.rust = h.rust_fraction * 100));
    if (h.detector_conf != null) push("conf", (S.conf = h.detector_conf));
    if ("soak" in h) S.soak = h.soak;
    if (h.gemini) { const k = JSON.stringify(h.gemini); if (k !== S.gkey) { S.gkey = k; S.gemini = h.gemini; S.gemAt = S.at || now(); } }
    if (h.gas) gasIn(h.gas);
    if (h.sensors) Object.entries(h.sensors).forEach(([id, v]) => Object.entries(v).forEach(([m, x]) => push(id + "." + m, x)));
    hooks.forEach((f) => f(h, S));
  }
  function robotCommands(rs) {
    [...C.keys()].filter((id) => id.startsWith("robot.")).forEach(uncommand);
    rs.forEach((r) => command({ id: "robot." + r.id, group: "Robots", label: r.name, icon: "robot", keys: "unit robot focus " + r.id, desc: "Focus this robot: its label pings, and First person and Sensors follow it.",
      hint: () => S.robots.find((x) => x.id === r.id)?.caption?.lead, active: () => S.focus === r.id, run: () => focus(r.id) }));
  }
  function focus(id) { S.focus = id; S.ping = id; clearTimeout(S.pingT); S.pingT = setTimeout(() => (S.ping = null), 1200); }
  const post = (name, args = {}) => fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, args }) }).then((r) => r.json()).catch(() => null);
  function event(name, args, local) {
    if (SIM) { if (local) local(); return; }
    post(name, args);
    addLog({ t: S.t, from: "Operator", to: name, reason: "sent to the sim " + JSON.stringify(args || {}) });
  }
  const judge = (to, reason, state) => { addLog({ t: S.t, from: "Judge", to, reason }); if (state) setState(state); };
  let side = 0;
  function shot(n) {
    S.shot = n;
    if (!SIM) { post("camera", { shot: n }); return; }
    const a = $(side ? "shotA" : "shotB"), b = $(side ? "shotB" : "shotA");
    side ^= 1; S.shotEl = a;
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
  let session = null, playT0 = null, cursor = 0, offset = +(Q.get("t") || 0);
  let pausedAt = null;
  const clock = () => ((pausedAt ?? now()) - playT0) * SPEED + offset;
  function pause(on = pausedAt == null) {
    if (on && pausedAt == null) pausedAt = now();
    else if (!on && pausedAt != null) { if (playT0 != null) playT0 += now() - pausedAt; pausedAt = null; }
    return pausedAt != null;
  }
  function jump(pt) {
    if (!session) return false;
    begin();
    const el = clock();
    if (pt < el - 0.05) return false;
    seek(pt); offset += pt - el;
    return true;
  }
  function layout(ids) {
    [...open.keys()].forEach((id) => { pins.delete(id); hide(id, true); });
    slot = null; dismissed = null;
    ids.forEach((id) => P.has(id) && togglePin(id));
    store.set("pins", [...pins]);
  }
  function playMock() {
    if (!session || playT0 == null) return;
    const el = clock(), snaps = session.snapshots;
    while (cursor < snaps.length && snaps[cursor].pt <= el) apply(snaps[cursor++]);
  }
  function seek(pt) {
    const snaps = session.snapshots, back = now();
    S.seeking = true;
    while (cursor < snaps.length && snaps[cursor].pt <= pt) { const s = snaps[cursor++]; S.at = back - (pt - s.pt) / SPEED; apply(s); }
    S.at = 0; S.seeking = false;
  }
  function begin() {
    if (S.started) return;
    S.started = true;
    shot(REPLAY ? 1 : 2);
    if (SIM) { playT0 = now(); if (offset && session) seek(offset); }
    if (REPLAY && !Q.has("view")) openView("map");
  }
  function liveState() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/state");
    ws.onmessage = (ev) => { try { apply(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(liveState, 1500);
  }
  const json = (u) => fetch(u).then((r) => (r.ok ? r.json() : null)).catch(() => null);
  function loadFleet() { if (!S.fleetLive) json("/api/fleet").then((f) => { if (f && !S.fleetLive) S.fleet = f; }); }
  command({ id: "start", group: "Demo", label: "Start demo", icon: "play", key: "Space", keys: "begin play run", desc: "Starts the mission from the patrol.", score: () => (S.started ? 0 : 1), run: begin });
  command({ id: "presenter", group: "Demo", label: "Presenter mode", icon: "film", key: "p", keys: "clean video record capture hide", desc: "Hides everything but the phase subtitle, for a clean take. ⌘K still works.", active: () => presenter, score: () => (presenter ? 1 : 0), run: () => { presenter = !presenter; B.classList.toggle("presenter", presenter); } });
  command({ id: "context", group: "Demo", label: "Context panels", icon: "layers", keys: "auto surface phase", desc: "Lets one panel surface by itself for the current phase. Pinned panels always stay.", active: () => auto, run: () => { auto = !auto; store.set("auto", auto); } });
  command({ id: "closeall", group: "Demo", label: "Close all panels", icon: "x", key: "x", keys: "clear hide reset unpin", desc: "Unpins and closes every docked card.", run: () => { [...open.keys()].forEach((id) => { pins.delete(id); hide(id); }); store.set("pins", []); dismissed = slot; } });
  command({ id: "labels", group: "Robots", label: "Robot labels", icon: "tag", key: "r", keys: "names tags overlay", desc: "Floating name and phase over each robot when several are in shot.", active: () => S.labels, run: () => store.set("labels", (S.labels = !S.labels)) });
  ["Wide rig|Aerial orbit around the whole platform.", "Hero deck|The repair deck at eye level.", "Close-up|Tight on the brush and the patch."].forEach((s, n) => {
    const [label, desc] = s.split("|");
    command({ id: "cam" + (n + 1), group: "Cameras", label, icon: "cam", key: String(n + 1), keys: "camera shot view", desc, active: () => S.shot === n + 1, run: () => shot(n + 1),
      preview: (el) => q(el, `<img class="pv-img" src="/api/mock/shot/${n + 1}" alt="">`) });
  });
  if (SIM) command({ id: "restart", group: "Demo", label: "Restart replay", icon: "back", keys: "reset reload again", desc: "Plays the recording again from the start.", run: () => location.reload() });
  command({ id: "keys", group: "Help", label: "Keyboard shortcuts", icon: "keys", key: "?", keys: "help controls", desc: "Every command has a row here; most have a key.",
    preview: (el) => q(el, `<ul class="pv-keys">${[[MAC ? "⌘K" : "Ctrl K", "Search"], ["/", "Search"], ["esc", "Close"], [MAC ? "⌘↵" : "Ctrl ↵", "Run, keep open"], ...[...C.values()].filter((c) => c.key && c.key !== "?").map((c) => [kbd(c.key), c.label])].map(([k, t]) => `<li><kbd>${esc(k)}</kbd>${esc(t)}</li>`).join("")}</ul>`) });
  command({ id: "about", group: "Help", label: "About this demo", icon: "info", keys: "help info", desc: "Quadrupeds find corrosion and gas leaks on an offshore rig, fix what they safely can and call a human for the rest. Perception runs on the robot; Gemini gives second opinions, plans and explains." });
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if ((e.metaKey || e.ctrlKey) && k === "k") { e.preventDefault(); palOpen ? closePal() : openPal(); return; }
    if (palOpen) {
      if (k === "escape") { e.preventDefault(); closePal(); }
      else if (k === "arrowdown" || k === "arrowup") { e.preventDefault(); select(sel + (k === "arrowdown" ? 1 : -1)); }
      else if (k === "enter" && items[sel]) { e.preventDefault(); run(items[sel].c, e.metaKey || e.ctrlKey); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA/.test(e.target.tagName)) return;
    if (k === "/") { e.preventDefault(); openPal(); return; }
    if (e.key === "?") { e.preventDefault(); openPal("help"); return; }
    if (k === "escape") { openView(null); return; }
    if (e.code === "Space") { e.preventDefault(); begin(); return; }
    const id = e.shiftKey ? (/^Digit\d$/.test(e.code) ? "⇧" + e.code[5] : /^Key/.test(e.code) ? "⇧" + e.code[3] : null) : k;
    const c = id && [...C.values()].find((x) => x.key === id);
    if (c) { e.preventDefault(); c.run(); }
  });
  let frames = 0, fpsT = now();
  function loop() {
    Object.values(DOCK).forEach(clip);
    playMock();
    surface();
    stepTweens();
    views.forEach((v) => (v.prev || !presenter) && v.fn(v.el, S));
    island(); subtitle(); labels();
    if (palOpen) peeks();
    frames++;
    if (SIM && now() - fpsT > 0.5) { set($("fps"), Math.round(frames / (now() - fpsT))); frames = 0; fpsT = now(); }
    requestAnimationFrame(loop);
  }
  isl.onclick = () => (palOpen ? closePal() : openPal());
  $("p-keep").textContent = MAC ? "⌘↵" : "Ctrl ↵";
  B.classList.toggle("presenter", presenter);
  window.HUD = { S, commands: () => [...C.values()], clock: () => (playT0 == null ? offset : clock()), speed: SPEED, session: () => session, begin, pause, paused: () => pausedAt != null, jump, layout, replayed: () => !!session, ask, panel, command, uncommand, view, on: (f) => hooks.push(f), apply, event, post, judge, banner, addLog, setState, focus, shot, openView, num, set, esc, q, fit, spark, push, icon, nice, now, json };
  addEventListener("DOMContentLoaded", () => {
    booted = true;
    pins.forEach((id) => show(id));
    if (SIM) {
      set($("status"), REPLAY ? "Replay " + REPLAY.split("/").pop() : "Mock replay");
      shot(1);
      json(REPLAY ? "/api/replay?path=" + encodeURIComponent(REPLAY) + (Q.get("video") ? "&video=" + encodeURIComponent(Q.get("video")) : "") : "/api/mock").then((s) => {
        if (!s) return set($("status"), "Replay not found");
        s.snapshots = s.snapshots.concat(s.fleet || []); session = s;
        if (playT0 != null && offset) seek(offset);
      });
    } else liveState();
    loadFleet();
    if (!SIM) setInterval(loadFleet, 4000);
    json("/api/fleet/map").then((m) => (S.fleetMap = m));
    if (Q.has("autostart")) begin();
    if (Q.get("view")) openView(Q.get("view"));
    (Q.get("pin") || "").split(",").filter((id) => P.has(id) && !pins.has(id)).forEach((id) => togglePin(id));
    requestAnimationFrame(loop);
  });
})();
