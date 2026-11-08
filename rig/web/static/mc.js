(() => {
  const Q = new URLSearchParams(location.search), MOCK = Q.has("mock") ? Q.get("mock") || "1" : null, FULL = MOCK === "1";
  const MAC = /Mac|iP/.test(navigator.platform), RM = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const $ = (id) => document.getElementById(id), B = document.body, now = () => performance.now() / 1000;
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const put = (el, html) => { if (el._h !== html) { el._h = html; el.innerHTML = html; } };
  const txt = (el, v) => { v = String(v ?? ""); if (el.textContent !== v) el.textContent = v; };
  const nice = (s) => (s ? String(s).replace(/_/g, " ").toLowerCase() : "");
  const fmt = (v) => (Math.abs(v) >= 100 || Number.isInteger(+v) ? String(Math.round(v)) : (+v).toFixed(1));
  const CMD = MAC ? "⌘K" : "Ctrl K";

  const SCEN = [["gas", "Gas leak", "Leak search, valve isolation, hot-work stop", "⇧1"], ["patrol", "Deck patrol", "Patrol route, re-routes, can pick-up", "⇧2"],
    ["repair", "Pipe repair", "Rust scrub, verify, paint, weld crack", "⇧3"], ["flyby", "Fly-by", "Ambient camera, no robot selected", "esc"]];
  const DEMOS = [["can", "Demo: Can pick-up", "Unit 3 walks to a dropped can, learned pick-up into the tote", "⇧C", "R3"], ["gas", "Demo: Gas leak", "Vapour from the orange line, Unit 2 finds the leak and closes the valve", "⇧G", "R2"], ["rust", "Demo: Rust scrub", "Unit 1 scrubs the rusted pipe clean", "⇧R", "R1"],
    ["nav", "Demo: Navigation", "Unit 3 walks a deck loop; drop a crate ahead and it re-routes", "⇧N", "R3"],
    ["deck", "Demo: Deck", "One deck camera; drop a can or add rust and the robots answer", "⇧D", null],
    ["reset", "Demo: Reset", "Back to the normal fleet", "⇧X", null]];
  const UNITS = { R1: ["Unit 1", "repair and weld"], R2: ["Unit 2", "gas"], R3: ["Unit 3", "patrol and pick-up"] };
  const CARDS = { f: ["fpv", "Head camera", ["fpv", "policy", "obs", "belief"]], b: ["boxes", "Detections", ["boxes", "fpv", "policy"]], l: ["lidar", "Lidar", ["lidar", "bev"]], d: ["depth", "Depth", ["depth", "obs", "belief", "policy"]] };
  const KNAME = { fpv: "Head camera", boxes: "Detections", lidar: "Lidar", bev: "Lidar", depth: "Depth", obs: "Policy input", belief: "Gas belief", policy: "Policy input" };
  const KIND = { rust: "rust repair", leak_check: "leak check", gas_survey: "gas survey", survey: "gas survey", leak_locate: "leak search", pick_can: "can pick-up", patrol: "patrol", weld: "weld repair",
    weld_repair: "weld repair", valve: "valve isolate", valve_isolate: "valve isolate", isolate: "valve isolate", retorque: "re-torque", paint: "paint", inspect: "inspection", inspection: "inspection", dock: "to dock" };
  const MET = { force_n: ["Force", "N", 1, 1], gas_lel: ["Gas", "%LEL", 100, 1], thermal_c: ["Temp", "°C", 1, 0], battery: ["Battery", "%", 100, 0], reach: ["Reach", "m", 1, 2], wind_ms: ["Wind", "m/s", 1, 1],
    rust: ["Rust", "%", 100, 0], rust_fraction: ["Rust", "%", 100, 0], cans: ["Cans", "", 1, 0], speed: ["Speed", "m/s", 1, 1], conf: ["Detector", "", 1, 2], tote: ["Tote", "cans", 1, 0] };

  const S = { h: {}, views: {}, series: {}, sel: null, selAt: -99, mc: false, card: null, g: false, strip: false, spot: false, help: false, sheet: null, seen: null, decided: new Set() };
  const hooks = [], subs = {}, sheets = [];
  const on = (n, f) => (typeof n === "function" ? hooks.push(n) : (subs[n] ||= []).push(f));
  const emit = (n, x) => (subs[n] || []).forEach((f) => { try { f(x); } catch (e) { console.warn(e); } });

  const scen = () => { const s = S.h.scenario; return s && s.id && s.id !== "flyby" ? s : null; };
  const selected = () => S.sel;
  const mode = () => (selected() ? "robot" : S.mc || scen() ? "mc" : "flyby");
  const robots = () => (Array.isArray(S.h.robots) ? S.h.robots.filter((r) => r && r.id) : []);
  const fleetBot = (id) => (S.h.fleet?.robots || []).find((r) => r && r.id === id);
  const ids = () => { const r = robots().map((x) => x.id); return r.length ? r : (S.h.fleet?.robots || []).map((x) => x?.id).filter(Boolean); };
  const bat = (id) => fleetBot(id)?.battery?.soc ?? S.h.sensors?.[id]?.battery;
  function task(id) {
    const f = fleetBot(id), k = f?.task ? String(f.task).split(":")[0] : "";
    if (k) return KIND[k] || nice(k);
    const st = f?.job_state || f?.state || robots().find((r) => r.id === id)?.state;
    return /DOCK|CHARG/.test(st || "") ? "docked" : st && st !== "WORK" ? nice(st) : "idle";
  }

  const queue = [];
  let busy = false;
  const qkey = (e) => e.name + (e.name === "views" ? ":" + e.args.robot : "");
  function post(name, args = {}) {
    if (MOCK) mockEvent(name, args);
    if (FULL) return;
    const e = { name, args }, i = /^(disturb|approve|reject)$/.test(name) ? -1 : queue.findIndex((x) => qkey(x) === qkey(e));
    if (i >= 0) queue[i] = e; else queue.push(e);
    pump();
  }
  async function pump() {
    if (busy || !queue.length) return;
    busy = true;
    const e = queue.shift();
    let wait = 1050;
    try {
      const r = await fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(e) });
      if (r.status === 429) { const j = await r.json().catch(() => ({})); if (!queue.some((x) => qkey(x) === qkey(e))) queue.unshift(e); wait = (+j.retry_s || 1) * 1000 + 60; }
      else if (!r.ok) console.warn("event", e.name, r.status);
    } catch (err) { console.warn(err); }
    setTimeout(() => { busy = false; pump(); }, wait);
  }

  function record(h) {
    const t = now();
    Object.entries(h.sensors || {}).forEach(([rid, v]) => v && typeof v === "object" && Object.entries(v).forEach(([k, x]) => {
      if (typeof x !== "number") return;
      const s = (S.series[rid + "." + k] ||= []);
      s.push([t, x]);
      while (s.length && s[0][0] < t - 30) s.shift();
    }));
  }
  function apply(h) {
    if (!h || typeof h !== "object") return;
    if (h.views && typeof h.views === "object") Object.entries(h.views).forEach(([rid, ks]) => { if (ks && typeof ks === "object") S.views[rid] = { ...(S.views[rid] || {}), ...ks }; });
    S.h = { ...S.h, ...h };
    const hs = "selected" in h ? h.selected || null : h.camera_mode && typeof h.camera_mode === "object" ? (h.camera_mode.mode === "chase" && h.camera_mode.robot) || null : undefined;
    if (hs !== undefined && now() - S.selAt > 4 && hs !== S.sel) { if (!hs) closeCard(); S.sel = hs; }
    const sid = h.scenario?.id && h.scenario.id !== "flyby" ? h.scenario.id : null;
    if ("scenario" in h && sid !== (S.scenId ?? null)) {
      if (sid) S.mc = true;
      const c = S.h.control || {}, ag = c.agent || S.h.orchestrator?.planner || {}, top = (xs) => (Array.isArray(xs) ? xs : []).reduce((m, x) => Math.max(m, +x?.t || 0), -Infinity);
      S.base = S.scenId === undefined || !sid ? null : { th: top(c.thoughts), calls: top(ag.calls), ref: top(c.refused || c.vetoed) };
    }
    if ("scenario" in h) S.scenId = sid;
    record(h);
    alerts(h.alerts);
    hooks.forEach((f) => { try { f(h, S); } catch (e) { console.warn(e); } });
    S.dirty = true;
  }
  function alerts(list) {
    if (!Array.isArray(list)) return;
    const top = list.reduce((m, a) => Math.max(m, +a?.seq || 0), 0);
    if (S.seen == null || top < S.seen) { S.seen = top; return; }
    list.filter((a) => a && +a.seq > S.seen).sort((a, b) => a.seq - b.seq).forEach((a) => {
      emit("alert", a);
      if (a.kind === "gas_alarm") pulse();
      if (a.kind === "toast") toast(a.text);
      if (a.kind === "approval" && mode() !== "mc") toast("Approval needed", (a.text ? a.text + " · " : "") + "press 0");
    });
    S.seen = top;
  }

  let pulseT = 0;
  function pulse() {
    B.classList.remove("alarm");
    void B.offsetWidth;
    B.classList.add("alarm");
    clearTimeout(pulseT);
    pulseT = setTimeout(() => B.classList.remove("alarm"), 3000);
  }
  function toast(title, sub, tone = "") {
    const box = $("toasts"), el = document.createElement("div");
    el.className = "toast glass " + tone;
    el.innerHTML = `<b>${esc(title)}</b>${sub ? `<span>${esc(sub)}</span>` : ""}`;
    box.prepend(el);
    while (box.children.length > 3) box.lastChild.remove();
    requestAnimationFrame(() => el.classList.add("on"));
    setTimeout(() => { el.classList.remove("on"); setTimeout(() => el.remove(), 320); }, 4200);
  }

  const onDemand = (k) => k === "f" || k === "b" || k === "d";
  function setSel(rid) {
    const old = selected();
    S.sel = rid; S.selAt = now();
    if (!rid) closeCard();
    else if (old && old !== rid && onDemand(S.card)) { post("views", { robot: old, on: false }); post("views", { robot: rid, on: true, kind: CARDS[S.card][0] }); }
    post("select", { robot: rid });
  }
  function wide() { S.mc = true; S.g = false; setSel(null); }
  const ZN = { rust: "Rust", can: "Can", flange: "Flange F5", weld: "Weld seam", obstacle: "Obstacle" };
  const cm = () => (S.h.camera_mode && typeof S.h.camera_mode === "object" ? S.h.camera_mode : {});
  function cam(mode, view) { S.mc = true; post("camera", view ? { mode, view } : { mode }); }
  function rigDeck() { cam(cm().mode === "deck" ? "rig" : "deck"); }
  function zoom() { const v = (S.h.cam_views || []).filter((k) => k !== "obstacle"); if (!v.length) { toast("No close-up here", "Start a scenario first"); return; } const i = v.indexOf(cm().view); cam("zoom", v[(i + 1) % v.length]); }
  function pick(n) { const rid = "R" + n; if (ids().length && !ids().includes(rid)) { toast(`No ${rid} on the rig`, "Robots reporting: " + ids().join(", ")); return; } setSel(rid); }
  function closeCard() { if (onDemand(S.card) && S.sel) post("views", { robot: S.sel, on: false }); S.card = null; }
  function toggleCard(k) {
    const rid = selected();
    if (!rid) { toast("Pick a robot first", "Press 1, 2 or 3"); return; }
    const prev = S.card;
    S.card = prev === k ? null : k;
    if (onDemand(S.card)) post("views", { robot: rid, on: true, kind: CARDS[S.card][0] });
    else if (onDemand(prev)) post("views", { robot: rid, on: false });
  }
  setInterval(() => { const rid = selected(); if (rid && onDemand(S.card)) post("views", { robot: rid, on: true, kind: CARDS[S.card][0] }); }, 20000);
  function startScenario(id) {
    if (id === "flyby") { flyby(); return; }
    closeCard();
    S.mc = true; S.sel = null; S.selAt = now();
    post("scenario", { id });
  }
  function demo(id, rid) {
    if (!rid) { closeCard(); S.mc = false; S.g = false; S.strip = false; S.sel = null; S.selAt = now(); post("demo", { id }); return; }
    if (S.sel && S.sel !== rid && onDemand(S.card)) post("views", { robot: S.sel, on: false });
    S.mc = true; S.sel = rid; S.selAt = now(); S.card = "f";
    post("demo", { id });
  }
  function flyby() {
    closeCard();
    S.mc = false; S.g = false; S.strip = false; S.sel = null; S.selAt = now();
    post("scenario", { id: "flyby" });
  }
  function disturb() {
    const s = scen();
    if (!s) { toast("No scenario running", "Start one from " + CMD); return; }
    const ds = Array.isArray(s.disturbances) ? s.disturbances : null, d = ds && s.next != null ? ds[s.next] : null;
    if (ds && !d) { toast("No disturbances left", s.name); return; }
    post("disturb", {});
    toast("Disturbance", d ? d.name : "Next one for " + (s.name || s.id));
  }
  function mute() {
    const v = !HUD.muted;
    if (window.Sound) window.Sound.mute(v); else HUD.muted = v;
    HUD.muted = v;
    emit("mute", v);
    toast(v ? "Sound off" : "Sound on", "M toggles");
  }
  function decide(job, ok) { S.decided.add(job); post(ok ? "approve" : "reject", { job, id: job }); toast(ok ? "Approved" : "Rejected", nice(job)); S.dirty = true; }

  function project(cam, p) {
    const T = cam.T, K = cam.K, d = [p[0] - T[0][3], p[1] - T[1][3], p[2] + 0.85 - T[2][3]];
    const c = [0, 1, 2].map((j) => d[0] * T[0][j] + d[1] * T[1][j] + d[2] * T[2][j]);
    return [K[0][0] * c[0] / c[2] + K[0][2], K[1][1] * c[1] / c[2] + K[1][2], c[2]];
  }
  function labels() {
    const L = $("labels"), m = mode(), cam = S.h.camera, sel = selected();
    const show = m !== "flyby" && S.h.camera_mode?.mode !== "flyby" && cam?.T && cam?.K && cam.w && !S.spot ? robots().filter((r) => Array.isArray(r.world_pos) && r.world_pos.length > 2 && (m === "mc" || r.id === sel)) : [];
    const keep = new Set(), W = innerWidth, H = innerHeight, at = [];
    const k = cam?.w ? Math.max(W / cam.w, H / cam.h) : 1, ox = cam?.w ? (W - cam.w * k) / 2 : 0, oy = cam?.h ? (H - cam.h * k) / 2 : 0;
    show.forEach((r) => {
      const [u, v, z] = project(cam, r.world_pos), x = ox + u * k, y = oy + v * k;
      if (!(z > 0.3) || z > 400 || x < 40 || x > W - 40 || y < 110 || y > H - 40 || !isFinite(x + y)) return;
      keep.add(r.id);
      let el = L.querySelector(`[data-id="${r.id}"]`);
      if (!el) { el = document.createElement("div"); el.dataset.id = r.id; el.innerHTML = `<div class="lc"><b></b><span></span></div>`; L.appendChild(el); }
      el.className = "lbl" + (r.id === sel ? " sel" : "") + (r.caption?.tone === "stop" ? " stop" : r.caption?.tone === "warn" ? " warn" : "");
      el.style.transform = `translate(${x.toFixed(1)}px,${y.toFixed(1)}px)`;
      txt(el.querySelector("b"), r.id);
      txt(el.querySelector("span"), task(r.id));
      at.push([el, x, y]);
    });
    [...L.children].forEach((el) => keep.has(el.dataset.id) || el.remove());
    const placed = [];
    at.sort((a, b) => b[2] - a[2]).forEach(([el, x, y]) => {
      const w = el.firstChild.offsetWidth, h = el.firstChild.offsetHeight, l = x - w / 2;
      let lift = 0;
      for (let i = 0; i < placed.length; i++) { const p = placed[i]; if (l < p[0] + p[2] + 6 && p[0] < l + w + 6 && Math.abs(y - lift - p[1]) < h + 4) { lift = y - p[1] + h + 6; i = -1; } }
      placed.push([l, y - lift, w]);
      el.style.setProperty("--lift", lift.toFixed(0) + "px");
    });
  }

  function capsule() {
    const s = scen(), el = $("capsule");
    el.classList.toggle("on", !!s);
    if (!s) return;
    const k = s.key, v = k && k.value != null && !Number.isNaN(+k.value) ? fmt(+k.value) : null;
    put(el, `<i></i><b>${esc(s.name || s.id)}</b>${s.step ? `<span>${esc(s.step)}</span>` : ""}${v != null ? `<em class="tnum" title="${esc(k.label || "")}">${esc(v)}<small>${esc(k.unit || "")}</small></em>` : ""}`);
  }
  let capLayer = 0;
  function caption() {
    const c = S.h.caption, el = $("caption");
    const html = mode() === "flyby" || !c || !(c.text || c.lead) ? "" : c.lead ? `<b>${esc(c.lead)}</b>${esc(c.rest || "")}` : esc(c.text);
    const key = html.replace(/\d+(\.\d+)?/g, "#"), cur = el.children[capLayer];
    if (key === el._k) { if (cur.innerHTML !== html) cur.innerHTML = html; return; }
    el._k = key; capLayer ^= 1;
    const nx = el.children[capLayer];
    nx.innerHTML = html; nx.className = "cap on " + (c?.tone === "stop" || c?.tone === "warn" ? c.tone : "");
    cur.classList.remove("on");
  }

  function fleetPanel() {
    const sel = selected(), rs = robots();
    const rows = ids().map((id, n) => {
      const b = bat(id), tone = rs.find((x) => x.id === id)?.caption?.tone;
      return `<button class="row ${id === sel ? "on" : ""}" data-robot="${esc(id)}"><i class="${esc(tone || "")}"></i><b>${esc(id)}</b><span>${esc(task(id))}</span><em class="tnum">${b == null ? "" : Math.round(b * 100) + "%"}</em></button>`;
    }).join("");
    const aps = (S.h.orchestrator?.approvals || []).filter((a) => a && a.job && !S.decided.has(a.job));
    const inbox = aps.map((a) => `<div class="ap"><b>${esc(nice(a.kind) || "request")} · ${esc(a.target || a.job)}</b>${a.text ? `<p>${esc(a.text)}</p>` : ""}<div class="ab"><button class="yes" data-ok="${esc(a.job)}">Approve</button><button data-no="${esc(a.job)}">Reject</button></div></div>`).join("");
    return `<h3>Fleet</h3>${rows || `<p class="none">No robots reporting yet</p>`}<h3>Operator inbox${aps.length ? ` <em class="tnum">${aps.length}</em>` : ""}</h3>${inbox || `<p class="none">Nothing waiting for you</p>`}`;
  }
  function geminiPanel() {
    const c = S.h.control || {}, ag = c.agent || S.h.orchestrator?.planner || {};
    const b = S.base, fresh = (k) => (x) => !b || (+x?.t || 0) > b[k];
    const th = (Array.isArray(c.thoughts) ? c.thoughts : []).filter((x) => x && x.text).filter(fresh("th")).slice(-6);
    const calls = (Array.isArray(ag.calls) ? ag.calls : []).filter(Boolean).filter(fresh("calls")).slice(-5).reverse();
    const vet = (Array.isArray(c.refused) ? c.refused : Array.isArray(c.vetoed) ? c.vetoed : []).filter(Boolean).filter(fresh("ref")).slice(-4).reverse();
    const cut = c.link === "CUT" || ag.link === "CUT";
    const meta = [cut ? "link cut · fleet on rules" : c.mode === "CLOUD" ? "cloud" : c.mode ? "edge" : "", c.calls_today != null ? c.calls_today + " calls today" : "", c.cost_today_usd != null ? "$" + (+c.cost_today_usd).toFixed(2) : ""].filter(Boolean).join(" · ");
    const sit = Array.isArray(c.situation) ? c.situation[0] : c.situation;
    const think = th.length ? `<ol class="th">${th.map((x, i) => `<li class="${i === th.length - 1 ? "new" : ""}">${esc(x.text)}</li>`).join("")}</ol>`
      : b ? `<p class="none">Argon is reading the new situation…</p>` : ag.text || sit ? `<ol class="th"><li class="new">${esc(String(ag.text || sit).split("\n")[0])}</li></ol>` : `<p class="none">Waiting for Argon's first thought</p>`;
    const callRows = calls.map((x) => `<div class="call ${x.ok === false ? "no" : ""}"><b>${esc(x.label || x.name || "call")}</b>${x.why || x.note ? `<p>${esc(x.why || x.note)}</p>` : ""}${x.ok === false && x.refused ? `<p class="ref">Refused · ${esc(x.refused)}</p>` : ""}</div>`).join("");
    const vetRows = vet.map((v) => `<div class="call no"><b>${esc(v.label || v.kind || "action")}</b>${v.why ? `<p class="ref">${esc(v.why)}</p>` : ""}</div>`).join("");
    return `<header><b>Gemini 4 Argon</b><span class="tnum">${esc(meta)}</span></header><h3>${c.thoughts_mode === "decisions" ? "Decisions" : "Thinking"}</h3>${think}<h3>Calls</h3>${callRows || `<p class="none">No calls yet</p>`}${vetRows ? `<h3>Refused by the safety envelope</h3>${vetRows}` : ""}`;
  }
  function left() {
    const el = $("left"), what = S.g ? "g" : mode() === "mc" ? "mc" : "";
    el.classList.toggle("on", !!what);
    el.classList.toggle("g", what === "g");
    if (what) put(el, what === "g" ? geminiPanel() : fleetPanel());
  }
  $("left").addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.dataset.robot) setSel(b.dataset.robot);
    else if (b.dataset.ok) decide(b.dataset.ok, true);
    else if (b.dataset.no) decide(b.dataset.no, false);
  });

  const C = {};
  $("card").querySelectorAll("[data-k]").forEach((e) => (C[e.dataset.k] = e));
  const pending = new Image();
  pending.onload = () => { C.img.src = pending.src; C.img.hidden = false; pending.bad = null; S.dirty = true; };
  pending.onerror = () => { pending.bad = pending._s; C.img.hidden = true; C.img.removeAttribute("src"); S.dirty = true; };
  function card() {
    const el = $("card"), k = S.card, rid = selected(), live = !!(k && rid);
    el.classList.toggle("on", live);
    B.classList.toggle("carded", live);
    if (!live) return;
    const [want, title, order] = CARDS[k], v = S.views[rid] || {}, kind = order.find((x) => v[x] && (v[x].url || v[x].jpg)), e = kind ? v[kind] : null;
    const src = e ? (e.jpg ? "data:image/jpeg;base64," + e.jpg : e.url) : null;
    txt(C.t, `${rid} · ${kind && kind !== want ? KNAME[kind] || nice(kind) : title}`);
    txt(C.k, k.toUpperCase());
    if (e?.w && e?.h) C.box.style.aspectRatio = `${e.w} / ${e.h}`;
    if (src && pending._s !== src) { pending._s = src; pending.src = src; }
    if (!src) { pending._s = null; C.img.hidden = true; }
    const ok = src && pending.bad !== src && !C.img.hidden;
    txt(C.none, ok ? "" : src && pending.bad !== src ? "Loading…" : onDemand(k) ? `Starting ${rid}'s ${k === "d" ? "depth view" : "head camera"}…` : `No ${title.toLowerCase()} view from ${rid} yet`);
    const bx = k === "b" && Array.isArray(e?.boxes) && e.w ? e.boxes : [], bk = JSON.stringify(bx) + (e?.w || 0);
    if (C.svg._k !== bk) {
      C.svg._k = bk;
      C.svg.setAttribute("viewBox", `0 0 ${e?.w || 1} ${e?.h || 1}`);
      C.svg.innerHTML = bx.map((b) => { const [x, y, w, h] = b.xywh || [b.x, b.y, b.w, b.h]; return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="2"/>${b.label ? `<text x="${x + 2}" y="${Math.max(e.w / 28, y - e.w / 120)}" font-size="${(e.w / 30).toFixed(1)}">${esc(b.label)}${b.score != null ? " " + (+b.score).toFixed(2) : ""}</text>` : ""}`; }).join("");
    }
    txt(C.cap, e?.caption ? String(e.caption).replace(new RegExp("^" + rid + "\\s*·\\s*"), "") : "");
  }

  function strip() {
    const el = $("strip");
    el.classList.toggle("on", S.strip);
    B.classList.toggle("stripped", S.strip);
    if (!S.strip) return;
    const rid = selected() || scen()?.main || ids()[0], s = S.h.sensors?.[rid] || {}, hl = new Set(Array.isArray(s.highlight) ? s.highlight : []);
    const keys = Object.keys(s).filter((k) => typeof s[k] === "number").sort((a, b) => hl.has(b) - hl.has(a));
    const shape = rid + "|" + keys.map((k) => k + (hl.has(k) ? "*" : "")).join();
    if (el._shape !== shape) {
      el._shape = shape;
      el.innerHTML = !rid ? `<p class="none">No sensors reporting yet</p>` : `<div class="who"><b>${esc(rid)}</b><small>${esc(task(rid))}</small></div>` + (keys.map((k) => {
        const [lb, un] = MET[k] || [nice(k).replace(/^./, (c) => c.toUpperCase()), ""];
        return `<div class="cell ${hl.has(k) ? "hl" : ""}" data-m="${esc(k)}"><span>${esc(lb)}</span><b class="tnum"><i></i><small>${esc(un)}</small></b><canvas></canvas></div>`;
      }).join("") || `<p class="none">No readings from ${esc(rid)} yet</p>`);
    }
    el.querySelectorAll(".cell").forEach((c) => {
      const m = c.dataset.m, mul = MET[m]?.[2] || 1, dg = MET[m]?.[3] ?? 1, x = s[m];
      txt(c.querySelector("i"), x == null ? "—" : (x * mul).toFixed(Math.abs(x * mul) >= 100 ? 0 : dg));
      spark(c.querySelector("canvas"), S.series[rid + "." + m] || [], c.classList.contains("hl") ? "#D97757" : "rgba(235,235,245,.45)");
    });
  }
  function spark(cv, data, col) {
    const dpr = Math.min(devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
    if (!w) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    if (data.length < 2) return;
    const t0 = now() - 30, vs = data.map((p) => p[1]), lo = Math.min(...vs), hi = Math.max(...vs), r = Math.max(hi - lo, Math.abs(hi) * 0.15, 1e-3);
    g.strokeStyle = col; g.lineWidth = 1.4; g.lineJoin = g.lineCap = "round";
    g.beginPath();
    data.forEach(([t, v], i) => { const x = ((t - t0) / 30) * w, y = h / 2 - ((v - (lo + hi) / 2) / r) * (h - 4); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.stroke();
  }

  const spot = $("spot"), inp = $("s-input"), list = $("s-list");
  let items = [], cur = 0;
  function rows() {
    const s = scen(), sid = s?.id || (S.h.scenario?.id ?? "flyby"), rid = selected(), R = [];
    DEMOS.forEach(([id, label, sub, key, r]) => R.push({ g: "Demos", id: "demo." + id, label, sub, key, on: sid === "demo_" + id || sid === id, run: () => demo(id, r) }));
    R.push({ g: "Demos", id: "deck.can", label: "Drop can", sub: "A can on the deck walkway; Unit 3 picks it up into its tote", key: "", run: () => post("demo", { id: "deck_can" }) });
    R.push({ g: "Demos", id: "deck.rust", label: "Add rust", sub: "Rust on the pipe; Unit 1 walks over and scrubs it clean", key: "", run: () => post("demo", { id: "deck_rust" }) });
    R.push({ g: "Demos", id: "drop.crate", label: "Drop crate", sub: "Crate 3 m in front of the walking robot; it re-routes", key: "", run: () => post("drop_obstacle", { kind: "crate", at: "ahead", ahead_m: 3 }) });
    SCEN.forEach(([id, label, sub, key]) => R.push({ g: "Scenarios", id: "scen." + id, label, sub, key, on: sid === id, run: () => startScenario(id) }));
    R.push({ g: "Views", id: "view.rig", label: "Rig", sub: "Wide exterior camera", key: "0", on: cm().mode === "rig" || cm().mode === "flyby", run: () => cam("rig") });
    R.push({ g: "Views", id: "view.deck", label: "Deck", sub: "Fixed camera over the work area", key: "0", on: cm().mode === "deck", run: () => cam("deck") });
    (S.h.cam_views || []).forEach((k) => R.push({ g: "Views", id: "view.z." + k, label: k === "obstacle" ? "Obstacle" : "Zoom: " + ZN[k], sub: k === "obstacle" ? "Latest drop on the walkway" : "Close fixed camera", key: k === "obstacle" ? "" : "Z", on: cm().view === k || (k === "obstacle" && cm().mode === "obstacle"), run: () => (k === "obstacle" ? cam("obstacle") : cam("zoom", k)) }));
    ["R1", "R2", "R3"].forEach((r, n) => R.push({ g: "Views", id: "view." + r, label: `${UNITS[r][0]} · ${r}`, sub: (ids().includes(r) ? task(r) : UNITS[r][1]), key: String(n + 1), on: rid === r, run: () => pick(n + 1) }));
    Object.entries(CARDS).forEach(([k, [, title]]) => R.push({ g: "Views", id: "card." + k, label: title, sub: rid ? `${rid}, right card` : "Select a robot first", key: k.toUpperCase(), on: S.card === k && !!rid, run: () => toggleCard(k) }));
    R.push({ g: "Views", id: "strip", label: "Sensor strip", sub: "Readings along the bottom", key: "S", on: S.strip, run: () => (S.strip = !S.strip) });
    R.push({ g: "Views", id: "gemini", label: "Gemini 4 Argon", sub: "Thoughts, calls and refusals", key: "G", on: S.g, run: () => (S.g = !S.g) });
    const ds = Array.isArray(s?.disturbances) ? s.disturbances : [];
    R.push({ g: "Disturbances", id: "clear", label: "Clear walkway", sub: "Remove every dropped can, crate, drum and puddle; routes restored", key: "", run: () => post("clear", { id: "all" }) });
    R.push({ g: "Disturbances", id: "drop.can", label: "Drop can in view", sub: "Small can where the camera looks; nearest gripper picks it up", key: "", run: () => post("drop_can", { size: "small", at: "view" }) });
    ds.forEach((d, i) => R.push({ g: "Disturbances", id: "dist." + (d.id || i), label: d.name || nice(d.id), sub: d.fired ? "Fired" : i === s.next ? "Next" : "Later", key: i === s.next ? "I" : d.fired ? "✓" : "", tag: i === s.next, dis: i !== s.next, run: disturb }));
    if (s && !ds.length) R.push({ g: "Disturbances", id: "dist.next", label: "Next disturbance", sub: s.name || s.id, key: "I", run: disturb });
    R.push({ g: "Help", id: "help", label: "Keyboard shortcuts", sub: "Every key on one card", key: "?", run: () => (S.help = true) });
    sheets.forEach((v) => R.push({ g: "Help", id: "sheet." + v.id, label: v.title, sub: v.sub || "Measured results", key: v.key || "", run: () => openSheet(v) }));
    R.push({ g: "Help", id: "qr", label: "Phone controls for judges", sub: "QR code for /interactive", key: "⇧J", run: qr });
    R.push({ g: "Help", id: "mute", label: HUD.muted ? "Sound on" : "Mute sound", sub: HUD.muted ? "Sound is off" : "Alarm, chime, ping, click", key: "M", run: mute });
    return R;
  }
  const GROUPS = ["Demos", "Scenarios", "Views", "Disturbances", "Help"];
  function fuzzy(q, s) {
    s = s.toLowerCase();
    let i = 0, sc = 0, prev = -2;
    const idx = [];
    for (let j = 0; j < s.length && i < q.length; j++) if (s[j] === q[i]) { sc += 1 + (j === prev + 1 ? 2 : 0) + (j === 0 || s[j - 1] === " " ? 3 : 0); prev = j; idx.push(j); i++; }
    return i === q.length ? { sc: sc - s.length * 0.02, idx } : null;
  }
  function filter(scroll = true) {
    const q = inp.value.trim().toLowerCase(), all = rows();
    items = !q ? all : all.map((r) => ({ ...r, m: fuzzy(q, r.label) || ((r.sub + " " + r.g + " " + r.key).toLowerCase().split(/\s+/).some((w) => w && w.startsWith(q)) ? { sc: 0.5, idx: [] } : null) }))
      .filter((r) => r.m).sort((a, b) => GROUPS.indexOf(a.g) - GROUPS.indexOf(b.g) || b.m.sc - a.m.sc);
    if (q && items.length) { const best = items.reduce((a, b) => (b.m.sc > a.m.sc ? b : a)); cur = items.indexOf(best); } else cur = Math.max(0, items.findIndex((r) => (scen() ? r.tag : r.g === "Scenarios")));
    draw(scroll);
  }
  const hl = (t, m) => (m?.idx.length ? [...t].map((ch, i) => (m.idx.includes(i) ? `<mark>${esc(ch)}</mark>` : esc(ch))).join("") : esc(t));
  function draw(scroll = true) {
    let g0 = "";
    list.innerHTML = items.map((r, i) => (r.g !== g0 ? `<li class="gh">${esc((g0 = r.g))}</li>` : "") +
      `<li role="option" data-i="${i}" class="${r.on ? "on" : ""} ${r.dis ? "dis" : ""} ${r.tag ? "next" : ""}"><i class="st"></i><span class="lb">${hl(r.label, r.m)}</span><span class="sb">${esc(r.sub || "")}</span>${r.key ? `<kbd>${esc(r.key)}</kbd>` : ""}</li>`).join("") || `<li class="empty">No results</li>`;
    mark(cur, scroll);
  }
  function mark(i, scroll = true) {
    if (!items.length) return;
    cur = (i + items.length) % items.length;
    list.querySelector("li.sel")?.classList.remove("sel");
    const li = list.querySelector(`li[data-i="${cur}"]`);
    li?.classList.add("sel");
    if (scroll) li?.scrollIntoView({ block: "nearest" });
  }
  function openSpot(text = "") { S.spot = true; spot.classList.add("on"); B.classList.add("spotting"); inp.value = text; filter(); inp.focus(); }
  function closeSpot() { S.spot = false; spot.classList.remove("on"); B.classList.remove("spotting"); inp.blur(); }
  function runRow(r) { if (!r || r.dis) return; emit("spotlight", { id: r.id, label: r.label }); closeSpot(); r.run(); S.dirty = true; }
  inp.addEventListener("input", filter);
  list.addEventListener("pointermove", (e) => { const li = e.target.closest("li[data-i]"); if (li && (e.movementX || e.movementY) && +li.dataset.i !== cur) mark(+li.dataset.i); });
  list.addEventListener("click", (e) => { const li = e.target.closest("li[data-i]"); if (li) runRow(items[+li.dataset.i]); });
  addEventListener("pointerdown", (e) => { if (S.spot && !spot.contains(e.target)) closeSpot(); });

  const HELP = [["⌘K", "Spotlight"], ["0", "Rig ↔ Deck camera"], ["Z", "Next close-up"], ["1 2 3", "Select a robot"], ["F", "Head camera"], ["B", "Detection boxes"], ["L", "Lidar"], ["D", "Depth"], ["S", "Sensor strip"],
    ["G", "Gemini panel"], ["I", "Next disturbance"], ["⇧1 ⇧2 ⇧3", "Start a scenario"], ["⇧C ⇧G ⇧R ⇧N ⇧D", "Demo: can pick-up, gas leak, rust scrub, navigation, deck"], ["⇧X", "Demo reset"], ["esc", "Back to fly-by"], ["M", "Mute"], ["?", "This card"]];
  function help() {
    const el = $("help");
    el.classList.toggle("on", S.help);
    if (S.help) put(el, `<h2>Keyboard</h2><dl>${HELP.map(([k, t]) => `<dt>${k.split(" ").map((x) => `<kbd>${esc(x === "⌘K" ? CMD : x)}</kbd>`).join("")}</dt><dd>${esc(t)}</dd>`).join("")}</dl><p>Same key again closes a card. One card at a time.</p>`);
  }
  $("help").addEventListener("click", () => (S.help = false));

  function openSheet(v) {
    S.sheet = v;
    const el = $("sheet");
    el.innerHTML = `<header><b>${esc(v.title)}</b><button aria-label="Close"><kbd>esc</kbd></button></header><div class="sb"></div>`;
    el.querySelector("button").onclick = () => (S.sheet = null);
  }
  function sheet() {
    const el = $("sheet");
    el.classList.toggle("on", !!S.sheet);
    if (S.sheet && now() - (el._t || 0) > 0.25) { el._t = now(); S.sheet.render(el.querySelector(".sb"), S); }
  }
  const PAD = "https://rig.zeecloud.org/interactive";
  let qrLib = null, pad = null;
  function qr() {
    if (pad) { pad.remove(); pad = null; return; }
    pad = document.createElement("div");
    pad.className = "qr-pop glass";
    pad.innerHTML = `<div class="qr-code"></div><div><b>Phone controls</b><p>Scan to drop a can, roll a drum, change the weather or close a walkway.</p><a href="/interactive" target="_blank" rel="noopener">rig.zeecloud.org/interactive</a></div>`;
    pad.onclick = (e) => { if (e.target.tagName !== "A") { pad.remove(); pad = null; } };
    B.appendChild(pad);
    qrLib ||= new Promise((ok) => { const sc = document.createElement("script"); sc.src = "https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"; sc.onload = ok; document.head.appendChild(sc); });
    const box = pad.querySelector(".qr-code");
    qrLib.then(() => new QRCode(box, { text: PAD, width: 180, height: 180, colorDark: "#1d1a18", colorLight: "#F5F4EE", correctLevel: QRCode.CorrectLevel.M }));
  }

  const KEYS = { 0: rigDeck, z: zoom, 1: () => pick(1), 2: () => pick(2), 3: () => pick(3), f: () => toggleCard("f"), b: () => toggleCard("b"), l: () => toggleCard("l"), d: () => toggleCard("d"),
    s: () => (S.strip = !S.strip), g: () => (S.g = !S.g), i: disturb, m: mute };
  addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if ((e.metaKey || e.ctrlKey) && k === "k") { e.preventDefault(); S.spot ? closeSpot() : openSpot(); return; }
    if (S.spot) {
      if (k === "escape") { e.preventDefault(); closeSpot(); }
      else if (k === "arrowdown" || k === "arrowup") { e.preventDefault(); mark(cur + (k === "arrowdown" ? 1 : -1)); }
      else if (k === "enter") { e.preventDefault(); runRow(items[cur]); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA/.test(e.target.tagName)) return;
    if (k === "/") { e.preventDefault(); openSpot(); return; }
    if (e.key === "?") { e.preventDefault(); S.help = !S.help; S.dirty = true; return; }
    if (k === "escape") {
      if (S.help || S.sheet || pad) { S.help = false; S.sheet = null; if (pad) { pad.remove(); pad = null; } }
      else flyby();
      S.dirty = true;
      return;
    }
    if (e.shiftKey && /^Digit[123]$/.test(e.code)) { e.preventDefault(); startScenario(SCEN[+e.code[5] - 1][0]); S.dirty = true; return; }
    if (e.shiftKey && e.code === "KeyJ") { qr(); return; }
    const dm = e.shiftKey && DEMOS.find((d) => "Key" + d[3].slice(1) === e.code);
    if (dm) { e.preventDefault(); demo(dm[0], dm[4]); S.dirty = true; return; }
    if (e.shiftKey && e.code === "KeyB" && sheets[0]) { S.sheet === sheets[0] ? (S.sheet = null) : openSheet(sheets[0]); return; }
    if (e.shiftKey) return;
    const a = KEYS[k];
    if (a) { e.preventDefault(); a(); S.dirty = true; }
  });

  let frameT = 0;
  function loop() {
    const m = mode();
    B.classList.toggle("flyby", m === "flyby");
    B.classList.toggle("robot", m === "robot");
    labels();
    if (S.dirty || now() - frameT > 0.12) {
      S.dirty = false; frameT = now();
      capsule(); caption(); left(); card(); strip(); help(); sheet();
      if (S.spot && now() - (spot._t || 0) > 1) { spot._t = now(); const k = cur, q = inp.value, st = list.scrollTop; filter(false); if (!q) mark(k, false); list.scrollTop = st; }
    }
    if (FULL) backdrop(m);
    requestAnimationFrame(loop);
  }
  function backdrop(m) {
    const el = $("backdrop"), n = m === "robot" ? 2 : 1;
    if (el._n !== n) { el._n = n; el.style.backgroundImage = `url(/api/mock/shot/${n})`; }
  }

  function liveState() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/state");
    ws.onmessage = (ev) => { try { const h = JSON.parse(ev.data); apply(MOCK ? { ...h, ...fill(h) } : h); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(liveState, 1500);
  }

  const MK = { scen: null, sel: null, t0: now(), alerts: [], seq: 0, fired: {} };
  const DIST = { gas: [["leak_reopens", "Leak reopens"], ["second_leak", "Second flange leaks"], ["cut_link", "Gemini link cut"]],
    patrol: [["more_cans", "More small cans"], ["unfamiliar_object", "Unfamiliar object"], ["drum", "205 L drum"]], repair: [["rust_returns", "Rust reappears"], ["gust", "Wind gust"]] };
  const STEPS = { gas: ["Gas sensor on flange F5 tripped", "Argon checking its plan in simulation", "Gas robot localising the leak", "Valve isolation waiting for the operator", "Weld held by the hot-work gate"],
    patrol: ["Patrol robot walking its route", "Re-routing around a blocked walkway", "Picking up a fallen can", "Back to dock to charge"], repair: ["Rust detector found a patch", "Scrubbing under the 40 N shield", "Verifying the patch is clean", "Repairing a weld crack"] };
  const CAPS = { gas: ["Gas detector GD-F5", " · 14 %LEL and rising"], patrol: ["Unit 3", " · walking the patrol route on the middle deck"], repair: ["Unit 1", " · scrubbing the riser at 31 N"] };
  const MAIN = { gas: "R2", patrol: "R3", repair: "R1" };
  const THOUGHTS = ["GD-F5 reads 14 %LEL and rising; wind from the west at 5 m/s puts the plume over the weld bay.", "Hot work near F5 must stop until gas is under 10 %LEL.",
    "Checking the valve isolation plan in simulation before filing it.", "Simulated plan keeps the weld frozen and R2 upwind of the plume.", "Filing valve isolate F5 for the operator. I can't approve it myself.",
    "R2's particle filter has narrowed the source to the F5 flange."];
  const mkAlert = (kind, text) => { MK.alerts = [...MK.alerts, { seq: ++MK.seq, kind, text, t: now() }].slice(-10); };
  function mockEvent(name, a) {
    if (name === "scenario") {
      MK.scen = a.id === "flyby" ? null : a.id; MK.t0 = now(); MK.fired = {}; MK.sel = null;
      if (MK.scen === "gas") setTimeout(() => MK.scen === "gas" && mkAlert("gas_alarm", "Gas at flange F5 · 14 %LEL"), 1800);
    }
    if (name === "disturb" && MK.scen) { const d = DIST[MK.scen].find(([id]) => !MK.fired[id]); if (d) { MK.fired[d[0]] = 1; mkAlert("disturbance", d[1]); if (/leak/.test(d[0])) mkAlert("gas_alarm", d[1]); } }
    if (name === "select") MK.sel = a.robot ?? null;
  }
  function mockScenario() {
    const sc = MK.scen;
    if (!sc) return null;
    const t = now() - MK.t0, steps = STEPS[sc], si = Math.min(steps.length - 1, Math.floor(t / 10));
    const ds = DIST[sc].map(([id, name]) => ({ id, name, fired: !!MK.fired[id] })), nx = ds.findIndex((d) => !d.fired);
    const key = sc === "gas" ? { label: "gas", value: 14.2 + 2.4 * Math.sin(t / 3), unit: "%LEL" } : sc === "repair" ? { label: "brush force", value: 31 + 5 * Math.sin(t * 1.3), unit: "N" } : { label: "cans", value: Math.min(5, Math.floor(t / 12)), unit: "cans" };
    return { id: sc, name: SCEN.find((s) => s[0] === sc)[1], step: steps[si], step_i: si, steps, key, disturbances: ds, next: nx < 0 ? null : nx, main: MAIN[sc] };
  }
  let lidarUrl = null, depthUrl = null;
  function synth() {
    const c = document.createElement("canvas"); c.width = 256; c.height = 192;
    const g = c.getContext("2d");
    g.fillStyle = "#0b0c0e"; g.fillRect(0, 0, 256, 192);
    for (let a = 0; a < 360; a += 1) { const r = 40 + 30 * Math.abs(Math.sin(a * 0.035)) + (a % 90 < 20 ? 25 : 0), x = 128 + r * Math.cos(a / 57.3), y = 96 + r * Math.sin(a / 57.3); g.fillStyle = "rgba(235,235,245,.85)"; g.fillRect(x, y, 1.6, 1.6); }
    g.fillStyle = "#D97757"; g.beginPath(); g.moveTo(128, 88); g.lineTo(134, 102); g.lineTo(122, 102); g.fill();
    lidarUrl = c.toDataURL("image/jpeg", 0.85);
    const gr = g.createLinearGradient(0, 0, 0, 192); gr.addColorStop(0, "#111"); gr.addColorStop(1, "#bbb"); g.fillStyle = gr; g.fillRect(0, 0, 256, 192);
    g.fillStyle = "#555"; g.fillRect(150, 50, 60, 90); g.fillStyle = "#888"; g.fillRect(40, 90, 50, 60);
    depthUrl = c.toDataURL("image/jpeg", 0.85);
  }
  function mockViews(rids) {
    if (!lidarUrl) synth();
    const out = {};
    rids.forEach((r, i) => {
      out[r] = { fpv: { url: "/api/mock/shot/3", w: 1280, h: 720, caption: `${r} · head cam · 512×288 · 5 Hz`, seq: 1 },
        boxes: { url: "/api/mock/shot/3", w: 1280, h: 720, caption: `${r} · detector · 2 boxes`, seq: 1, boxes: [{ xywh: [520, 250, 260, 200], label: "rust", score: 0.91 }, { xywh: [860, 420, 160, 120], label: "can", score: 0.78 }] },
        lidar: { url: lidarUrl, w: 256, h: 192, caption: `${r} · lidar BEV · 360 rays`, seq: 1 }, depth: { url: depthUrl, w: 256, h: 192, caption: `${r} · depth · ray cast`, seq: 1 } };
    });
    return out;
  }
  function mockControl(t) {
    const n = MK.scen ? Math.min(THOUGHTS.length, 1 + Math.floor(t / 5)) : 2;
    return { mode: MK.fired.cut_link ? "EDGE" : "CLOUD", link: MK.fired.cut_link ? "CUT" : "UP", model: "gemini-4-argon", calls_today: 67, cost_today_usd: 0.46,
      thoughts: THOUGHTS.slice(0, n).map((text, i) => ({ t: i * 5, text })),
      agent: { calls: [{ label: "Dispatch R2 gas survey", ok: true, why: "Nearest gas robot with a full battery, 40 m from F5." }, { label: "Request valve isolate F5", ok: true, why: "Leak localised to F5; isolation needs a human." },
        { label: "Resume weld on R1", ok: false, why: "Seam is nearly finished.", refused: "hot-work gate: gas above 10 %LEL" }] },
      vetoed: [{ label: "Approve valve isolate F5", why: "Gemini can never approve its own request" }] };
  }
  function fill(h) {
    const t = now() - MK.t0, out = { scenario: mockScenario() };
    if (!Array.isArray(h.alerts)) out.alerts = MK.alerts;
    if (!Array.isArray(h.control?.thoughts)) out.control = { ...(h.control || {}), ...(h.control ? { thoughts: mockControl(t).thoughts } : mockControl(t)) };
    const rids = (h.robots || []).map((r) => r.id).filter(Boolean), mv = mockViews(rids);
    out.views = Object.fromEntries(rids.map((r) => [r, { ...mv[r], ...Object.fromEntries(Object.entries(h.views?.[r] || {}).filter(([, v]) => v)) }]));
    if (h.sensors) out.sensors = Object.fromEntries(Object.entries(h.sensors).map(([r, v]) => [r, Array.isArray(v?.highlight) ? v : { ...v, highlight: MK.scen === "gas" ? ["gas_lel"] : MK.scen === "repair" ? ["force_n"] : ["battery"] }]));
    if (MK.scen === "gas" && t > 20 && !h.orchestrator?.approvals?.length) out.orchestrator = { ...(h.orchestrator || {}), approvals: [{ job: "valve_isolate:F5", kind: "valve_isolate", target: "F5", text: "Close the F5 isolation valve. R2 localised the leak and Argon checked the plan in simulation." }] };
    return out;
  }
  function mockHeader() {
    const t = now() - MK.t0, T = now(), sel = MK.sel, sc = MK.scen;
    const place = (id, i) => { const [u, v, z] = sel ? (id === sel ? [960, 600, 12] : [-900, 600, 30]) : [[560, 640, 30], [930, 560, 34], [1320, 690, 28]][i]; return [(u - 960 + 18 * Math.sin(T / 3 + i)) * z / 1400, (v - 540) * z / 1400, z - 0.85]; };
    const tasks = { R1: sc === "gas" ? "weld:riser_east" : "rust:riser_mid", R2: sc === "gas" ? "gas_survey:F5" : "leak_check:F5", R3: sc === "patrol" ? "pick_can:can_2" : "patrol:route_a" };
    const rs = ["R1", "R2", "R3"].map((id, i) => ({ id, name: `${UNITS[id][0]} · ${UNITS[id][1]}`, state: "WORK", world_pos: place(id, i), caption: { tone: sc === "gas" && id === "R1" ? "warn" : "act" } }));
    const caps = sc ? CAPS[sc] : ["Live", " · three robots on the rig"];
    const sens = {
      R1: { force_n: 31 + 5 * Math.sin(T * 1.3), thermal_c: 38 + Math.sin(T / 4), battery: 0.91, reach: 0.62, highlight: sc === "repair" ? ["force_n"] : sc === "gas" ? ["thermal_c"] : [] },
      R2: { gas_lel: 0.142 + 0.024 * Math.sin(T / 3), wind_ms: 5.4 + 0.6 * Math.sin(T / 2), battery: 0.74, thermal_c: 33, highlight: ["gas_lel", "wind_ms"] },
      R3: { battery: 0.63, speed: 0.8 + 0.2 * Math.sin(T), tote: 2, thermal_c: 31, highlight: sc === "patrol" ? ["tote", "battery"] : [] } };
    return { t: T, scenario: mockScenario(), selected: sel, robots: rs, camera: { K: [[1400, 0, 960], [0, 1400, 540], [0, 0, 1]], T: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], w: 1920, h: 1080 },
      caption: { lead: caps[0], rest: caps[1], text: caps.join("") }, alerts: MK.alerts, control: mockControl(t), sensors: sens, views: mockViews(["R1", "R2", "R3"]),
      fleet: { robots: rs.map((r) => ({ id: r.id, task: tasks[r.id], battery: { soc: sens[r.id].battery } })) },
      orchestrator: { approvals: sc === "gas" && t > 20 ? [{ job: "valve_isolate:F5", kind: "valve_isolate", target: "F5", text: "Close the F5 isolation valve. R2 localised the leak and Argon checked the plan in simulation." }] : [] } };
  }

  window.HUD = { S, on, emit, post, toast, apply, esc, now, json: (u) => fetch(u).then((r) => (r.ok ? r.json() : null)).catch(() => null),
    view: (v) => sheets.push({ sub: v.id === "bench" ? "Measured results, 95 % intervals" : "", ...v }), select: setSel, scenario: startScenario, disturb, pulse };
  addEventListener("DOMContentLoaded", () => {
    $("hint").textContent = CMD;
    if (FULL) { apply(mockHeader()); setInterval(() => apply(mockHeader()), 200); txt($("status"), "mock"); B.classList.add("mock"); } else liveState();
    if (!Q.has("quiet")) openSpot();
    requestAnimationFrame(loop);
  });
})();
