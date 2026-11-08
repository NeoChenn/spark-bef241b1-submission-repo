(() => {
  const DECKS = ["top_deck", "middle_deck", "lower_deck"];
  const ROBOT = ["#6FA8FF", "#5CCB8A", "#F2C14E", "#E58AD8"];
  const CORAL = "#D97757", RED = "#F85149", GHOST = "rgba(200,204,210,", AMBER = "#FF9F43";
  const DRAW_MS = 900, GHOST_MS = 9000, PAD = 8, HEAD = 14;
  const canvas = document.getElementById("fleet-canvas");
  if (!canvas) return;
  const wx = document.getElementById("fleet-wx"), msg = document.getElementById("fleet-msg");
  let map = null, snap = null, deckZ = {}, geom = null, shown = null;
  const anim = {};
  const now = () => performance.now();

  const deckOf = (z) => {
    let best = null, d = 1e9;
    for (const k of DECKS) { const dz = Math.abs(deckZ[k] - z); if (dz < d) { d = dz; best = k; } }
    return d < 1.5 ? best : null;
  };

  function layout() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = canvas.clientWidth || 540, h = canvas.clientHeight || 170;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const b = map.bounds, pw = (w - PAD * 4) / 3, ph = h - HEAD - PAD;
    const s = Math.min(pw / (b.xmax - b.xmin), ph / (b.ymax - b.ymin));
    geom = { w, h, pw, ph, s, b };
    return g;
  }

  const px = (deck, x, y) => {
    const { pw, s, b } = geom, i = DECKS.indexOf(deck);
    return [PAD + i * (pw + PAD) + (x - b.xmin) * s, HEAD + (b.ymax - y) * s];
  };

  function segs(pts) {
    const out = [];
    for (let k = 1; k < (pts || []).length; k++) {
      const a = pts[k - 1], c = pts[k], da = deckOf(a[2]), dc = deckOf(c[2]);
      if (da && da === dc) out.push([px(da, a[0], a[1]), px(dc, c[0], c[1])]);
    }
    return out;
  }

  function stroke(g, list, color, width, dash, frac) {
    g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []); g.lineCap = "round";
    let total = 0;
    list.forEach(([a, c]) => { total += Math.hypot(c[0] - a[0], c[1] - a[1]); });
    let left = frac == null ? Infinity : total * frac;
    g.beginPath();
    for (const [a, c] of list) {
      if (left <= 0) break;
      const L = Math.hypot(c[0] - a[0], c[1] - a[1]), u = Math.min(1, left / (L || 1));
      g.moveTo(a[0], a[1]); g.lineTo(a[0] + (c[0] - a[0]) * u, a[1] + (c[1] - a[1]) * u);
      left -= L;
    }
    g.stroke(); g.setLineDash([]);
  }

  function draw() {
    requestAnimationFrame(draw);
    if (!map || canvas.offsetParent === null) return;
    const g = layout(), t = now();
    g.clearRect(0, 0, geom.w, geom.h);
    g.font = "10px Inter, system-ui, sans-serif";
    DECKS.forEach((d, i) => {
      const x0 = PAD + i * (geom.pw + PAD);
      g.fillStyle = "rgba(255,255,255,0.03)"; g.fillRect(x0, HEAD, geom.pw, geom.ph);
      g.fillStyle = "rgba(245,244,238,0.45)"; g.fillText(d.replace("_deck", "").toUpperCase(), x0 + 2, 10);
    });
    for (const w of map.walkways) {
      if (w.kind === "stairs") {
        for (const p of [w.a, w.b]) { const d = deckOf(p[2]); if (!d) continue; const [x, y] = px(d, p[0], p[1]); g.strokeStyle = "#9A7BE0"; g.lineWidth = 1.2; g.strokeRect(x - 3, y - 3, 6, 6); }
        continue;
      }
      stroke(g, segs([w.a, w.b]), w.narrow ? "rgba(201,138,58,0.55)" : "rgba(160,170,182,0.55)", w.narrow ? 1 : 2);
    }
    if (!snap) return;
    const hz = snap.hazards || {};
    for (const c of hz.closed || []) stroke(g, segs([c.a, c.b]), RED, 2, [4, 3]);
    for (const z of snap.zones || []) {
      const d = deckOf(z.xyz[2]); if (!d) continue;
      const [x, y] = px(d, z.xyz[0], z.xyz[1]);
      g.strokeStyle = z.kind === "gas" ? RED : "#F2D14E"; g.lineWidth = 1;
      g.beginPath(); g.arc(x, y, Math.max(3, z.radius * geom.s), 0, 7); g.stroke();
    }
    for (const o of hz.obstacles || []) {
      const d = deckOf(o.xyz[2]); if (!d) continue;
      const [x, y] = px(d, o.xyz[0], o.xyz[1]), r = Math.max(4, o.radius * geom.s);
      g.fillStyle = o.effect === "slip" ? "rgba(90,160,255,0.35)" : "rgba(255,159,67,0.35)";
      g.strokeStyle = o.effect === "slip" ? "#5AA0FF" : AMBER; g.lineWidth = 1.5;
      g.beginPath(); g.roundRect ? g.roundRect(x - r, y - r, 2 * r, 2 * r, 2) : g.rect(x - r, y - r, 2 * r, 2 * r); g.fill(); g.stroke();
      g.fillStyle = "#F5F4EE"; g.fillText(o.kind[0].toUpperCase(), x - 3, y + 3.5);
    }
    snap.robots.forEach((r, k) => {
      const a = anim[r.id] || {};
      if (r.prev_path && a.ghost !== r.reroute_t) { a.ghost = r.reroute_t; a.t0 = t; a.prev = r.prev_path; anim[r.id] = a; }
      const age = a.t0 ? t - a.t0 : Infinity;
      if (a.prev && r.prev_path) stroke(g, segs(a.prev), GHOST + Math.max(0.25, 0.8 - 0.55 * age / GHOST_MS) + ")", 2, [3, 3]);
      stroke(g, segs(r.path), CORAL, 2.2, null, Math.min(1, age / DRAW_MS));
      const d = deckOf(r.pose.z) || DECKS.reduce((m, n) => (Math.abs(deckZ[n] - r.pose.z) < Math.abs(deckZ[m] - r.pose.z) ? n : m));
      const [x, y] = px(d, r.pose.x, r.pose.y);
      g.fillStyle = r.state === "FAILED" ? RED : ROBOT[k % ROBOT.length];
      g.strokeStyle = "#141413"; g.lineWidth = 1.5;
      g.beginPath(); g.arc(x, y, age < DRAW_MS * 2 ? 5.5 : 4.5, 0, 7); g.fill(); g.stroke();
      g.fillStyle = "#F5F4EE"; g.fillText(r.id, x + 6, y - 5);
    });
  }

  function badge(w) {
    if (!wx || !w) return;
    wx.textContent = w.level === "calm" ? "calm" : `${w.level} · wind ${w.wind.toFixed(1)} → ${Math.round(w.wind_dir)}° · sight ${Math.round(w.perception_range_m)} m`;
    wx.className = "fleet-wx " + (w.level === "calm" ? "" : w.closes_outer ? "stop" : "warn");
  }

  function say(text, cls) {
    if (!msg || !text) return;
    msg.textContent = text;
    msg.className = "fleet-msg " + (cls || "");
  }

  function onState(s) {
    snap = s;
    badge((s.hazards || {}).weather);
    const evs = (s.recent_events || []).filter((e) => ["reroute", "needs_human", "drop_obstacle", "set_weather", "close_walkway", "close_stairs", "clear", "reopen", "blocked"].includes(e.kind));
    const e = evs[evs.length - 1], key = e && e.t + e.message;
    if (e && key !== shown) { shown = key; say(`t=${Math.round(e.t)}s  ${e.message}`, e.kind === "reroute" ? "re" : e.kind === "needs_human" ? "stop" : ""); }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(proto + "//" + location.host + "/ws/fleet");
    ws.onmessage = (ev) => { try { onState(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(connect, 2000);
  }

  function pick(e) {
    if (!geom) return null;
    const rc = canvas.getBoundingClientRect(), mx = e.clientX - rc.left, my = e.clientY - rc.top;
    const i = Math.floor((mx - PAD) / (geom.pw + PAD));
    if (i < 0 || i > 2) return null;
    const deck = DECKS[i], { pw, s, b } = geom;
    const x = (mx - PAD - i * (pw + PAD)) / s + b.xmin, y = b.ymax - (my - HEAD) / s;
    let best = null;
    for (const w of map.walkways) {
      if (w.kind === "stairs" || deckOf(w.a[2]) !== deck) continue;
      const dx = w.b[0] - w.a[0], dy = w.b[1] - w.a[1], L2 = dx * dx + dy * dy || 1;
      const u = Math.max(0, Math.min(1, ((x - w.a[0]) * dx + (y - w.a[1]) * dy) / L2));
      const d = Math.hypot(w.a[0] + u * dx - x, w.a[1] + u * dy - y);
      if (!best || d < best.d) best = { d, w };
    }
    return best && best.d * s < 10 ? { deck, x, y, w: best.w } : null;
  }

  canvas.addEventListener("click", (e) => {
    const p = pick(e);
    if (!p || !window.RIG_POST) return;
    if (e.shiftKey) window.RIG_POST("close_walkway", { edge: p.w.id });
    else window.RIG_POST("drop_obstacle", { kind: "crate", at: [p.x, p.y], deck: p.deck });
  });
  addEventListener("rig:command-result", (e) => {
    const { ok, body } = e.detail;
    if (!ok) say(body && body.detail ? String(body.detail) : "command failed", "stop");
    else if (body && body.fleet && body.fleet.log && body.fleet.log.length) say(body.fleet.log[body.fleet.log.length - 1].message);
  });

  fetch("/api/fleet/map").then((r) => (r.ok ? r.json() : null)).then((m) => {
    if (!m) { say("fleet planner offline", "stop"); return; }
    map = m;
    m.decks.forEach((d) => { deckZ[d.name] = d.z; });
    fetch("/api/fleet").then((r) => r.json()).then(onState).catch(() => {});
    connect();
  }).catch(() => say("fleet planner offline", "stop"));
  requestAnimationFrame(draw);
  window.FleetMap = { state: () => snap, map: () => map };
})();
