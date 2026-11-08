(() => {
  const $ = (id) => document.getElementById(id);
  const S = { fleet: null, map: null, deck: null, level: "calm", time: "night", seen: null, last: {}, dirty: true };
  const NAMES = { calm: "Clear", wind: "Wind", rain: "Rain", fog: "Fog", storm: "Storm", dusk: "Dusk", night: "Night" };
  const SHOW = { reroute: "warn", blocked: "stop", barrel_stopped: "ok", needs_human: "stop" };
  const pretty = (m) => String(m || "").replace(/ -> /g, " → ").replace(/ deg\b/g, "°").replace(/_/g, " ");

  function toast(kind, title, sub, ms = 7000) {
    const box = $("toasts"), el = document.createElement("div");
    el.className = "toast"; el.dataset.k = kind;
    el.innerHTML = `<i></i><div><b></b><span></span></div>`;
    el.querySelector("b").textContent = title; el.querySelector("span").textContent = sub || "";
    box.appendChild(el);
    while (box.children.length > 3) box.firstChild.remove();
    setTimeout(() => { el.classList.add("out"); setTimeout(() => el.remove(), 300); }, ms);
  }

  async function post(name, args) {
    try {
      const r = await fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, args }) });
      return { status: r.status, body: await r.json().catch(() => ({})) };
    } catch (e) { return { status: 0, body: { detail: "offline" } }; }
  }

  const robots = () => S.fleet?.robots || [];
  const deckOf = (z) => (S.map?.decks || []).reduce((a, d) => (!a || Math.abs(d.z - z) < Math.abs(a.z - z) ? d : a), null)?.name;
  function mover() {
    const rs = robots().filter((r) => r.state === "TRANSIT" && (r.path || []).length > 1);
    const r = rs.find((x) => x.deck === S.deck) || rs[0] || robots()[0] || { id: "R1" };
    if (r.deck) setDeck(r.deck);
    return r.id;
  }
  const drop = (kind, title) => () => ["drop_obstacle", { kind, at: mover() }, title];
  const ACTS = {
    large: drop("barrel", "Large can rolling"),
    puddle: drop("puddle", "Puddle"),
    small: () => ["drop_can", { size: "small", at: mover() }, "Small can"],
    crate: drop("crate", "Crate"),
    spill: drop("spill", "Oil spill"),
    debris: drop("debris", "Debris"),
    scaffold: drop("scaffold", "Scaffold"),
    hose: drop("hose", "Hose"),
    stairs: () => ["close_stairs", {}, "Stairs closed"],
    walkway: () => ["close_walkway", {}, "Walkway closed"],
    clear: () => ["clear", { id: "all" }, "All clear"],
  };

  async function press(btn, name, args, title, after) {
    navigator.vibrate?.(12);
    btn.classList.add("busy");
    const { status, body } = await post(name, args);
    btn.classList.remove("busy");
    if (status === 429) return toast("warn", "One change per second", "Give the planner a moment");
    if (status !== 200) return toast("stop", "Not sent", pretty(body.detail || "server unreachable"));
    after?.();
    const log = body.fleet?.log || [];
    const m = log.find((e) => e.kind !== "reroute");
    if (m?.hazard) S.follow = m.hazard;
    const [head, ...rest] = pretty(m?.message || "").split(" → ");
    toast(name === "clear" || name === "set_weather" ? "ok" : "warn", title, rest.join(" → ") || head || "Sent to the sim");
    log.filter((e) => e.kind === "reroute").forEach(showEvent);
    S.dirty = true;
  }

  document.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
    const [name, args, title] = ACTS[b.dataset.act]();
    press(b, name, args, title);
  }));
  const weather = (btn, level, time) => press(btn, "set_weather", { level, time }, `${NAMES[time]} · ${NAMES[level] || level}`, () => { S.level = level; S.time = time; paintWeather(); });
  document.querySelectorAll("[data-level]").forEach((b) => b.addEventListener("click", () => weather(b, b.dataset.level, S.time)));
  document.querySelectorAll("[data-time]").forEach((b) => b.addEventListener("click", () => weather(b, S.level, b.dataset.time)));
  function paintWeather() {
    document.querySelectorAll("[data-level]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.level === S.level));
    document.querySelectorAll("[data-time]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.time === S.time));
    $("wx").textContent = `${NAMES[S.time]} · ${NAMES[S.level] || S.level}`;
  }

  function setDeck(d) {
    S.deck = d; S.dirty = true;
    document.querySelectorAll("[data-deck]").forEach((b) => b.setAttribute("aria-selected", b.dataset.deck === d));
  }
  document.querySelectorAll("[data-deck]").forEach((b) => b.addEventListener("click", () => setDeck(b.dataset.deck)));

  function showEvent(e) {
    const kind = SHOW[e.kind];
    if (!kind || !S.seen) return;
    const key = e.t + e.kind + e.message;
    if (S.seen.has(key)) return;
    S.seen.add(key);
    const now = performance.now(), slot = e.kind + (e.robot || "");
    if (now - (S.last[slot] || 0) < 2500) return;
    S.last[slot] = now;
    if (e.kind === "reroute" && e.new_path?.length) {
      const sign = (v) => (v >= 0 ? "+" : "") + Math.round(v);
      return toast("warn", `${e.robot} re-routing${e.via ? " via " + pretty(e.via) : ""} ${sign(e.extra_m)} m`, `Around ${pretty(e.cause)} · ${sign(e.extra_s)} s`);
    }
    const [head, tail] = pretty(e.message).split(" → ");
    toast(kind, head, tail || "");
  }

  function onFleet(f) {
    S.fleet = f; S.dirty = true;
    const fo = S.follow && (f.hazards?.obstacles || []).find((o) => o.id === S.follow);
    if (fo) { S.follow = null; setDeck(deckOf(fo.xyz[2])); }
    const evs = f.recent_events || [];
    if (!S.seen) S.seen = new Set(evs.map((e) => e.t + e.kind + e.message));
    else evs.forEach(showEvent);
    const w = f.hazards?.weather;
    if (w?.level) S.level = w.level;
    if (w?.time) S.time = w.time;
    paintWeather();
    if (!S.deck && robots().length && S.map) setDeck(robots().find((r) => r.state === "TRANSIT")?.deck || "top_deck");
    $("map-t").textContent = `t ${Math.round(f.t || 0)} s · ${(f.hazards?.obstacles || []).length} on deck`;
  }
  let poll = null;
  function connect() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/fleet");
    ws.onmessage = (ev) => { try { onFleet(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onopen = () => { clearInterval(poll); poll = null; };
    ws.onclose = () => {
      if (!poll) poll = setInterval(() => fetch("/api/fleet").then((r) => r.json()).then(onFleet).catch(() => {}), 2000);
      setTimeout(connect, 3000);
    };
  }

  const STYLE = { block: ["rgba(255,159,67,.4)", "#FF9F43"], slow: ["rgba(255,159,67,.25)", "#FFB86B"], slip: ["rgba(150,120,255,.3)", "#A08CFF"], soft: ["rgba(90,160,255,.3)", "#5AA0FF"] };
  const cv = $("map"), g = cv.getContext("2d");
  function draw() {
    requestAnimationFrame(draw);
    if (!S.map || !S.dirty) return;
    S.dirty = false;
    const dpr = devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight;
    if (cv.width !== Math.round(W * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const b = S.map.bounds, pad = 10, k = Math.min((W - 2 * pad) / (b.xmax - b.xmin), (H - 2 * pad) / (b.ymax - b.ymin));
    const ox = (W - k * (b.xmax - b.xmin)) / 2, oy = (H - k * (b.ymax - b.ymin)) / 2;
    const X = (x) => ox + (x - b.xmin) * k, Y = (y) => H - oy - (y - b.ymin) * k;
    const on = (p) => deckOf(p[2]) === S.deck;
    const seg = (a, c, color, w, dash) => { g.strokeStyle = color; g.lineWidth = w; g.setLineDash(dash || []); g.beginPath(); g.moveTo(X(a[0]), Y(a[1])); g.lineTo(X(c[0]), Y(c[1])); g.stroke(); g.setLineDash([]); };
    const poly = (pts, color, w, dash) => { for (let i = 1; i < (pts || []).length; i++) if (on(pts[i - 1]) || on(pts[i])) seg(pts[i - 1], pts[i], color, w, dash); };
    g.lineCap = "round";
    S.map.walkways.forEach((c) => {
      if (c.deck === S.deck || (c.kind === "stairs" && (on(c.a) || on(c.b))))
        seg(c.a, c.b, c.kind === "stairs" ? "rgba(227,179,65,.55)" : "rgba(245,244,238,.22)", c.kind === "walk" ? 5 : 3.5, c.kind === "stairs" ? [2, 4] : null);
    });
    const hz = S.fleet?.hazards || {};
    (hz.closed || []).forEach((c) => on(c.a) && seg(c.a, c.b, "rgba(248,81,73,.95)", 3, [5, 4]));
    robots().forEach((r) => { if (r.prev_path) poly(r.prev_path, "rgba(220,224,230,.45)", 2, [3, 4]); poly(r.path, "#D97757", 2.6); });
    (hz.obstacles || []).forEach((o) => {
      if (!on(o.xyz)) return;
      const x = X(o.xyz[0]), y = Y(o.xyz[1]), r = Math.max(4.5, o.radius * k);
      if (o.kind === "barrel") {
        g.fillStyle = "rgba(248,81,73,.28)"; g.strokeStyle = "#F85149"; g.lineWidth = 1.8;
        g.beginPath(); g.arc(x, y, r, 0, 7); g.fill(); g.stroke();
        const v = o.vel || [0, 0], s = Math.hypot(v[0], v[1]);
        if (o.rolling && s > 0.05) { const L = 10 + 10 * s, ux = v[0] / s, uy = -v[1] / s; g.beginPath(); g.moveTo(x, y); g.lineTo(x + ux * L, y + uy * L); g.stroke(); g.beginPath(); g.arc(x + ux * L, y + uy * L, 2.4, 0, 7); g.fillStyle = "#F85149"; g.fill(); }
        return;
      }
      const [fill, line] = STYLE[o.effect] || STYLE.block;
      g.fillStyle = fill; g.strokeStyle = line; g.lineWidth = 1.5; g.beginPath();
      if (o.effect === "soft" || o.effect === "slip") g.ellipse(x, y, r * 1.2, r * 0.85, 0.3, 0, 7); else g.rect(x - r, y - r, 2 * r, 2 * r);
      g.fill(); g.stroke();
    });
    g.font = "600 11px -apple-system, system-ui, sans-serif";
    robots().forEach((r) => {
      const p = r.pose, here = deckOf(p.z) === S.deck, x = X(p.x), y = Y(p.y);
      g.globalAlpha = here ? 1 : 0.28;
      g.fillStyle = r.state === "BLOCKED" || r.state === "FAILED" ? "#F85149" : r.state === "WORK" ? "#3FB950" : "#F5F4EE";
      g.beginPath(); g.arc(x, y, here ? 5.5 : 4, 0, 7); g.fill();
      g.strokeStyle = "rgba(38,38,36,.9)"; g.lineWidth = 2; g.stroke();
      if (here) { g.fillStyle = "#F5F4EE"; g.fillText(r.id, x + 8, y - 6); }
      g.globalAlpha = 1;
    });
  }
  addEventListener("resize", () => (S.dirty = true));

  const img = $("frame");
  function frame() {
    const im = new Image();
    im.onload = () => { img.src = im.src; img.classList.add("on"); S.frameAt = Date.now(); };
    im.src = "/api/frame.jpg?t=" + Date.now();
  }
  async function health() {
    try {
      const h = await (await fetch("/api/health")).json(), p = $("health");
      p.dataset.s = h.state === "live" ? "live" : h.state === "no_sim" ? "off" : "warn";
      $("health-t").textContent = h.state === "live" ? `Live · ${Math.round(h.fps)} fps` : h.state === "no_sim" ? "Sim offline" : h.state === "warming" ? "Warming up" : "Render paused";
    } catch (e) { $("health").dataset.s = "off"; $("health-t").textContent = "Offline"; }
    if (S.frameAt) $("frame-age").textContent = Math.round((Date.now() - S.frameAt) / 1000) + " s ago";
  }

  fetch("/api/fleet/map").then((r) => r.json()).then((m) => { S.map = m; if (S.fleet) onFleet(S.fleet); else setDeck("top_deck"); }).catch(() => {});
  connect();
  paintWeather();
  frame(); setInterval(frame, 2000);
  health(); setInterval(health, 4000);
  requestAnimationFrame(draw);
})();
