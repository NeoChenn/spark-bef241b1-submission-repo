(() => {
  const H = HUD, S = H.S, Q = new URLSearchParams(location.search);
  const LIVE = !(Q.has("mock") || Q.get("replay")) || Q.has("fleet");
  const CORAL = "#2EE6D6", RED = "rgba(255,77,79,.95)", DRAW_S = 0.9, GHOST_S = 9;
  const SHOW = new Set(["reroute", "needs_human", "drop_obstacle", "drop_can", "dispatch", "close_walkway", "close_stairs", "clear", "set_weather", "reopen", "blocked"]);
  const anim = {}, seen = new Set();

  const deckOf = (z) => {
    const ds = S.fleetMap?.decks || [];
    const d = ds.reduce((a, b) => (Math.abs(b.z - z) < Math.abs(a.z - z) ? b : a), ds[0] || { z: 0 });
    return d && Math.abs(d.z - z) < 1.5 ? d.name : null;
  };
  const onDeck = (p, deck) => Array.isArray(p) && (deck === "all" || deckOf(p[2]) === deck);

  function line(g, o, pts, deck, color, width, dash, frac) {
    const segs = [];
    for (let i = 1; i < (pts || []).length; i++) if (onDeck(pts[i - 1], deck) && onDeck(pts[i], deck)) segs.push([pts[i - 1], pts[i]]);
    const L = (a, b) => Math.hypot(o.X(b[0]) - o.X(a[0]), o.Y(b[1]) - o.Y(a[1]));
    let left = frac == null ? Infinity : segs.reduce((s, [a, b]) => s + L(a, b), 0) * frac;
    g.strokeStyle = color; g.lineWidth = width; g.setLineDash(dash || []); g.lineCap = "round";
    g.beginPath();
    for (const [a, b] of segs) {
      if (left <= 0) break;
      const u = Math.min(1, left / (L(a, b) || 1));
      g.moveTo(o.X(a[0]), o.Y(a[1])); g.lineTo(o.X(a[0] + (b[0] - a[0]) * u), o.Y(a[1] + (b[1] - a[1]) * u));
      left -= L(a, b);
    }
    g.stroke(); g.setLineDash([]);
  }

  H.drawRoutes = (g, o, S) => {
    const f = S.fleet, hz = f?.hazards;
    if (!f) return;
    const t = H.now(), wide = o.big ? 1 : 0.7;
    (hz?.closed || []).forEach((c) => line(g, o, [c.a, c.b], o.deck, RED, 2.4 * wide, [5, 4]));
    (hz?.obstacles || []).forEach((ob) => {
      if (!onDeck(ob.xyz, o.deck)) return;
      const x = o.X(ob.xyz[0]), y = o.Y(ob.xyz[1]), r = Math.max(4, (ob.radius || 0.5) * o.k);
      g.fillStyle = ob.effect === "slip" ? "rgba(90,160,255,.35)" : "rgba(255,176,32,.4)";
      g.strokeStyle = ob.effect === "slip" ? "#5AA0FF" : "#FFB020"; g.lineWidth = 1.5;
      g.beginPath(); g.rect(x - r, y - r, 2 * r, 2 * r); g.fill(); g.stroke();
      if (o.big) { g.fillStyle = "rgba(255,255,255,.85)"; g.font = "500 11px Inter, system-ui, sans-serif"; g.fillText(ob.kind, x + r + 4, y + 4); }
    });
    (f.robots || []).filter((r) => r?.id).forEach((r) => {
      const a = anim[r.id] || (anim[r.id] = {});
      if (r.prev_path && a.k !== r.reroute_t) { a.k = r.reroute_t; a.t0 = t; a.prev = r.prev_path; }
      const age = a.t0 ? t - a.t0 : Infinity;
      if (a.prev && r.prev_path) line(g, o, a.prev, o.deck, `rgba(220,224,230,${Math.max(0.22, 0.75 - (0.5 * age) / GHOST_S)})`, 2 * wide, [3, 4]);
      line(g, o, r.path, o.deck, H.rcol ? H.rcol(r) : CORAL, (age < GHOST_S ? 3 : 2.2) * wide, null, Math.min(1, age / DRAW_S));
    });
    const w = hz?.weather;
    if (o.big && w && w.level !== "calm") {
      g.font = "600 12px Inter, system-ui, sans-serif"; g.fillStyle = w.closes_outer ? RED : "rgba(255,176,32,.95)";
      g.fillText([String(w.level || "").toUpperCase(), w.wind != null ? `wind ${(+w.wind).toFixed(1)} → ${Math.round(w.wind_dir || 0)}°` : "", w.speed?.walk != null ? `walk ×${(+w.speed.walk).toFixed(2)}` : "", w.perception_range_m != null ? `sight ${Math.round(w.perception_range_m)} m` : ""].filter(Boolean).join(" · "), 12, 18);
    }
  };

  function onState(snap) {
    H.apply({ fleet: snap });
    if (snap.hazards?.weather) S.weather = { ...(S.weather || {}), ...snap.hazards.weather };
    (Array.isArray(snap.recent_events) ? snap.recent_events : []).filter((e) => e && SHOW.has(e.kind)).forEach((e) => {
      e = { ...e, message: String(e.message || e.kind) };
      const key = e.t + e.kind + e.message;
      if (seen.has(key)) return;
      seen.add(key);
      H.addLog({ t: e.t, from: "Planner", to: e.kind === "reroute" ? "RE-ROUTE" : e.kind.toUpperCase(), reason: e.message });
      if (e.kind === "reroute" && e.new_path?.length) H.banner("warn", `${H.rname ? H.rname({ id: e.robot }).split(" ·")[0] : e.robot} re-routes`, `${e.cause} · ${e.extra_m >= 0 ? "+" : ""}${Math.round(e.extra_m)} m, ${e.extra_s >= 0 ? "+" : ""}${Math.round(e.extra_s)} s`, "map", 3500);
      if (e.kind === "needs_human") H.banner("stop", "Cut off", e.message.split(" -> ")[0], "warn", 3500);
      if (e.kind === "dispatch") H.banner("ok", "Can pick-up", e.message, "map", 3500);
    });
  }

  function connect() {
    const ws = new WebSocket((location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/fleet");
    ws.onmessage = (ev) => { try { onState(JSON.parse(ev.data)); } catch (e) { console.warn(e); } };
    ws.onclose = () => setTimeout(connect, 2500);
  }
  if (LIVE) connect();

  const send = (name, args, why) => {
    H.post(name, args).then((r) => H.addLog({ t: S.t, from: "Judge", to: name.toUpperCase(), reason: r?.fleet?.log?.[0]?.message || r?.detail || why }));
  };
  const C = (id, label, ic, name, args, desc, key) => H.command({ id: "nav." + id, group: "Unscripted change", label, icon: ic, key, keys: "nav route reroute obstacle block path planner " + name, desc, run: () => send(name, args, label) });
  C("crate", "Drop a crate on R1's route", "block", "drop_obstacle", { kind: "crate", at: "R1" }, "A crate lands on the walkway ahead of R1. The planner closes that segment and R1 re-routes, via the stairs if it must.");
  C("debris", "Drop debris on R2's route", "block", "drop_obstacle", { kind: "debris", at: "R2" }, "Debris blocks the walkway ahead of R2; R2 re-plans around it.");
  C("scaffold", "Scaffold tower on R3's route", "block", "drop_obstacle", { kind: "scaffold", at: "R3" }, "A scaffold tower goes up on R3's walkway; R3 re-plans.");
  C("can", "Drop a can on R1's walkway", "block", "drop_can", { at: "R1" }, "A can lands on the walkway ahead of R1. The nearest robot with a gripper is dispatched to pick it up; the others re-route until it is cleared.");
  C("hose", "Hose across R1's route", "block", "drop_obstacle", { kind: "hose", at: "R1" }, "A hose across the grating: passable but slow, so the planner weighs a detour.");
  C("spill", "Oil spill on the west stairs", "warn", "drop_obstacle", { kind: "spill", at: "stairs_west_upper" }, "Slippery stairs are closed; flat walkways nearby are slowed.");
  C("stairs", "Close the west stairs", "block", "close_stairs", { id: "stairs_west_upper" }, "Robots on the stairs back out to the landing; the rest re-route.");
  C("catwalk", "Close the middle-deck catwalk", "block", "close_walkway", { edge: "catwalk_mid" }, "The narrow cross catwalk on the middle deck is shut.");
  C("worker", "Worker on the top-deck walkway", "person", "person_at", { xy: [-1, 4], deck: "top_deck", duration: 90 }, "Robots keep 3 m clear and route around the worker for 90 s.");
  C("clear", "Clear obstacles and closures", "x", "clear", { id: "all" }, "Removes every obstacle and closure; routes and cut-off jobs come back.");
  C("barrel", "Roll a barrel at R1", "wind", "drop_obstacle", { kind: "barrel", at: "R1" }, "An oil drum breaks loose ahead of R1 and rolls with the wind. Its blocked segments move with it; robots re-plan until it comes to rest.");
  const PAD = "https://rig.zeecloud.org/interactive";
  let qrLib = null;
  const qr = (el, size) => {
    if (el._qr) return;
    el._qr = 1;
    qrLib = qrLib || new Promise((ok) => { const sc = document.createElement("script"); sc.src = "https://cdnjs.cloudflare.com/ajax/libs/qrcodejs/1.0.0/qrcode.min.js"; sc.onload = ok; document.head.appendChild(sc); });
    qrLib.then(() => new QRCode(el, { text: PAD, width: size, height: size, colorDark: "#1d1a18", colorLight: "#F5F4EE", correctLevel: QRCode.CorrectLevel.M }));
  };
  const card = (el, size) => {
    if (el._card) return;
    el._card = 1;
    el.innerHTML = `<div class="qr-card"><div class="qr-code"></div><div><b>Interactive controls</b><p>Scan to change the weather, roll a barrel or block a walkway from your phone.</p><a href="/interactive" target="_blank" rel="noopener">rig.zeecloud.org/interactive</a></div></div>`;
    qr(el.querySelector(".qr-code"), size);
  };
  let pad = null;
  H.command({ id: "interactive", group: "Unscripted change", label: "Interactive controls", icon: "spark", key: "⇧J", keys: "judge pad phone qr mobile interactive weather barrel obstacle",
    desc: "Opens rig.zeecloud.org/interactive on a phone: weather, rolling barrel, obstacles and incidents, with the planner's reply.",
    preview: (el) => card(el, 132), active: () => !!pad,
    run: () => {
      if (pad) { pad.remove(); pad = null; return; }
      pad = document.createElement("div");
      pad.className = "qr-pop glass hl";
      card(pad, 200);
      pad.onclick = (e) => { if (e.target.tagName !== "A") { pad.remove(); pad = null; } };
      document.body.appendChild(pad);
    } });
})();
