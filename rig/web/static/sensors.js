(() => {
  // Fixed fire-and-gas detectors around the rig (GD-xx). Input, any of:
  //   state.detectors = [{id, type, position:[x,y,z], reading, unit:"LEL", threshold, state:"ok|warn|alarm", dispatched?}]
  //   state.sensors as an array of the same (state.sensors as an object stays the per-robot sensor map)
  //   state.fleet.detectors / state.fleet.sensors (array) from fleet_state.v1
  // reading is a fraction of LEL (0.22 = 22 % LEL); the map uses position x, y in the fleet_map.v1 frame.
  const H = HUD, { S, q, esc, now } = H, B = H.brand;
  const RANK = { alarm: 2, warn: 1, ok: 0 };
  const dog = (id) => { if (!id) return ""; const m = String(id).match(/(\d+)$/); return /^(go2|unit|dog|[qr])-?\d/i.test(id) && m ? "Dog-" + m[1] : H.rname({ id }); };
  const stateOf = (d) => (RANK[d.state] != null ? d.state : d.reading >= d.threshold ? "alarm" : d.reading >= d.threshold / 2 ? "warn" : "ok");
  const lel = (v) => (v == null || !Number.isFinite(+v) ? "—" : (+v).toFixed(2));
  const norm = (list) => list.filter((d) => d && d.id).map((d) => {
    const p = Array.isArray(d.position) ? d.position : Array.isArray(d.xyz) ? d.xyz : null;
    const x = { ...d, position: p, reading: +(d.reading ?? d.lel ?? 0), threshold: +(d.threshold ?? 0.2), unit: d.unit || "LEL" };
    x.state = stateOf(x);
    return x;
  });
  const pick = (h) => (Array.isArray(h.detectors) ? h.detectors : Array.isArray(h.sensors) ? h.sensors
    : Array.isArray(h.fleet?.detectors) ? h.fleet.detectors : Array.isArray(h.fleet?.sensors) ? h.fleet.sensors : null);
  const seen = new Map(), alarmed = new Set();
  H.on((h) => {
    const list = pick(h);
    if (!list) return;
    S.detectors = norm(list);
    S.detectors.forEach((d) => {
      const was = seen.get(d.id) || { state: "ok", sent: null };
      const sent = d.dispatched || null;
      if (d.state === "alarm" && was.state !== "alarm") {
        alarmed.add(d.id);
        const to = sent ? " → " + dog(sent) + " dispatched" : "";
        H.addLog({ t: S.t, from: "Detector", to: "GAS_ALARM", reason: `${d.id} alarm ${lel(d.reading)} ${d.unit}${to}` });
        if (!S.seeking) H.banner("stop", `${d.id} · ${Math.round(d.reading * 100)} % LEL`, sent ? dog(sent) + " dispatched" : "Fixed detector in alarm", "wind", 4200);
        S.detAt = now();
      } else if (sent && sent !== was.sent && (alarmed.has(d.id) || d.state !== "ok")) {
        H.addLog({ t: S.t, from: "Detector", to: "DISPATCH", reason: `${d.id} ${lel(d.reading)} ${d.unit} → ${dog(sent)} dispatched` });
      }
      seen.set(d.id, { state: d.state, sent: sent || was.sent });
    });
  });
  const sorted = (S) => [...(S.detectors || [])].sort((a, b) => RANK[b.state] - RANK[a.state] || b.reading - a.reading || String(a.id).localeCompare(b.id));
  const worst = (S) => sorted(S)[0];

  // Map layer: called from H.drawRig before the robots are drawn.
  const deckOf = (z) => { const ds = S.fleetMap?.decks; if (!ds?.length || z == null) return null; return ds.reduce((a, b) => (Math.abs(b.z - z) < Math.abs(a.z - z) ? b : a)).name; };
  H.drawDetectors = function (g, m, S, c) {
    const ds = (S.detectors || []).filter((d) => d.position && Number.isFinite(+d.position[0]) && Number.isFinite(+d.position[1]));
    const pts = [], t = now(), s = m.big ? 5 : 3.5;
    ds.forEach((d) => {
      const x = m.X(+d.position[0]), y = m.Y(+d.position[1]), dk = deckOf(d.position[2]), off = m.deck !== "all" && dk && dk !== m.deck;
      g.globalAlpha = off ? 0.35 : 1;
      const col = d.state === "alarm" ? B.stop : d.state === "warn" ? B.warn : B.muted;
      if (d.state === "alarm") {
        const u = (t * 1.1) % 1;
        g.strokeStyle = B.stop; g.lineWidth = 1.6; g.globalAlpha = (off ? 0.35 : 1) * (1 - u);
        g.beginPath(); g.arc(x, y, s + 3 + u * (m.big ? 16 : 10), 0, 7); g.stroke();
        g.globalAlpha = off ? 0.35 : 1;
        g.beginPath(); g.arc(x, y, s + 3, 0, 7); g.stroke();
      }
      g.fillStyle = col; g.strokeStyle = B.bg; g.lineWidth = 1.2;
      g.beginPath(); g.roundRect(x - s, y - s, 2 * s, 2 * s, 1.5); g.fill(); g.stroke();
      g.fillStyle = B.bg; g.beginPath(); g.arc(x, y, m.big ? 1.6 : 1.1, 0, 7); g.fill();
      const crowded = d.state === "ok" && ds.some((o) => o !== d && Math.hypot(m.X(+o.position[0]) - x, m.Y(+o.position[1]) - y) < 40);
      if ((m.big && !crowded) || d.state !== "ok") {
        g.font = `500 ${m.big ? 10 : 9}px "JetBrains Mono", ui-monospace, monospace`;
        g.fillStyle = d.state === "ok" ? "rgba(154,163,174,.75)" : col;
        g.fillText(d.id + (d.state !== "ok" && m.big ? " " + lel(d.reading) : ""), x + s + 3, y + s + 7);
      }
      g.globalAlpha = 1;
      pts.push({ d, x, y });
    });
    if (c) {
      c._dets = pts;
      if (!c._dh) {
        c._dh = 1;
        c.addEventListener("mousemove", (e) => {
          const r = c.getBoundingClientRect(), hit = (c._dets || []).find((p) => Math.hypot(p.x - e.clientX + r.left, p.y - e.clientY + r.top) < 10);
          const tip = hit ? `${hit.d.id} · ${String(hit.d.type || "detector").replace(/_/g, " ")} · ${lel(hit.d.reading)} ${hit.d.unit} (alarm ${lel(hit.d.threshold)}) · ${hit.d.state}` : "";
          if (c.title !== tip) c.title = tip;
          c.style.cursor = hit ? "help" : "";
        });
      }
    }
    return pts;
  };

  const row = (d) => `<div class="det-row ${d.state}"><i class="det-ic"></i><b>${esc(d.id)}</b><span class="dim">${esc(d.state === "ok" ? "" : d.dispatched ? "→ " + dog(d.dispatched) : d.state)}</span><em class="tnum">${lel(d.reading)}</em><small class="tnum">/ ${lel(d.threshold)}</small></div>`;
  H.panel({ id: "detectors", title: "Fixed detectors", meta: "LEL", icon: "wind", key: "d", edge: "left", keys: "fixed sensors detectors gas fire GD lel alarm",
    priority: (S) => (worst(S)?.state === "alarm" ? 85 : worst(S)?.state === "warn" ? 40 : 0),
    phases: (S) => worst(S)?.state === "alarm" || (S.detAt != null && now() - S.detAt < 10 / S.speed),
    desc: "The fire-and-gas detectors fixed around the rig, live reading in LEL fraction. Alarms sort to the top; an alarm dispatches a dog.",
    hint: (S) => { const w = worst(S); return !w ? "" : w.state === "ok" ? (S.detectors.length + " ok") : `${w.id} ${lel(w.reading)}`; },
    render(el, S) {
      const x = q(el, `<div class="det-list" data-k="rows"></div>`), ds = sorted(S);
      const html = ds.length ? ds.map(row).join("") : `<p class="none">No fixed detectors reporting</p>`;
      if (x.rows._h !== html) { x.rows._h = html; x.rows.innerHTML = html; }
    } });
})();
