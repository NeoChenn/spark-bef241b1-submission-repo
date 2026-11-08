(() => {
  const H = HUD, { S, esc } = H;
  const VERDICT = { LEAK_CONFIRMED: ["stop", "Leak confirmed"], PIPE_SENSOR_FAULT: ["warn", "Sensor fault"], LEAK_UNSEEN_BY_FIXED: ["stop", "Leak unseen by fixed"], UNVERIFIED: ["warn", "Unverified"], OK: ["ok", "OK"] };
  const CMD = { hold_for_crew: ["warn", "Hold for crew", "hand"], reroute: ["warn", "Reroute", "block"], go_check: ["act", "Go check", "target"], fix_leak: ["act", "Fix leak", "bolt"], retreat: ["stop", "Retreat", "warn"], resume_rust: ["ok", "Resume rust", "check"], log_sensor_fault: ["warn", "Log sensor fault", "tag"], hold: ["warn", "Hold", "hand"] };
  const WHO = { unit1: "Unit 1", unit2: "Unit 2", supv: "Supervisor" };
  const who = (id) => WHO[id] || (S.robots.find((r) => r.id === id)?.name || id || "").split(" ·")[0];
  const aim = (c) => (c.finding === "OBJECT_ON_WALKWAY" || c.kind === "hold_for_crew" ? "object on walkway" : c.target ? "→ " + c.target : "");
  const lel = (v) => (v == null ? "—" : (+v).toFixed(1));
  const ledOf = (v) => (v == null ? "grey" : v >= 10 ? "red" : v >= 5 ? "amber" : "green");
  Object.assign(S, { findings: {}, commands: [], tickets: [], argonCalls: [] });
  function row(r, quiet) {
    if (r.type === "finding") {
      S.findings[r.sensor] = r;
      if (!quiet) H.addLog({ t: r.t, from: "Argon fusion", to: r.kind, reason: `${r.sensor} · ${r.reason}` });
    } else if (r.type === "command") {
      S.commands.unshift(r); S.commands.length = Math.min(S.commands.length, 8);
      const [tone, label, ic] = CMD[r.kind] || ["act", r.kind, "bolt"], why = String(r.reason || "").split(" | ")[0];
      relog(r, r.kind.toUpperCase(), r.kind.toUpperCase(), `${who(r.robot)} · ${aim(r)}${r.source === "edge" ? " (edge rule)" : ""} · ${r.reason}`, r.source === "edge" ? "Edge rule" : "Argon");
      H.banner(tone === "act" ? "run" : tone, `${r.source === "edge" ? "Edge rule" : "Argon"} · ${label} · ${who(r.robot)}`, why, ic, 3600);
    } else if (r.type === "ticket") {
      S.tickets.push(r);
      if (!quiet) H.addLog({ t: r.t, from: "Argon", to: "TICKET", reason: `${r.id} · ${r.sensor} · ${r.reason || "maintenance"}` });
    } else if (r.type === "vetoed") {
      S.commands.unshift({ ...r, vetoed: true });
      const label = (CMD[r.kind] || [, r.kind])[1], reason = `${label} for ${who(r.robot)} refused · ${r.why || ""}`;
      const e = S.log.find((x) => +x.t === +r.t && x.to === String(r.kind).toUpperCase());
      if (e) { Object.assign(e, { from: "Argon", to: "VETOED", reason }); S.logN++; } else H.addLog({ t: r.t, from: "Argon", to: "VETOED", reason });
      H.banner("warn", `Refused · Argon's ${label.toLowerCase()}`, r.why || "safety envelope", "block", 3600);
    } else if (r.type === "argon_call") { S.argonCall = r; S.argonCalls.push(r); }
    else if (r.type === "scenario") scene(r, quiet);
    else if (r.type === "object_report") relog(r, "OBJECT_REPORT", "OBJECT_REPORT", `${who(r.robot)} lidar: ${r.object === "box" ? "crate" : r.object} ${(+r.dist_m).toFixed(1)} m ahead · stopped and reported`, "Object report");
    else if (r.type === "nav_event") nav(r);
  }
  function relog(r, match, to, reason, from) {
    const e = S.log.find((x) => +x.t === +r.t && x.to === match && !x._re);
    if (e) { Object.assign(e, { to, reason, from: from || e.from, _re: 1 }); S.logN++; } else H.addLog({ t: r.t, from: from || "Scenario", to, reason, _re: 1 });
  }
  function scene(r, quiet) {
    const what = r.kind || r.object || "object";
    if (r.name === "object_dropped") {
      S.crate = { xyz: r.xyz, size: r.size_m, kind: what };
      relog(r, "CRATE", "CRATE_DROPPED", `${(r.size_m?.[0] ?? 0.5)} m ${what} dropped on ${who(r.robot)}'s walkway`);
      H.banner("warn", `A ${what} on the walkway`, `Dropped in ${who(r.robot)}'s path`, "block", 3200);
    } else if (r.name === "object_cleared") {
      S.crate = null;
      relog(r, "CRATE", "CRATE_CLEARED", `Deck crew cleared the ${what}`);
      H.banner("ok", `${what[0].toUpperCase() + what.slice(1)} cleared`, `${who(r.robot)} can walk on`, "check", 2800);
    } else if (r.name === "push") {
      S.pushAt = +r.t;
      relog(r, "PUSH", "PUSH", `${who(r.robot)} shoved ${(+r.offset_m).toFixed(1)} m sideways`);
      H.banner("warn", `Push · ${who(r.robot)}`, `${(+r.offset_m).toFixed(1)} m lateral shove`, "hand", 3000);
    } else if (!quiet) H.addLog({ t: r.t, from: "Scenario", to: String(r.name).toUpperCase(), reason: r.flange || r.detector || "" });
  }
  function nav(r) {
    const lat = r.lateral_m != null ? Math.abs(+r.lateral_m).toFixed(2) + " m" : "";
    if (r.event === "object_on_walkway") H.banner("warn", `${who(r.robot)} · lidar stop`, `Object ${(+r.ahead_m).toFixed(1)} m ahead · reporting it`, "eye", 3000);
    else if (r.event === "cleared") H.addLog({ t: r.t, from: "Nav", to: "WALKS_ON", reason: `${who(r.robot)} path clear, walking on` });
    else if (r.event === "off_path" && S.pushAt != null && +r.t - S.pushAt < 8) H.addLog({ t: r.t, from: "Nav", to: "OFF_PATH", reason: `${who(r.robot)} ${lat} off its path` });
    else if (r.event === "recovered" && S.pushAt != null && +r.t - S.pushAt < 8) {
      H.addLog({ t: r.t, from: "Nav", to: "BACK_ON_PATH", reason: `${who(r.robot)} back within ${lat} of its path` });
      H.banner("ok", `${who(r.robot)} back on its path`, `Recovered ${(+r.t - S.pushAt).toFixed(1)} s after the push`, "check", 3000);
    }
  }
  const NAV = (r) => { const n = r.nav || {}, lid = n.lidar_min, lat = Math.abs(n.lateral_m || 0);
    return `<div class="nv"><b>${esc(who(r.id))}</b><span class="dim">${esc(H.nice(n.state || ""))}</span><em class="tnum">${n.vx != null ? (+n.vx).toFixed(2) + " m/s" : ""}</em>`
      + `<span>Lidar</span><i><s style="width:${lid == null ? 0 : Math.min(100, (lid / 2) * 100)}%;background:${lid != null && lid < 0.6 ? "var(--warn)" : ""}"></s></i><em class="tnum">${lid == null ? "—" : (+lid).toFixed(2) + " m"}</em>`
      + `<span>Off path</span><i><s style="width:${Math.min(100, (lat / 0.5) * 100)}%;background:${lat > 0.25 ? "var(--warn)" : ""}"></s><u></u></i><em class="tnum">${lat.toFixed(2)} m</em></div>`; };
  H.panel({ id: "nav", title: "Navigation", meta: "lidar · path", icon: "map", key: "j", edge: "right", priority: 75, keys: "lidar obstacle crate push path lateral walkway",
    phases: (S) => S.robots.some((r) => r.nav && (r.nav.state === "REPORT" || Math.abs(r.nav.lateral_m || 0) > 0.25)),
    desc: "Each robot's walking state, nearest lidar return and how far it is off its planned path (0.25 m mark).", hint: (S) => (S.crate ? "Object on the walkway" : ""),
    render(el, S) {
      const html = S.robots.filter((r) => r.nav).map(NAV).join("") || `<p class="none">No navigation data in this run</p>`;
      if (el._h !== html) { el._h = html; el.innerHTML = html; }
    } });
  H.on((h) => {
    if (h.argon) S.argon = h.argon;
    if (h.detectors) S.detectors = h.detectors;
    if (Array.isArray(h.fusion)) S.fusionNow = Object.fromEntries(h.fusion.map((f) => [f.sensor, f]));
    (h.argon_rows || []).forEach((r) => row(r, !!h.events?.length));
    if (h.robots && !h.fleet && h.robots.some((r) => r.world_pos)) S.fleet = { schema: "fleet_state.v1", t: h.t, zones: S.fleet?.zones || [], tasks: S.fleet?.tasks || [],
      robots: h.robots.filter((r) => r.world_pos).map((r) => ({ id: r.id, name: r.name, state: r.state, job_state: r.state, deck: "top_deck", pose: { x: r.world_pos[0], y: r.world_pos[1], z: r.world_pos[2], yaw: r.yaw || 0 } })) };
  });
  const live = () => !!(S.argon || S.detectors || S.fusionNow || Object.keys(S.findings).length);
  H.fusionHot = (S) => {
    if (!live()) return 0;
    const v = Object.values({ ...(S.argon?.verdicts || {}), ...Object.fromEntries(Object.entries(S.fusionNow || S.findings).map(([k, f]) => [k, f.kind])) });
    return v.some((k) => k !== "OK") || S.argon?.pending || S.argon?.held ? 92 : 40;
  };
  H.fusionHint = (S) => {
    const c = S.commands[0] || S.argon?.last_command;
    return c ? `${(CMD[c.kind] || [, c.kind])[1]} · ${who(c.robot)}` : live() ? "Watching" : "";
  };
  function sensors(S) {
    const ids = new Set([...(S.detectors || []).map((d) => d.id), ...Object.keys(S.argon?.verdicts || {}), ...Object.keys(S.fusionNow || {}), ...Object.keys(S.findings)]);
    const quiet = [], rows = [...ids].filter((id) => id !== "zone").map((id) => {
      const d = (S.detectors || []).find((x) => x.id === id), f = S.fusionNow?.[id] || S.findings[id], kind = f?.kind || S.argon?.verdicts?.[id];
      if (ids.size > 3 && (kind || "OK") === "OK" && (d?.led || "green") === "green") { quiet.push(id); return ""; }
      const fixed = d ? d.lel : f?.fixed_lel, [vt, vl] = VERDICT[kind] || ["", kind ? kind.replace(/_/g, " ").toLowerCase() : "—"];
      const rob = f?.robot_lel != null ? `${who(f.robot)} ${lel(f.robot_lel)}${f.robot_dist != null ? " @ " + (+f.robot_dist).toFixed(1) + " m" : ""}` : "no sniffer near";
      return `<div class="fu-row"><i class="led ${esc(d?.led || ledOf(fixed))}"></i><b>${esc(id)}${d?.flange || f?.flange ? ` <em>${esc(d?.flange || f?.flange)}</em>` : ""}</b>${kind ? `<span class="vchip ${vt}">${esc(vl)}</span>` : "<span></span>"}<small class="tnum">fixed ${lel(fixed)} <em>vs</em> ${esc(rob)}</small></div>`;
    }).join("");
    return rows + (quiet.length ? `<p class="fu-quiet"><i class="led green"></i>${esc(quiet.join(", "))} · OK</p>` : "");
  }
  const cmdRow = (c) => { const [tone, label] = CMD[c.kind] || ["act", c.kind]; const [sit, why] = String(c.reason || "").split(" | ");
    return `<li class="${c.vetoed ? "vetoed" : ""}"><span class="tnum">${c.t != null ? (+c.t).toFixed(1) : ""}</span><b class="${c.vetoed ? "stop" : tone}">${c.vetoed ? `<s>${esc(label)}</s> refused` : esc(label)}</b><span>${esc(who(c.robot))} ${esc(aim(c))}<small>${c.vetoed ? "Edge veto · " + esc(c.why || "") : `${c.source === "edge" ? "Edge rule" : "Argon " + Math.round((c.confidence ?? 1) * 100)} · ${esc(why || sit)}`}</small></span></li>`; };
  H.fusion = (S) => {
    if (!live()) return "";
    const a = S.argon || {}, cmds = S.commands.length ? S.commands : a.last_command ? [a.last_command] : [];
    const meta = `<div class="cc-meta"><span class="badge ${a.link === "CUT" ? "" : "cloud"}">${a.link === "CUT" ? "Edge" : "Argon"}</span><span>${esc(a.pending ? "thinking…" : a.status || "")}</span>${a.latency_ms != null ? `<span class="tnum">${(a.latency_ms / 1000).toFixed(1)} s</span>` : ""}${a.cost_usd != null ? `<span class="tnum">$${(+a.cost_usd).toFixed(4)}</span>` : ""}${a.held ? `<span class="hold">Unit 1 held</span>` : ""}</div>`;
    const spark = a.spark?.length ? `<p class="edge">Spark rule · ${esc(a.spark.join(", "))} within 6 m of the Scrubber</p>` : "";
    const tk = [...(a.tickets || []), ...S.tickets.filter((t) => !(a.tickets || []).some((x) => x.startsWith(t.id))).map((t) => t.id + " " + t.sensor)];
    return meta + spark + `<h4 class="fu-h">Fusion · fixed vs robot sniffer <em>% LEL</em></h4>${sensors(S) || `<p class="none">No detectors reporting</p>`}`
      + (cmds.length ? `<h4 class="fu-h">Commands</h4><ol class="fu-cmd">${cmds.slice(0, 3).map(cmdRow).join("")}</ol>` : "")
      + (tk.length ? `<p class="fu-tk">${tk.map((t) => `<kbd>${esc(t)}</kbd>`).join(" ")}</p>` : "")
      + (S.intent && !S.control ? `<p class="intent">“${esc(S.intent)}”</p>` : "");
  };
  H.command({ id: "order", group: "Control centre", label: "Give Argon an order…", icon: "spark", key: "⇧A", keys: "order intent argon command centre instruct",
    desc: "Sends a plain-words order to Argon as an intent event. Argon still answers to the safety envelope and never approves its own SUPV jobs.",
    score: () => (live() ? 0.3 : 0), run: () => setTimeout(() => H.ask("Give Argon an order", "Send the Fixer to check flange F1", (text) => { H.post("intent", { text }); S.intent = text; H.addLog({ t: S.t, from: "Operator", to: "INTENT", reason: text }); }), 0) });
})();
