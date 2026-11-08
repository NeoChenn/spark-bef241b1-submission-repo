(() => {
  const H = HUD, { S, esc } = H;
  const VERDICT = { LEAK_CONFIRMED: ["stop", "Leak confirmed"], PIPE_SENSOR_FAULT: ["warn", "Sensor fault"], LEAK_UNSEEN_BY_FIXED: ["stop", "Leak unseen by fixed"], UNVERIFIED: ["warn", "Unverified"], OK: ["ok", "OK"] };
  const CMD = { reroute: ["warn", "Reroute", "block"], go_check: ["act", "Go check", "target"], fix_leak: ["act", "Fix leak", "bolt"], retreat: ["stop", "Retreat", "warn"], resume_rust: ["ok", "Resume rust", "check"], log_sensor_fault: ["warn", "Log sensor fault", "tag"], hold: ["warn", "Hold", "hand"] };
  const WHO = { unit1: "Unit 1", unit2: "Unit 2", supv: "Supervisor" };
  const who = (id) => WHO[id] || (S.robots.find((r) => r.id === id)?.name || id || "").split(" ·")[0];
  const lel = (v) => (v == null ? "—" : (+v).toFixed(1));
  const ledOf = (v) => (v == null ? "grey" : v >= 10 ? "red" : v >= 5 ? "amber" : "green");
  Object.assign(S, { findings: {}, commands: [], tickets: [] });
  function row(r, quiet) {
    if (r.type === "finding") {
      S.findings[r.sensor] = r;
      if (!quiet) H.addLog({ t: r.t, from: "Argon fusion", to: r.kind, reason: `${r.sensor} · ${r.reason}` });
    } else if (r.type === "command") {
      S.commands.unshift(r); S.commands.length = Math.min(S.commands.length, 8);
      const [tone, label, ic] = CMD[r.kind] || ["act", r.kind, "bolt"], why = String(r.reason || "").split(" | ")[0];
      if (!quiet) H.addLog({ t: r.t, from: r.source === "edge" ? "Edge rule" : "Argon", to: r.kind.toUpperCase(), reason: `${who(r.robot)}${r.target ? " → " + r.target : ""} · ${r.reason}` });
      H.banner(tone === "act" ? "run" : tone, `Argon · ${label} · ${who(r.robot)}`, why, ic, 3600);
    } else if (r.type === "ticket") {
      S.tickets.push(r);
      if (!quiet) H.addLog({ t: r.t, from: "Argon", to: "TICKET", reason: `${r.id} · ${r.sensor} · ${r.reason || "maintenance"}` });
    } else if (r.type === "vetoed") {
      if (!quiet) H.addLog({ t: r.t, from: "Argon", to: "VETOED", reason: `${r.kind} ${who(r.robot)} · ${r.why || ""}` });
    } else if (r.type === "argon_call") S.argonCall = r;
  }
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
    return [...ids].filter((id) => id !== "zone").map((id) => {
      const d = (S.detectors || []).find((x) => x.id === id), f = S.fusionNow?.[id] || S.findings[id], kind = f?.kind || S.argon?.verdicts?.[id];
      const fixed = d ? d.lel : f?.fixed_lel, [vt, vl] = VERDICT[kind] || ["", kind ? kind.replace(/_/g, " ").toLowerCase() : "—"];
      const rob = f?.robot_lel != null ? `${who(f.robot)} ${lel(f.robot_lel)}${f.robot_dist != null ? " @ " + (+f.robot_dist).toFixed(1) + " m" : ""}` : "no sniffer near";
      return `<div class="fu-row"><i class="led ${esc(d?.led || ledOf(fixed))}"></i><b>${esc(id)}${d?.flange || f?.flange ? ` <em>${esc(d?.flange || f?.flange)}</em>` : ""}</b>${kind ? `<span class="vchip ${vt}">${esc(vl)}</span>` : "<span></span>"}<small class="tnum">fixed ${lel(fixed)} <em>vs</em> ${esc(rob)}</small></div>`;
    }).join("");
  }
  const cmdRow = (c) => { const [tone, label] = CMD[c.kind] || ["act", c.kind]; const [sit, why] = String(c.reason || "").split(" | ");
    return `<li><span class="tnum">${c.t != null ? (+c.t).toFixed(1) : ""}</span><b class="${tone}">${esc(label)}</b><span>${esc(who(c.robot))}${c.target ? " → " + esc(c.target) : ""}<small>${c.source === "edge" ? "Edge rule" : "Argon " + Math.round((c.confidence ?? 1) * 100)} · ${esc(why || sit)}</small></span></li>`; };
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
