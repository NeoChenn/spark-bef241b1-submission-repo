(() => {
  // Demo-video layer for Mission Control: EDGE / CLOUD badge, scenario chapters, key-number callouts, policy and Gemini
  // tags, "Judge pressed" toasts for unscripted moments, and a clean recording mode (?record=1).
  // Reads only what the state feed already carries; optional fields that make it exact (see the PR):
  //   state.scenario | state.chapter            1..3 or "gas_leak" / "deck_patrol" / "pipe_repair"
  //   robot.policy = {skill, kind: "trained"|"scripted"} or robot.skill   which learned policy drives the robot now
  //   robot.advisor = "robotics-er" | "argon"   Gemini is advising this robot right now
  //   events[] entries with source: "judge" (server.py tags phone-pad presses this way)
  const H = HUD, S = H.S, Q = new URLSearchParams(location.search), B = document.body, esc = H.esc, now = H.now;
  const REC = Q.has("record") && Q.get("record") !== "0";
  const dogs = (s) => (H.dogs ? H.dogs(s) : s);
  B.classList.toggle("record", REC);
  const add = (id, html) => { const d = document.createElement("div"); d.id = id; d.innerHTML = html; B.appendChild(d); return d; };
  const top = add("topright", `<div id="chap-pill" hidden><span class="tnum"></span><b></b></div><div id="netbadge" data-mode=""><i></i><b></b><span class="tnum"></span></div>`);
  const pill = top.firstChild, net = top.lastChild;
  const big = add("chapter", `<span class="tnum"></span><b></b>`);
  const call = add("callout", `<small></small><b class="tnum"></b>`);
  const toast = add("toast", `<span class="uns">Unscripted</span><b></b>`);
  const line = add("logline", "");
  const until = { chapter: 0, callout: 0, toast: 0 };

  // ---- 2. policy and Gemini tags ------------------------------------------------------------------------------
  const WALK = /^(TRANSIT|PATROL|APPROACH|LEAK_APPROACH|TO_DOCK|RETURN|WALK|GAS_CHECK|GAS_SEEK|SEEK|REROUTE|RE_ROUTE|GOTO|MOVE)/;
  const SCRUB = /^(SCRUB|BACK_OFF)/, PICK = /^(PICK|GRASP|LIFT|PLACE|CAN)/, SEE = /^(DETECT|SECOND_VIEW|VERIFY|VERIFY_PAINT|HOLD|INSPECT)/;
  const stOf = (r) => String(r?.job_state || r?.state || "").toUpperCase();
  const taskOf = (r) => String(r?.task?.kind || r?.task || r?.fleet?.task || "").toLowerCase();
  H.skillOf = (r) => {
    const p = r.policy ?? r.fleet?.policy;
    if (p) return typeof p === "string" ? p : p.kind === "scripted" ? null : p.skill || null;
    if (r.skill) return r.skill;
    const st = stOf(r);
    if (SCRUB.test(st)) return "scrubbing";
    if (PICK.test(st) || (/pick_can|^can/.test(taskOf(r)) && /WORK|SETUP/.test(st))) return "can pick-up";
    if (WALK.test(st)) return "walking";
    return null;
  };
  const cut = () => !!(S.linkCut || S.control?.link === "CUT" || S.control?.link === "DOWN" || S.control?.online === false);
  const cloud = () => !cut() && S.control?.mode === "CLOUD";
  const same = (a, b) => a != null && b != null && dogs(H.rname({ id: a })) === dogs(H.rname({ id: b }));
  H.gemOf = (r) => {
    if (cut()) return [];
    const out = [], adv = String(r.advisor || r.fleet?.advisor || "").toLowerCase();
    if (/er|robotics/.test(adv) || (S.gemini && now() - S.gemAt < 9 / S.speed && SEE.test(stOf(r)))) out.push("Gemini Robotics-ER");
    if (/argon|advisor/.test(adv) || (cloud() && (S.control?.actions || []).some((a) => a && a.source !== "edge" && same(a.robot, r.id)))) out.push("Gemini advisor");
    return out;
  };
  H.tags = (r) => {
    if (!r) return "";
    const sk = H.skillOf(r), col = H.rcol ? H.rcol(r) : "";
    return (sk ? `<span class="tag pol" style="--c:${col}">Trained policy · ${esc(sk)}</span>` : "") + H.gemOf(r).map((g) => `<span class="tag gem">${esc(g)}</span>`).join("");
  };

  // ---- 3. EDGE / CLOUD badge --------------------------------------------------------------------------------
  let wasCloud = null;
  function badge() {
    const c = S.control, isCloud = cloud(), mode = isCloud ? "cloud" : "edge";
    if (net.dataset.mode !== mode) net.dataset.mode = mode;
    H.set(net.children[1], isCloud ? "CLOUD" : "EDGE ONLY");
    const lat = typeof c?.latency_ms === "number" ? ` · ${(c.latency_ms / 1000).toFixed(1)} s` : "";
    H.set(net.children[2], isCloud ? "Argon" + lat : cut() ? "Gemini link lost" : c ? "edge rules" : "no Argon feed");
    if (wasCloud === true && !isCloud) {
      H.addLog({ t: S.t, from: "Edge", to: "EDGE", reason: "Gemini link lost → fleet continues on edge rules" });
      if (!S.seeking) H.banner("warn", "Gemini link lost", "Fleet continues on edge rules", "link", 3200);
    } else if (wasCloud === false && isCloud && S.started) H.addLog({ t: S.t, from: "Edge", to: "CLOUD", reason: "Gemini link restored → Argon advising again" });
    if (c || S.linkCut) wasCloud = isCloud;
  }

  // ---- 5. scenario chapters ---------------------------------------------------------------------------------
  const CH = { 1: "Gas leak", 2: "Deck patrol", 3: "Pipe repair" };
  const parseCh = (v) => {
    if (v == null || v === "") return null;
    const n = +v;
    if (n >= 1 && n <= 3) return n;
    const s = String(v).toLowerCase();
    return /gas|leak|hot.?work/.test(s) ? 1 : /patrol|housekeep|deck|can/.test(s) ? 2 : /pipe|repair|rust|weld|scrub|paint/.test(s) ? 3 : null;
  };
  let chapter = null, manual = parseCh(Q.get("chapter")), cand = null, candAt = 0, field = null;
  H.on((h) => { const f = parseCh(h.scenario ?? h.chapter ?? h.mission?.scenario); if (f) field = f; });
  const GAS_ST = /GAS|LEAK|RETORQUE|VALVE|ISOLATE/, REPAIR_ST = /^(DETECT|SECOND_VIEW|APPROACH|SPRAY|SOAK|SCRUB|BACK_OFF|VERIFY|PAINT|VERIFY_PAINT|WELD|CRACK|HOLD|WIPE_LENS|WAIT_LULL)/;
  function derive() {
    if (field) return field;
    const bs = H.bots ? H.bots(S) : [];
    if (S.gasAlarm || (S.detectors || []).some((d) => d.state === "alarm") || bs.some((r) => GAS_ST.test(stOf(r)))) return 1;
    if (REPAIR_ST.test(String(S.state || "")) || bs.some((r) => REPAIR_ST.test(stOf(r)) || /rust|weld|crack/.test(taskOf(r)) && /WORK|SETUP/.test(stOf(r)))) return 3;
    if (bs.length) return 2;
    return null;
  }
  function setChapter(n, why) {
    if (!CH[n] || n === chapter) return;
    chapter = n;
    const t = `${n}/3`;
    H.set(big.children[0], t); H.set(big.children[1], CH[n]); H.set(pill.children[0], t); H.set(pill.children[1], CH[n]);
    pill.hidden = false;
    big.classList.remove("on"); big.offsetWidth; big.classList.add("on");
    until.chapter = now() + 2.5;
    H.addLog({ t: S.t, from: why ? "Operator" : "Mission", to: "CHAPTER", reason: `${t} · ${CH[n]}` });
  }
  function chapters() {
    if (manual) setChapter(manual, "operator");
    else if (S.started) {
      const d = derive();
      if (d !== cand) { cand = d; candAt = now(); }
      if (d && now() - candAt > 1.5 / S.speed) setChapter(d);
    }
    if (big.classList.contains("on") && now() > until.chapter) big.classList.remove("on");
  }

  // ---- 6. key-number callouts -------------------------------------------------------------------------------
  const queue = [];
  function callout(label, value, tone = "act") { if (!S.seeking) queue.push({ label, value, tone }); }
  function callouts() {
    if (call.classList.contains("on") && now() > until.callout) call.classList.remove("on");
    if (call.classList.contains("on") || !queue.length) return;
    const c = queue.shift();
    H.set(call.children[0], c.label); H.set(call.children[1], c.value); call.dataset.tone = c.tone;
    call.classList.add("on"); until.callout = now() + 3;
  }
  H.callout = callout;
  const detSeen = new Map();
  let gasWas = false, srcShown = false, scrub = null, lastT = null;
  function watch() {
    // After a seek or a jump in sim time, take the current state as the baseline instead of calling it all out.
    const jump = lastT == null || Math.abs((S.t || 0) - lastT) > 2;
    lastT = S.t || 0;
    if (jump) {
      (S.detectors || []).forEach((d) => detSeen.set(d.id, d.state));
      gasWas = S.gasAlarm || (S.detectors || []).some((d) => d.state === "alarm");
      return;
    }
    (S.detectors || []).forEach((d) => {
      if (d.state === "alarm" && detSeen.get(d.id) !== "alarm") callout("Fixed detector alarm", `${d.id} · ${(+d.reading).toFixed(2)} ${d.unit || "LEL"}`, "stop");
      detSeen.set(d.id, d.state);
    });
    const detAlarm = (S.detectors || []).some((d) => d.state === "alarm");
    if (S.gasAlarm && !gasWas && !detAlarm) {
      // Gas alarm raised by the gas feed (10 % LEL) before any fixed detector crosses its own threshold: call out the
      // highest reading, preferring a fixed detector (GD-xx) over the generic gas sensor list.
      const hi = (xs, v) => xs.reduce((a, b) => (v(b) > (a ? v(a) : -1) ? b : a), null);
      const d = hi(S.detectors || [], (x) => +x.reading || 0), s = hi(S.gas?.sensors || [], (x) => +x.lel || 0);
      if (d && +d.reading >= 0.05) callout("Gas alarm · fixed detector", `${d.id} · ${(+d.reading).toFixed(2)} ${d.unit || "LEL"}`, "stop");
      else if (s) callout("Gas alarm", `${s.id} · ${(+s.lel).toFixed(2)} LEL`, "stop");
    }
    const alarm = S.gasAlarm || detAlarm;
    if (!alarm && gasWas) srcShown = false;
    gasWas = alarm;
    const sig = S.gas?.source?.sigma_m ?? S.gas?.belief?.sigma_m;
    if (!srcShown && Number.isFinite(+sig) && sig != null && +sig <= 1.0) { srcShown = true; callout("Leak localised", `Source ±${(+sig).toFixed(1)} m`, "act"); }
    const bs = H.bots ? H.bots(S) : [], r = bs.find((x) => SCRUB.test(stOf(x))) || (SCRUB.test(String(S.state || "")) ? { id: null } : null);
    if (!r) { scrub = null; return; }
    const st = String(r.id ? stOf(r) : S.state);
    if (!scrub || scrub.id !== r.id || (st === "SCRUB" && scrub.st === "BACK_OFF")) scrub = { id: r.id, at: now(), shown: false };
    scrub.st = st;
    if (scrub.shown || st !== "SCRUB" || now() - scrub.at < 3 / S.speed) return;
    const ser = (r.id && S.series[r.id + ".force_n"]) || S.series.force || [], t0 = ser.length ? ser.at(-1)[0] - 3 : 0;
    const pts = ser.filter((p) => p[0] >= t0 && p[1] > 0);
    if (!pts.length) return;
    scrub.shown = true;
    const pk = Math.max(...pts.map((p) => p[1]));
    callout("Brush force · force shield", pk < 40 ? `${pk.toFixed(0)} N < 40 N` : `${pk.toFixed(0)} N > 40 N · backing off`, pk < 40 ? "ok" : "stop");
  }
  H.on((h) => (h.events || []).forEach((e) => {
    const m = /LEAK_FOUND/.test(e?.to || "") && /([\d.]+) m from the flange/.exec(e.reason || "");
    if (m && !srcShown) { srcShown = true; callout("Leak localised", `Source ±${(+m[1]).toFixed(1)} m`, "act"); }
  }));

  // ---- 4. unscripted moments --------------------------------------------------------------------------------
  const EV = { drop_obstacle: "Drop an obstacle", drop_can: "Drop a can", close_walkway: "Close a walkway", close_stairs: "Close the stairs", clear: "Clear changes",
    set_weather: "Weather", gas_leak: "Gas leak", cut_link: "Cut the Gemini link", person_at: "Worker on deck", battery_fault: "Battery fault", spawn_patch: "New rust patch",
    force_spike: "Weld bead", set_glare: "Sun glare", occlude_camera: "Lens mist", reopen: "Reopen" };
  const evLabel = (e) => {
    const a = e.args || {}, base = EV[e.event] || H.nice(String(e.event || e.to || ""));
    return a.kind ? `${base} · ${a.kind}` : a.level ? `${base} · ${a.level}` : e.event === "cut_link" && a.on === false ? "Restore the Gemini link" : base;
  };
  let lastPress = { at: -9, name: null };
  H.unscripted = (label, name) => {
    lastPress = { at: now(), name };
    H.set(toast.lastChild, "Judge pressed: " + dogs(label));
    toast.classList.remove("on"); toast.offsetWidth; toast.classList.add("on");
    B.classList.add("unscripted"); until.toast = now() + 4;
    H.addLog({ t: S.t, from: "Judge", to: "UNSCRIPTED", reason: "Judge pressed: " + dogs(label) });
  };
  const born = now();
  let logN = S.logN;
  function scanLog() {
    const n = Math.min(S.logN - logN, S.log.length);
    logN = S.logN;
    if (n <= 0 || now() - born < 2) return;
    S.log.slice(0, n).reverse().forEach((e) => {
      if (e?.source !== "judge") return;
      if (now() - lastPress.at < 1.5 && lastPress.name === e.event) return;
      H.unscripted(evLabel(e), e.event);
    });
  }

  // ---- 8. recording mode: one-line log, keys ----------------------------------------------------------------
  function logline() {
    if (!REC) return;
    const e = S.log[0], html = e ? `<span class="tnum">${Number(e.t || 0).toFixed(1)}</span><b>${esc(dogs(e.from || ""))}</b><em>${esc(H.nice(e.to))}</em><span>${esc(dogs(String(e.reason || "").replace(/->/g, "→")))}</span>` : "";
    if (line._h !== html) { line._h = html; line.innerHTML = html; }
  }
  if (REC) {
    addEventListener("keydown", (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || /INPUT|TEXTAREA/.test(e.target.tagName)) return;
      if (/^[0-3]$/.test(e.key)) {
        e.preventDefault(); e.stopImmediatePropagation();
        manual = +e.key || null;
        if (!manual) { cand = null; chapter = null; }
      }
    }, true);
    addEventListener("DOMContentLoaded", () => setTimeout(() => H.pinOnce?.("robot"), 50));
  }
  H.command({ id: "record", group: "Demo", label: "Recording mode", icon: "film", keys: "record video take clean chapter callout",
    desc: "Clean take for the demo video: logo, EDGE / CLOUD badge, chapter label, robot view, tags, callouts and a one-line log. Keys 1/2/3 set the chapter, 0 goes back to automatic.",
    active: () => REC, run: () => { const u = new URL(location.href); REC ? u.searchParams.delete("record") : u.searchParams.set("record", "1"); location.href = u.toString(); } });

  function loop() {
    try {
      badge(); scanLog(); watch(); chapters(); callouts(); logline();
      if (now() > until.toast && toast.classList.contains("on")) { toast.classList.remove("on"); B.classList.remove("unscripted"); }
    } catch (err) { console.warn("video layer", err); }
    requestAnimationFrame(loop);
  }
  addEventListener("DOMContentLoaded", () => requestAnimationFrame(loop));
})();
