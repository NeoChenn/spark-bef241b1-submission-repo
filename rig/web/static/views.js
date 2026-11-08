(() => {
  const H = HUD, { S, q, set, esc, spark, now } = H;
  const short = (id) => H.rname({ id }).split(" ·")[0];
  const fleetBots = () => (S.robots.length ? S.robots : (S.fleet?.robots || []).map((r) => ({ ...r, name: H.rname(r) })));
  const bot = () => fleetBots().find((r) => r.id === S.focus) || fleetBots()[0];
  const chips = (cur) => fleetBots().map((r) => `<button data-id="${esc(r.id)}" class="${r.id === cur ? "on" : ""}">${esc(r.name || H.rname(r))}</button>`).join("");
  const onFocus = (el) => { if (!el._f) { el._f = 1; el.addEventListener("click", (e) => { const b = e.target.closest("[data-id]"); if (b) H.focus(b.dataset.id); }); } };
  const last = (k) => S.series[k]?.at(-1)?.[1];
  const DECKS = [["auto", "Busiest deck"], ["top_deck", "Top"], ["middle_deck", "Middle"], ["lower_deck", "Lower"], ["all", "All decks"]];
  H.view({ id: "map", title: "Mission map", icon: "map", key: "⇧M", keys: "top down rig decks walkways plume people dock no-fly fleet",
    desc: "Top-down rig with decks as layers: robots with heading and status, walkways, jobs, gas plume, people and docks. Click a robot to focus it.",
    preview: (el) => { const x = q(el, `<canvas class="rig big" data-k="c"></canvas>`); if (now() - (el._t || 0) > 0.1) { el._t = now(); H.drawRig(x.c, S, { big: true }); } },
    render(el) {
      const x = q(el, `<div class="mm"><div class="seg" data-k="decks"></div><canvas data-k="c"></canvas></div><aside class="side" data-k="side"></aside>`);
      if (!x.c._b) {
        x.c._b = 1;
        x.c.onclick = (e) => { const r = x.c.getBoundingClientRect(), hit = (x.c._hits || []).find((p) => Math.hypot(p.x - e.clientX + r.left, p.y - e.clientY + r.top) < 18); if (hit) H.focus(hit.id); };
        x.decks.onclick = (e) => { const b = e.target.closest("button"); if (b) S.deck = b.dataset.d; };
        onFocus(x.side);
      }
      if (now() - (el._t || 0) > 0.066) { el._t = now(); x.c._hits = H.drawRig(x.c, S, { big: true, deck: S.deck }); }
      const d = `${DECKS.map(([k, l]) => `<button data-d="${k}" class="${(S.deck || "auto") === k ? "on" : ""}">${l}</button>`).join("")}`;
      if (x.decks._h !== d) { x.decks._h = d; x.decks.innerHTML = d; }
      const rows = (S.fleet?.robots || []).map((r) => `<div class="mrow ${r.id === S.focus ? "on" : ""}" data-id="${esc(r.id)}"><i style="background:${H.COL[H.tone(r.job_state || r.state)]}"></i><b>${esc(H.rname(r))}</b><span>${esc(H.nice(r.job_state || r.state))}${r.task ? " · " + esc(r.task) : ""}</span><em class="tnum">${r.battery?.soc == null ? "" : Math.round(r.battery.soc * 100) + " %"}</em></div>`).join("")
        + `<div class="legend col"><span><i class="lg r"></i>robot · heading</span><span><i class="lg t"></i>drone</span><span><i class="lg g"></i>gas plume</span><span><i class="lg p"></i>person · 3 m</span><span><i class="lg d"></i>dock</span><span><i class="lg n"></i>no-fly zone</span></div>`;
      if (x.side._h !== rows) { x.side._h = rows; x.side.innerHTML = (S.fleet ? "" : `<p class="none">No fleet snapshot yet</p>`) + rows; }
    } });
  const KIND = { fpv: ["Camera", "cam"], rgb: ["Camera", "cam"], policy: ["Detector boxes", "target"], det: ["Detector boxes", "target"], ogi: ["OGI gas", "wind"], gas: ["OGI gas", "wind"], bev: ["Lidar BEV", "map"], lidar: ["Lidar BEV", "map"], depth: ["Depth", "layers"], thermal: ["Thermal", "sun"] };
  const ORDER = Object.keys(KIND);
  const kname = (k) => KIND[k]?.[0] || String(k).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
  H.on((h) => {
    if (!h.views || typeof h.views !== "object") return;
    const v = (S.views ||= {});
    Object.entries(h.views).forEach(([rid, ks]) => { if (ks && typeof ks === "object") v[rid] = { ...(v[rid] || {}), ...ks }; });
    const ks = [...new Set(Object.values(v).flatMap((x) => Object.keys(x || {})))].sort((a, b) => (ORDER.indexOf(a) + 99) % 99 - (ORDER.indexOf(b) + 99) % 99);
    if (ks.join() === S.vkinds) return;
    S.vkinds = ks.join();
    [...Array(8).keys()].forEach((n) => H.uncommand("pv." + n));
    ks.slice(0, 8).forEach((k, n) => H.command({ id: "pv." + n, group: "Views", label: "Policy view: " + kname(k), icon: KIND[k]?.[1] || "eye", keys: "perspective policy sees inset overlay " + k,
      desc: "Opens First person on what the focused robot's policy is conditioned on: " + kname(k).toLowerCase() + ".", active: () => S.kind === k && H.sheetId?.() === "fpv", run: () => { S.kind = k; H.openView("fpv"); } }));
  });
  const kindsOf = (id) => { const v = S.views?.[id] || {}; return Object.keys(v).filter((k) => v[k] && (v[k].jpg || v[k].seq != null || v[k].url)).sort((a, b) => (ORDER.indexOf(a) + 99) % 99 - (ORDER.indexOf(b) + 99) % 99); };
  const entry = (id, k) => S.views?.[id]?.[k];
  const srcOf = (id, k, e = entry(id, k)) => (!e ? null : e.jpg ? "data:image/jpeg;base64," + e.jpg : e.url || `/api/views/${encodeURIComponent(id)}/${encodeURIComponent(k)}.jpg?seq=${e.seq}`);
  const legacy = (r) => { const f = S.fpv?.[r?.id]; return f?.jpg ? f : r && r.id === fleetBots()[0]?.id && S.inspection?.jpg ? S.inspection : null; };
  H.viewSrc = (id) => { const ks = kindsOf(id); if (ks.length) return srcOf(id, ks.includes(S.kind) ? S.kind : ks[0]); const f = legacy({ id }); return f ? "data:image/jpeg;base64," + f.jpg : null; };
  const boxes = (e) => (Array.isArray(e?.boxes) && e.w ? e.boxes.map((b) => { const [x, y, w, h] = b.xywh || [b.x, b.y, b.w, b.h]; return `<rect x="${x}" y="${y}" width="${w}" height="${h}"/>${b.label ? `<text x="${x + 4}" y="${Math.max(12, y - 5)}">${esc(b.label)}${b.score != null ? " " + (+b.score).toFixed(2) : ""}</text>` : ""}`; }).join("") : "");
  H.view({ id: "fpv", title: "First person", icon: "cam", key: "⇧F", keys: "camera robot pov perception policy detector boxes ogi gas lidar bev mask overlay",
    desc: "The focused robot's own sensors and what its policy is conditioned on: camera, detector boxes, OGI gas false colour and lidar BEV. Switch robot and view at the top.",
    render(el) {
      const r = bot(), id = r?.id, ks = kindsOf(id), k = ks.includes(S.kind) ? S.kind : ks[0], e = k ? entry(id, k) : null, f = e ? null : legacy(r);
      const x = q(el, `<div class="fp-bar"><div class="seg" data-k="who"></div><div class="seg" data-k="kinds"></div></div><div class="fp"><div class="fp-img"><img data-k="img" alt=""><svg data-k="pts"></svg><span class="fp-cap" data-k="cap"></span><p class="none" data-k="none"></p></div><aside class="side" data-k="side"></aside></div>`);
      onFocus(x.who);
      if (!x.kinds._b) { x.kinds._b = 1; x.kinds.onclick = (ev) => { const b = ev.target.closest("[data-kind]"); if (b) S.kind = b.dataset.kind; }; }
      const c = chips(id);
      if (x.who._h !== c) { x.who._h = c; x.who.innerHTML = c; }
      const kc = ks.length > 1 ? ks.map((kk) => `<button data-kind="${esc(kk)}" class="${kk === k ? "on" : ""}">${esc(kname(kk))}</button>`).join("") : "";
      if (x.kinds._h !== kc) { x.kinds._h = kc; x.kinds.innerHTML = kc; }
      const src = e ? srcOf(id, k, e) : f ? "data:image/jpeg;base64," + f.jpg : null;
      if (src && x.img._s !== src) { x.img._s = src; x.img.src = src; }
      x.img.hidden = !src;
      set(x.none, src ? "" : `No camera feed from ${r ? r.name || H.rname(r) : "any robot"} yet`);
      set(x.cap, e ? [kname(k), e.caption].filter(Boolean).join(" · ") : f ? "Inspection camera" : "");
      const g = S.gemini, gk = f && g && f === S.inspection ? S.gkey : "", bk = e ? k + (e.seq ?? "") + JSON.stringify(e.boxes || "") : gk;
      if (x.pts._k !== bk) {
        x.pts._k = bk; const iw = (e || f)?.w || 1, ih = (e || f)?.h || 1;
        x.pts.setAttribute("viewBox", `0 0 ${iw} ${ih}`); x.pts.setAttribute("preserveAspectRatio", "xMidYMid meet");
        x.pts.style.setProperty("--sw", iw / 320);
        x.pts.innerHTML = e ? boxes(e) : gk ? (g.points || []).map(([a, b]) => `<circle cx="${a}" cy="${b}" r="${f.w / 40}" stroke-width="${f.w / 160}"/>`).join("") : "";
      }
      const gas = last(id + ".gas_lel") ?? (S.gasMax || null), lead = r && r.id === fleetBots()[0]?.id;
      const kv = [["Phase", r?.caption?.text || H.nice(r?.state) || "—"], ["Rust mask", f && f === S.inspection ? "outlined in red" : "—"], ["Gauge", f?.gauge ?? "—"],
        ["Gas", gas == null ? "—" : (gas * 100).toFixed(1) + " % LEL"], ["Brush force", last(id + ".force_n") != null ? last(id + ".force_n").toFixed(1) + " N" : lead ? S.force.toFixed(1) + " N" : "—"], ["Detector", lead && S.conf != null ? S.conf.toFixed(2) : "—"],
        ["Gemini", lead && g ? `${String(g.verdict || "").replace("_", " ")} · ${(g.confidence || 0).toFixed(2)}` : "—"]].filter(([kk, v]) => v !== "—" || ["Phase", "Gas"].includes(kk));
      const th = ks.length > 1 ? `<div class="fp-th">${ks.filter((kk) => kk !== k).map((kk) => `<button data-kind="${esc(kk)}"><img src="${srcOf(id, kk)}" alt=""><span>${esc(kname(kk))}</span></button>`).join("")}</div>` : "";
      const side = kv.map(([kk, v]) => `<div class="kv"><span>${kk}</span><b>${esc(v)}</b></div>`).join("") + th;
      if (x.side._h !== side) { x.side._h = side; x.side.innerHTML = side; if (!x.side._b) { x.side._b = 1; x.side.addEventListener("click", (ev) => { const b = ev.target.closest("[data-kind]"); if (b) S.kind = b.dataset.kind; }); } }
    } });
  H.panel({ id: "persp", title: "Policy view", icon: "eye", key: "j", keys: "perspective inset policy sees camera detector ogi lidar bev pip",
    desc: "A small inset of what the focused robot's policy sees. ⇧F opens it full size.", suggest: (S) => (S.vkinds ? 0.3 : 0), hint: () => (S.vkinds ? kname(kindsOf(bot()?.id)[0] || "") : ""),
    render(el) {
      const r = bot(), id = r?.id, ks = kindsOf(id), k = ks.includes(S.kind) ? S.kind : ks[0], src = k ? srcOf(id, k) : null;
      const x = q(el, `<div class="pip on"><img data-k="img" alt=""><span class="fp-cap" data-k="cap"></span><span class="pip-none" data-k="none"></span></div>`);
      if (src && x.img._s !== src) { x.img._s = src; x.img.src = src; }
      x.img.hidden = !src;
      set(x.cap, src ? `${r.name || H.rname(r)} · ${kname(k)}` : ""); set(x.none, src ? "" : "No policy views from the sim yet");
    } });
  const MET = [["gas_lel", "Gas", "% LEL", 100, 1, { min: 0, max: 30, color: "#F85149", refs: [[10, "rgba(227,179,65,.5)"]] }],
    ["thermal_c", "Thermal", "°C", 1, 1, { min: 0, max: 80, color: "rgba(255,255,255,.75)" }],
    ["force_n", "Contact force", "N", 1, 1, { min: 0, max: 70, ramp: H.fc, refs: [[40, "rgba(248,81,73,.6)"]] }],
    ["battery", "Battery", "%", 100, 0, { min: 0, max: 1, color: "#3FB950", refs: [[0.2, "rgba(227,179,65,.5)"]] }],
    ["reach", "IK reach", "%", 100, 0, { min: 0, max: 1, color: "#D97757" }]];
  H.view({ id: "sensors", title: "Sensors", icon: "wave", key: "⇧S", keys: "gas ppm thermal force battery reach sparkline per robot",
    desc: "Per robot: gas, thermal, contact force against the 40 N line, battery and IK reach, over the last 30 s.",
    render(el) {
      const ids = fleetBots().map((r) => r.id), head = `<div class="sg" style="--n:${MET.length}"><span></span>${MET.map((m) => `<span class="sh">${m[1]} <em>${m[2]}</em></span>`).join("")}${ids.map((id) =>
        `<b class="${id === S.focus ? "on" : ""}" data-id="${esc(id)}">${esc(short(id))}</b>${MET.map((m) => `<div class="cell"><b class="tnum" data-k="${id}.${m[0]}"></b><canvas data-k="${id}.${m[0]}C"></canvas></div>`).join("")}`).join("")}</div>`;
      const x = q(el, ids.length ? `<div class="sv">${head}<div class="pvs" data-k="pvs"></div></div>` : `<p class="none">No robots reporting yet</p>`);
      if (now() - (el._t || 0) < 0.1) return;
      el._t = now();
      if (x.pvs) {
        onFocus(x.pvs.parentNode);
        const rows = ids.filter((id) => kindsOf(id).length).map((id) => `<div class="pv-row"><b>${esc(short(id))}</b>${kindsOf(id).map((k) => `<figure><img src="${srcOf(id, k)}" alt=""><figcaption>${esc(kname(k))}${entry(id, k)?.caption ? ` · ${esc(entry(id, k).caption)}` : ""}</figcaption></figure>`).join("")}</div>`).join("");
        const html = rows ? `<h3>What each policy sees</h3>${rows}` : "";
        if (x.pvs._h !== html) { x.pvs._h = html; x.pvs.innerHTML = html; }
      }
      ids.forEach((id, n) => MET.forEach(([m, , , mul, d, o]) => {
        const k = id + "." + m, data = S.series[k] || (n === 0 && m === "force_n" ? S.series.force : null);
        const v = data?.at(-1)?.[1] ?? (m === "battery" ? (S.fleet?.robots || []).find((r) => r.id === id)?.battery?.soc : null);
        set(x[k], v == null ? "—" : (v * mul).toFixed(d)); spark(x[k + "C"], data || [], o);
      }));
    } });
  const GROUP = [["Running", /RUNNING|ASSIGNED|active|assigned/], ["Waiting approval", /WAITING/], ["Queued", /QUEUED|pending/], ["Done", /DONE|FINDING|done/], ["Crew and failed", /HUMAN|FAIL|escalat/]];
  const decided = new Map();
  H.decided = decided;
  function decide(job, ok) {
    decided.set(job, ok ? "QUEUED" : "REJECTED");
    H.event(ok ? "approve" : "reject", { job, id: job });
    H.addLog({ t: S.t, from: "Operator", to: ok ? "APPROVED" : "REJECTED", reason: job });
  }
  H.decide = decide;
  H.view({ id: "orch", title: "Orchestrator", icon: "inbox", key: "⇧O", keys: "jobs queue approvals inbox approve reject decisions why",
    desc: "The job queue by status, the approval inbox with Gemini's reasoning, and every decision with its why.",
    render(el) {
      const x = q(el, `<div class="orch"><section><h3>Job queue</h3><div data-k="jobs"></div></section><section><h3>Approval inbox</h3><div data-k="inbox"></div></section><section><h3>Decisions</h3><ol class="log long" data-k="log"></ol></section></div>`);
      if (!x.inbox._b) { x.inbox._b = 1; x.inbox.onclick = (e) => { const b = e.target.closest("[data-job]"); if (b) decide(b.dataset.job, b.dataset.a === "1"); }; }
      const o = S.orchestrator, jobs = (o?.jobs || (S.fleet?.tasks || []).map((t) => ({ id: t.id, kind: t.kind, target: t.poi, status: t.status, robot: t.robot, priority: t.priority }))).map((j) => (decided.has(j.id) ? { ...j, status: decided.get(j.id) } : j));
      const aps = (o?.approvals || []).filter((a) => a && !decided.has(a.job)), c = H.ctl(S), key = JSON.stringify(o) + decided.size + jobs.length + JSON.stringify(c?.agent) + JSON.stringify(c?.vetoed);
      if (x.jobs._k !== key) {
        x.jobs._k = key;
        x.jobs.innerHTML = GROUP.map(([g, re]) => { const js = jobs.filter((j) => re.test(j.status)); return js.length ? `<h4>${g} <em>${js.length}</em></h4>` + js.slice(0, 8).map((j) =>
          `<div class="job"><i class="p${Math.min(5, Math.round(j.priority || 3))}"></i><span>${esc((j.kind || "").replace(/_/g, " "))} <em>${esc(j.target || "")}</em></span><em>${j.robot ? esc(short(j.robot)) : ""}</em>${j.level ? `<kbd>${esc(j.level)}</kbd>` : ""}</div>`).join("") : ""; }).join("") || `<p class="none">No jobs yet</p>`;
        x.inbox.innerHTML = aps.map((a) => `<div class="ap"><b>${esc((a.kind || "").replace(/_/g, " "))} · ${esc(a.target || a.job)}</b><p>${esc(a.text)}</p><div><button data-job="${esc(a.job)}" data-a="1" class="yes">Approve</button><button data-job="${esc(a.job)}" data-a="0">Reject</button></div></div>`).join("") || `<p class="none">Nothing waiting for you</p>`;
        x.inbox.innerHTML += c?.agent || c?.vetoed?.length ? `<h3 class="sub3">Planner</h3>${H.planner(c.agent, 8)}${H.vetoes(c, 3)}` : "";
        const sit = Array.isArray(c?.situation) ? c.situation : c?.situation ? [c.situation] : [];
        x.inbox.innerHTML += c ? `<h3 class="sub3">Argon <em class="dim">${esc(c.mode === "CLOUD" ? "cloud" : "edge")}${c.model ? " · " + esc(String(c.model).replace(/ \(.*\)$/, "")) : ""}</em></h3><div class="cc-sum">${sit.slice(0, 3).map((l) => `<p>${esc(l)}</p>`).join("")}</div><ol class="cc-acts">${(c.actions || []).filter(Boolean).slice(0, 5).map((a, n) => `<li><span class="tnum">${a.rank || n + 1}</span><span>${esc(a.label)}<small>${a.source === "edge" ? "Edge rule" : "Argon"}${a.rationale ? " · " + esc(a.rationale) : ""}</small></span><b class="tnum">${a.source === "edge" ? "hard" : Math.round((a.confidence || 0) * 100)}</b></li>`).join("")}</ol>` : "";
      }
      if (x.log._n !== S.logN || !S.log.length) { x.log._n = S.logN; x.log.innerHTML = H.logRows(S, 16) || `<p class="none">No decisions yet · state changes and re-plans land here with their reasons</p>`; }
    } });
  H.on((h) => {
    const ap = h.orchestrator?.approvals;
    if (!ap) return;
    [...Array(6).keys()].forEach((n) => H.uncommand("ap." + n));
    ap.filter((a) => a && !decided.has(a.job)).slice(0, 6).forEach((a, n) => H.command({ id: "ap." + n, group: "Control centre", label: `Approve ${(a.kind || "").replace(/_/g, " ")} · ${a.target || a.job}`, icon: "check",
      keys: "approve approval supv inbox", desc: a.text, hint: "SUPV", score: () => 0.92, run: () => decide(a.job, true) }));
  });
  H.command({ id: "intent", group: "Control centre", label: "Tell Argon…", icon: "spark", key: "⇧T", keys: "intent instruct operator gemini planner order",
    desc: "Type an instruction in plain words, such as “inspect deck 2, isolate any leak, don't paint without me”. Gemini turns it into jobs; the safety envelope still applies.",
    hint: () => (S.intent || S.control?.agent?.intent ? "Has an intent" : ""), score: () => (S.control ? 0.25 : 0),
    run: () => setTimeout(() => H.ask("Tell Argon", "Inspect deck 2, isolate any leak, don't paint without me", tell), 0) });
  function tell(text) { S.intent = text; H.event("intent", { text }, () => H.addLog({ t: S.t, from: "Operator", to: "INTENT", reason: text })); H.banner("run", "Sent to Argon", text.length > 64 ? text.slice(0, 62) + "…" : text, "spark", 2600); }
  H.free((text) => ({ id: "tell", group: "Tell Argon", icon: "spark", label: `Tell Argon “${text}”`, desc: "Sends this to Gemini 4 Argon as an operator intent. It turns it into jobs through the coordinator; the edge safety envelope still applies and it cannot approve its own SUPV jobs.", run: () => tell(text) }));
  const T = {};
  const U = (id, label, ic, key, name, args, local, o = {}) => H.command({ id: "u." + id, group: "Unscripted change", label, icon: ic, key, keys: "unscripted inject judge change " + (o.keys || ""), desc: o.desc,
    active: o.tog ? () => !!T[id] : undefined, score: o.score, run: () => { if (o.tog) T[id] = !T[id]; if (o.pre) o.pre(); H.event(name, typeof args === "function" ? args() : args, local); } });
  const scrub = () => S.state === "SCRUB";
  U("rust", "New rust patch", "spark", "⇧1", "spawn_patch", {}, () => {
    if (S.rust_grid) { const [bi, bj] = S.brush || [20, 30]; S.rust_grid = S.rust_grid.map((row, i) => row.map((v, j) => { const d = Math.hypot((i - bi + 6) / 3.5, (((j - bj + 13 + 64) % 64) / 4) - 1.5); return Math.max(v, d < 1.6 ? 0.85 * (1 - d / 1.6) + 0.1 : 0); })); }
    H.judge("SPRAY", "RE-PLAN: new rust injected by judge -> spray it, extend soak", "SPRAY");
  }, { desc: "Injects a fresh patch next to the brush. The planner re-plans: spray it, then extend the soak.", score: () => (scrub() ? 0.5 : 0) });
  U("glare", "Sun glare", "sun", "⇧2", "set_glare", () => ({ on: !!T.glare }), () => { document.getElementById("flash").classList.toggle("on", !!T.glare); H.judge("DETECT", T.glare ? "glare on inspection view -> detector unsure" : "glare cleared"); }, { tog: 1, desc: "Floods the inspection view with glare, so the detector loses confidence." });
  U("mist", "Lens mist", "cloud", "⇧3", "occlude_camera", () => ({ on: !!T.mist }), () => { S.mist = !!T.mist; H.judge(T.mist ? "HOLD" : "VERIFY", T.mist ? "perception lost (lens mist) -> hold position" : "camera view restored", T.mist ? "HOLD" : "VERIFY"); }, { tog: 1, desc: "Fogs the inspection lens. The robot holds until it can see again.", score: () => (S.state === "VERIFY" ? 0.6 : 0) });
  U("bead", "Weld bead", "bolt", "⇧4", "force_spike", {}, () => { S.spike = S.force = 64.8; S.peak = Math.max(S.peak, 64.8); H.push("force", 64.8); setTimeout(() => (S.spike = 0), 400); H.judge("BACK_OFF", "contact force 65 N > 40 N limit -> retract", "BACK_OFF"); }, { desc: "The brush hits a weld bead at 65 N. The arm backs off and retries softer.", score: () => (scrub() ? 0.8 : 0) });
  U("worker", "Worker walks up", "person", "⇧5", "person_at", () => ({ distance: T.worker ? 2.4 : null }), () => H.judge(T.worker ? "PAUSE_PERSON" : "SCRUB", T.worker ? "person at 2.4 m < 3.0 m -> pause" : "person clear -> resume", T.worker ? "PAUSE_PERSON" : "SCRUB"), { tog: 1, desc: "A person steps within 3 m. Every arm stops until they leave.", score: () => (scrub() ? 0.4 : 0) });
  let leak = 0;
  U("leak", "Gas leak", "wind", "⇧6", "gas_leak", () => ({ on: !!T.leak }), () => H.judge("GAS", T.leak ? "flange F2 leaking -> detectors rising" : "leak isolated"), { tog: 1, keys: "lel plume alarm", desc: "Flange F2 starts leaking. Detectors climb, the plume grows and Control centre ranks a response.", score: () => (S.started && !S.gasAlarm ? 0.35 : 0) });
  setInterval(() => {
    if (!S.sim || (!T.leak && leak < 0.003)) return;
    leak += ((T.leak ? 0.18 : 0) - leak) * 0.06;
    H.apply({ gas: { sensors: [["GD-1", "Flange F2", 1], ["GD-2", "Separator", 0.42], ["GD-3", "Compressor", 0.12], ["GD-4", "Wellhead", 0.05]].map(([i, at, k]) => ({ id: i, at, lel: Math.max(0.01, leak * k) })), alarm: leak >= 0.1, wind: { dir_deg: 225, speed_ms: 6 } } });
  }, 250);
  U("battery", "Robot battery fault", "battery", "⇧7", "battery_fault", () => ({ robot: bot()?.id || "unit-1" }), () => { const r = bot(); H.banner("warn", "Battery fault", `${r ? r.name || H.rname(r) : "Unit 1"} · to the dock, its job goes back to the queue`, "battery"); H.judge("TO_DOCK", "battery fault -> dock, requeue job"); }, { keys: "fault dock power", desc: "The focused robot's battery faults. It docks and the coordinator re-assigns its job." });
  U("link", "Cut the Gemini link", "link", "⇧8", "cut_link", () => ({ on: !!T.link }), null, { tog: 1, keys: "edge offline autonomy cloud", desc: "Drops the cloud link. The coordinator and robots keep working on their last plan, on the edge.",
    pre: () => { S.linkCut = !!T.link; H.banner(T.link ? "warn" : "ok", T.link ? "Gemini link cut" : "Gemini link back", T.link ? "Robots keep their last plan · edge only" : "Planner and second opinions resume", "link"); H.addLog({ t: S.t, from: "Judge", to: "EDGE", reason: T.link ? "cloud link cut -> edge autonomy, last plan kept" : "cloud link restored" }); },
    score: () => (S.control || S.gemini ? 0.3 : 0) });
  ["calm", "wind", "rain", "fog", "storm"].forEach((lv) => U("w" + lv, "Weather: " + lv, lv === "storm" || lv === "wind" ? "wind" : lv === "calm" ? "sun" : "cloud", lv === "storm" ? "⇧9" : undefined, "set_weather",
    { level: lv, ...(lv === "wind" || lv === "storm" ? { wind_dir: 225, gust: lv === "storm" ? 26 : 14 } : {}) }, null,
    { keys: "weather " + lv, desc: `Sets the weather to ${lv}. Gusts make the arm retract and wait; rain and fog cut detector confidence.`, pre: () => { S.weather = { level: lv }; H.addLog({ t: S.t, from: "Judge", to: "WEATHER", reason: "weather -> " + lv }); if (lv === "storm") H.banner("warn", "Storm", "Gusts to 26 m/s · the arm retracts in each gust", "wind"); } }));
  [["obstacle", "Drop an obstacle", "block", "⇧0", "drop_obstacle", "obstacle on the walkway -> re-route"], ["walkway", "Close a walkway", "block", undefined, "close_walkway", "walkway closed -> re-route via the next segment"],
    ["stairs", "Close the stairs", "block", undefined, "close_stairs", "stairs closed -> jobs on other decks wait"], ["clear", "Clear all changes", "x", undefined, "clear", "scenario cleared"]]
    .forEach(([id, label, ic, key, name, why]) => U(id, label, ic, key, name, {}, null, { keys: "nav route block path", desc: "Planner re-routes around it; robots on the segment wait.", pre: () => H.addLog({ t: S.t, from: "Judge", to: name.toUpperCase(), reason: why }) }));
})();
