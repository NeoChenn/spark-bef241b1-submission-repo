(() => {
  const H = HUD, { S, esc, icon } = H, Q = new URLSearchParams(location.search);
  const calls = () => {
    const c = S.argonCalls || [], lat = c.map((r) => +r.latency_ms || 0), cost = c.reduce((a, r) => a + (+r.cost_usd || 0), 0);
    return c.length ? `${c.length} Argon calls, ${(lat.reduce((a, b) => a + b, 0) / lat.length / 1000).toFixed(1)} s mean latency, $${cost.toFixed(4)} in total.` : "";
  };
  const STEPS = [
    { t: 0, at: "0 s", title: "The setup", pause: true, pins: ["fleet", "control"], keys: ["B", "A", "⇧M"],
      say: "Two Go2 robots on the top deck: Unit 1 Scrubber does rust, Unit 2 Fixer does gas. Six fixed detectors G-F1…G-F6. Argon (gemini-4-argon) is the command centre." },
    { t: 2, at: "2–11 s", title: "Rust job", pins: ["vitals", "mission"], keys: ["V", "N"],
      say: "Unit 1 sprays remover, soaks, then scrubs. Brush force stays under the 40 N limit." },
    { t: 4.5, at: "5–10 s", title: "A faulty detector", pins: ["gas", "control", "log"], keys: ["G", "A", "L"],
      say: "G-F6 reads 14 % LEL, but Unit 2 right beside it reads 2 %: PIPE_SENSOR_FAULT. Argon files ticket MT-001 for a human. No repair is sent." },
    { t: 17.5, at: "18–21 s", title: "Leak at F5", pins: ["gas", "control", "log"], keys: ["G", "A"],
      say: "Bolt 3 at F5 works loose. G-F5 goes amber, then red. The spark rule holds Unit 1 at once, on the edge, without waiting for Gemini." },
    { t: 23.5, at: "24.8 s", title: "Dispatch, and a refusal", pins: ["control", "log"], keys: ["A", "L"],
      say: "Argon sends Unit 2 to check. The 'resume rust' in the same reply is refused, because G-F5 is still red." },
    { t: 40.5, at: "41–57 s", title: "Unit 2 fixes the leak", pins: ["fleet", "control", "log"], keys: ["B", "A"],
      say: "Unit 2 sniffs at the flange: both sources agree, LEAK_CONFIRMED. Argon sends fix_leak, and Unit 2 re-torques bolt 3." },
    { t: 57.5, at: "58–69 s", title: "All clear", pins: ["gas", "control", "log"], keys: ["G", "A"],
      say: "G-F5 falls back to green. Argon clears the zone and Unit 1 resumes the paused rust job." },
    { t: 80.5, at: "81–88 s", title: "Finish", pins: ["vitals", "pipe", "control"], keys: ["V", "M"],
      say: () => `Unit 1 verifies and paints; Unit 2 is docked. ${calls()}` },
    { at: "Live only", title: "Cut the link, give an order", keys: ["⇧8", "⇧A"],
      say: "In this replay, ⇧8 and the order box only add a banner and a log row; the recorded run does not change. In a live run the edge rules take over when the link is cut, and Argon reads the order." },
  ];
  const btn = document.createElement("button");
  btn.id = "guide-btn";
  btn.className = "glass";
  btn.innerHTML = `${icon("play")}<span>Guide</span><kbd>T</kbd>`;
  const card = document.createElement("section");
  card.className = "card guide";
  card.hidden = true;
  document.body.append(btn);
  let cur = +(Q.get("guide") || 0), open = false, shown = "";
  const now_ = () => STEPS.reduce((k, st, i) => (st.t != null && S.started && S.t >= st.t - 0.6 ? i : k), -1);
  function render() {
    const st = STEPS[cur], n = now_(), say = typeof st.say === "function" ? st.say() : st.say;
    const html = `<header>${icon("play")}<span>Guide</span><em>${cur + 1} / ${STEPS.length}</em><button class="gp" data-a="prev" aria-label="Previous step">‹</button><button class="gp" data-a="next" aria-label="Next step">›</button><button class="x" data-a="close" aria-label="Close guide">${icon("x")}</button></header>`
      + `<div class="g-chips">${STEPS.map((x, i) => `<button data-i="${i}" class="${i === cur ? "on" : ""} ${i === n ? "now" : ""} ${x.t == null ? "live" : ""}">${i + 1}</button>`).join("")}</div>`
      + `<div class="g-step"><p class="g-t"><b>${esc(st.title)}</b><em>${esc(st.at)}</em></p><p class="g-say">${esc(say)}</p>`
      + `<p class="g-keys">${st.keys.map((k) => `<kbd>${esc(k)}</kbd>`).join("")}</p>`
      + `<div class="g-act">${st.t != null ? `<button class="g-jump" data-a="jump">Jump to ${st.t} s</button>` : ""}<button data-a="play">${H.paused() ? "Play" : "Pause"} <kbd>K</kbd></button></div></div>`;
    if (html !== shown) { shown = html; card.innerHTML = html; }
  }
  function go(i, jump) {
    cur = (i + STEPS.length) % STEPS.length;
    const st = STEPS[cur];
    if (jump && st.t != null) {
      if (H.replayed() && !H.jump(st.t)) {
        const p = new URLSearchParams(location.search);
        ["pin", "view", "t", "guide", "autostart"].forEach((k) => p.delete(k));
        location.search = `${p}&t=${st.t}&guide=${cur}&autostart&view=`;
        return;
      }
      H.openView(null); H.layout(st.pins || []); H.pause(!!st.pause);
    }
    render();
  }
  function toggle(on = !open) {
    open = on; S.guide = on; card.hidden = !on; btn.classList.toggle("on", on);
    if (on) { document.getElementById("dock-left").insertBefore(card, document.querySelector("#dock-left > section")); H.hideTip?.(); render(); }
  }
  card.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const a = b.dataset.a;
    if (b.dataset.i != null) go(+b.dataset.i, true);
    else if (a === "prev" || a === "next") go(cur + (a === "next" ? 1 : -1), true);
    else if (a === "jump") go(cur, true);
    else if (a === "play") { H.pause(); render(); }
    else if (a === "close") toggle(false);
  });
  btn.onclick = () => toggle();
  H.command({ id: "guide", group: "Demo", label: "Guided tour", icon: "play", key: "t", keys: "guide tour walkthrough steps help demo", desc: "A 9-step walkthrough of the fleet run. Each step jumps the replay to its moment and opens the right panels.", active: () => open, run: () => toggle() });
  H.command({ id: "pause", group: "Demo", label: "Pause or play the replay", icon: "play", key: "k", keys: "pause play stop freeze", desc: "Freezes the replay clock; press again to carry on.", active: () => H.paused(), run: () => { H.pause(); if (open) render(); } });
  setInterval(() => open && render(), 250);
  addEventListener("DOMContentLoaded", () => {
    if (!Q.has("guide")) return;
    toggle(true);
    const st = STEPS[cur];
    H.openView(null); H.layout(st.pins || []); H.pause(!!st.pause); render();
  });
})();
