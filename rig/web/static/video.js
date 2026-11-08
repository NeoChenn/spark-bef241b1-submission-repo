(() => {
  const H = HUD, { S } = H, Q = new URLSearchParams(location.search);
  if (!Q.get("replay")) return;
  ["cam1", "cam2", "cam3"].forEach((id) => { const c = H.commands().find((x) => x.id === id); if (c) H.command({ ...c, label: c.label + " (live sim only)", desc: c.desc + " Live sim only: the replay plays the recorded Isaac render instead." }); });
  let v = null, info = null;
  function attach(i) {
    info = i;
    v = document.createElement("video");
    Object.assign(v, { id: "sim-video", muted: true, playsInline: true, preload: "auto", src: i.url });
    v.addEventListener("error", () => { document.body.classList.remove("video"); v.remove(); v = null; });
    document.getElementById("backdrop").appendChild(v);
  }
  function sync() {
    if (!v || v.readyState < 1) return;
    const target = H.clock() - info.t0, end = Math.min(v.duration || Infinity, info.t1 > info.t0 ? info.t1 - info.t0 + 0.04 : Infinity);
    document.body.classList.toggle("video", target > -0.5 && target < end + 0.5);
    const playing = S.started && !H.paused() && target >= 0 && target < end, want = Math.max(0, Math.min(target, end - 0.04));
    v.playbackRate = H.speed;
    if (Math.abs(v.currentTime - want) > (playing ? 0.35 : 0.05)) v.currentTime = want;
    if (playing && v.paused) v.play().catch(() => {});
    else if (!playing && !v.paused) v.pause();
  }
  let cam = null, t0 = 0;
  const CLOSE = /_(job|scrub|spray|soak|paint)$/;
  const busy = (r) => r && !/DOCKED|IDLE|HOLD|DONE|QUEUED/.test(r.state || "");
  const nearest = (fr, t) => { let lo = 0, hi = fr.length - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (fr[m][0] <= t) lo = m; else hi = m; } return Math.abs(fr[lo][0] - t) <= Math.abs(fr[hi][0] - t) ? fr[lo] : fr[hi]; };
  const spans = () => { const r = []; cam.shots.map((x) => [x.frames[0][0], x.frames.at(-1)[0]]).sort((a, b) => a[0] - b[0]).forEach(([a, b]) => { const l = r.at(-1); if (l && a <= l[1] + 1.5) l[1] = Math.max(l[1], b); else r.push([a, b]); }); return r.map(([a, b]) => `${Math.round(a)}–${Math.round(b)}`).join(", "); };
  H.simcam = (S) => {
    const s = H.session();
    if (!s) return { none: "Loading the recording…" };
    if (v && document.body.classList.contains("video")) return { video: v, label: "Sim camera · Isaac render" };
    if (!cam) return { none: "No camera in this recording" };
    const t = H.clock() + t0, pref = S.focus || (S.robots.find((r) => busy(r)) || {}).id;
    let best = null;
    cam.shots.forEach((sh) => {
      if (t < sh.frames[0][0] - 1 || t > sh.frames.at(-1)[0] + 1) return;
      const f = nearest(sh.frames, t), dt = Math.abs(f[0] - t);
      if (dt > 1) return;
      const score = dt + (sh.robot && sh.robot === pref ? 0 : 5) + (CLOSE.test(sh.shot) ? 0 : 1);
      if (!best || score < best.score) best = { score, sh, f };
    });
    if (!best) return { none: `No sim camera at ${t.toFixed(1)} s · rendered shots cover ${spans()} s` };
    return { img: `${best.sh.base}/${String(best.f[1]).padStart(5, "0")}.jpg`, label: `Sim camera · ${best.sh.shot}`, shot: best.sh.shot, sim_t: best.f[0] };
  };
  const wait = setInterval(() => { const s = H.session(); if (!s) return; clearInterval(wait); t0 = +(s.snapshots[0]?.t || 0); cam = s.simcam || null; if (s.video) attach(s.video); }, 200);
  setInterval(sync, 100);
})();
