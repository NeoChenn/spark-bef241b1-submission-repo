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
  const wait = setInterval(() => { const s = H.session(); if (!s) return; clearInterval(wait); if (s.video) attach(s.video); }, 200);
  setInterval(sync, 100);
})();
