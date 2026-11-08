(() => {
  const KEY = "hud.muted";
  const load = () => { try { return localStorage.getItem(KEY) === "1"; } catch { return false; } };
  const save = (v) => { try { localStorage.setItem(KEY, v ? "1" : "0"); } catch {} };
  const AC = window.AudioContext || window.webkitAudioContext;
  let ctx = null, out = null, noise = null, weld = null, last = null, own = load();

  const hud = () => window.HUD || {};
  const muted = () => (typeof hud().muted === "boolean" ? hud().muted : own);
  const live = () => !!ctx && ctx.state === "running" && !muted();

  function boot() {
    if (!AC) return null;
    if (!ctx) {
      ctx = new AC();
      const comp = ctx.createDynamicsCompressor();
      comp.threshold.value = -18; comp.ratio.value = 4;
      out = ctx.createGain(); out.gain.value = 0.6;
      out.connect(comp); comp.connect(ctx.destination);
      const n = ctx.sampleRate, b = ctx.createBuffer(1, n, n), d = b.getChannelData(0);
      for (let i = 0; i < n; i++) d[i] = Math.random() * 2 - 1;
      noise = b;
    }
    if (ctx.state === "suspended" && ctx.resume) ctx.resume();
    return ctx;
  }

  function env(g, t, peak, a, r) {
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + a);
    g.gain.exponentialRampToValueAtTime(0.0001, t + a + r);
  }

  function tone(f, t, peak, a, r, type = "sine", dest = out) {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type; o.frequency.setValueAtTime(f, t);
    env(g, t, peak, a, r);
    o.connect(g); g.connect(dest);
    o.start(t); o.stop(t + a + r + 0.05);
    return o;
  }

  function burst(t, dur, peak, f, q, dest = out) {
    const s = ctx.createBufferSource(), bp = ctx.createBiquadFilter(), g = ctx.createGain();
    s.buffer = noise;
    bp.type = "bandpass"; bp.frequency.value = f; bp.Q.value = q;
    env(g, t, peak, 0.002, dur);
    s.connect(bp); bp.connect(g); g.connect(dest);
    s.start(t, Math.random() * 0.9); s.stop(t + dur + 0.05);
  }

  const SOUNDS = {
    gas_alarm(t) {
      const lp = ctx.createBiquadFilter();
      lp.type = "lowpass"; lp.frequency.value = 2200; lp.connect(out);
      for (let i = 0; i < 8; i++) tone(i % 2 ? 784 : 587, t + i * 0.375, 0.14, 0.02, 0.33, "triangle", lp);
    },
    approval(t) {
      [[659.3, 0], [987.8, 0.14]].forEach(([f, d]) => { tone(f, t + d, 0.12, 0.005, 0.9); tone(f * 2, t + d, 0.025, 0.005, 0.4); });
    },
    disturbance(t) { tone(1174.7, t, 0.1, 0.004, 0.55); tone(587.3, t, 0.04, 0.004, 0.35); },
    click(t) { burst(t, 0.018, 0.08, 3200, 1.2); tone(1800, t, 0.03, 0.001, 0.02); },
    toast(t) { tone(2093, t, 0.035, 0.002, 0.05); },
  };
  SOUNDS.weld = SOUNDS.toast;

  const GAP = { gas_alarm: 3, approval: 1, disturbance: 0.8 }, at = {};
  function play(kind) {
    const f = SOUNDS[kind];
    if (!f || !live()) return false;
    const t = ctx.currentTime;
    if (t - (at[kind] ?? -99) < (GAP[kind] ?? 0.06)) return false;
    at[kind] = t;
    f(t + 0.01);
    return true;
  }

  function crackle() {
    if (!live()) return;
    const t = ctx.currentTime;
    for (let k = 0, n = 1 + Math.floor(Math.random() * 4); k < n; k++)
      burst(t + Math.random() * 0.12, 0.004 + Math.random() * 0.03, 0.02 + Math.random() * 0.05, 2000 + Math.random() * 4000, 0.8, weld.g);
  }

  function weldOn() {
    if (weld || !live()) return;
    const g = ctx.createGain(), bp = ctx.createBiquadFilter(), hiss = ctx.createBufferSource(), hg = ctx.createGain();
    g.gain.value = 0.7; g.connect(out);
    hiss.buffer = noise; hiss.loop = true;
    bp.type = "bandpass"; bp.frequency.value = 3500; bp.Q.value = 0.6;
    hg.gain.setValueAtTime(0.0001, ctx.currentTime);
    hg.gain.exponentialRampToValueAtTime(0.012, ctx.currentTime + 0.3);
    hiss.connect(bp); bp.connect(hg); hg.connect(g); hiss.start();
    weld = { g, hiss, timer: setInterval(crackle, 120) };
  }

  function weldOff() {
    if (!weld) return;
    const w = weld; weld = null;
    clearInterval(w.timer);
    const t = ctx.currentTime;
    w.g.gain.setValueAtTime(w.g.gain.value, t);
    w.g.gain.linearRampToValueAtTime(0, t + 0.25);
    w.hiss.stop(t + 0.3);
  }

  const welding = (S) => [...(S?.robots || []), ...(S?.fleet?.robots || [])].some((r) => r && (r.state === "WELD" || r.job_state === "WELD"));
  const tick = (S) => (welding(S) && live() ? weldOn() : weldOff());

  function alert(a) {
    if (!a || a.seq == null || (last != null && a.seq <= last)) return;
    last = a.seq;
    play(a.kind);
  }

  function alerts(list) {
    const as = Array.isArray(list) ? list.filter((a) => a && a.seq != null).sort((x, y) => x.seq - y.seq) : [];
    if (!as.length) return;
    if (last == null) last = as[as.length - 1].seq;
    else as.forEach(alert);
  }

  function setMuted(v) {
    own = !!v; save(own);
    if (window.HUD) window.HUD.muted = own;
    if (own) weldOff();
    else { boot(); play("toast"); }
  }

  const gesture = () => boot();
  ["pointerdown", "keydown", "touchstart"].forEach((e) => addEventListener(e, gesture, { capture: true, passive: true }));

  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || e.shiftKey || /INPUT|TEXTAREA/.test(e.target?.tagName || "")) return;
    if ((e.key || "").toLowerCase() !== "m") return;
    e.preventDefault(); e.stopImmediatePropagation();
    setMuted(!muted());
  }, true);

  function wire() {
    const H = window.HUD;
    if (!H) return false;
    if (typeof H.muted !== "boolean") H.muted = own;
    if (typeof H.on === "function") {
      if (H.on.length >= 2) {
        H.on("alert", alert);
        H.on("spotlight", () => play("click"));
      } else H.on((h) => alerts(h?.alerts));
    }
    if (!(typeof H.on === "function" && H.on.length >= 2)) {
      const pal = () => document.getElementById("palette")?.classList.contains("open");
      document.addEventListener("click", (e) => { if (e.target?.closest?.("#p-list")) play("click"); }, true);
      addEventListener("keydown", (e) => { if (e.key === "Enter" && pal()) play("click"); }, true);
    }
    setInterval(() => { alerts(H.S?.alerts); tick(H.S); }, 500);
    return true;
  }

  if (!wire()) addEventListener("DOMContentLoaded", wire, { once: true });
  window.Sound = { play, boot, mute: setMuted, muted, weld: () => !!weld };
})();
