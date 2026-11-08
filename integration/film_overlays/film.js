(() => {
  const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
  const ease = (x) => { x = clamp(x); return x * x * (3 - 2 * x); };
  const fade = (t, a, b) => ease((t - a) / (b - a));
  const pulse = (t) => 0.5 + 0.5 * Math.sin(2 * Math.PI * t - Math.PI / 2);
  function at(series, t) {
    if (!series || !series.length) return null;
    if (t <= series[0][0]) return series[0][1];
    for (let i = 1; i < series.length; i++) if (t <= series[i][0]) { const [t0, v0] = series[i - 1], [t1, v1] = series[i]; return v0 + (v1 - v0) * ((t - t0) / (t1 - t0 || 1)); }
    return series[series.length - 1][1];
  }
  function spark(series, t0, t1, now, w, h, lo, hi) {
    const pts = [], n = 48;
    for (let i = 0; i <= n; i++) { const t = t0 + (t1 - t0) * (i / n); if (t > now) break; pts.push([t, at(series, t)]); }
    if (pts.length < 2) return "";
    const x = (t) => ((t - t0) / (t1 - t0)) * w, y = (v) => h - 1 - (clamp((v - lo) / (hi - lo)) * (h - 2));
    return pts.map(([t, v], i) => `${i ? "L" : "M"}${x(t).toFixed(1)} ${y(v).toFixed(1)}`).join("");
  }
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  window.K = { clamp, ease, fade, pulse, at, spark, $, esc, data: window.FILM || {} };
})();
