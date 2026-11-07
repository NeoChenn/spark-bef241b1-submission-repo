(() => {
  const canvas = document.getElementById("view");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");
  const fpsEl = document.getElementById("fps");
  const ageEl = document.getElementById("age");

  const wsUrl = (location.protocol === "https:" ? "wss:" : "ws:") + "//" + location.host + "/ws/video";
  let backoff = 500;
  let pending = null;
  let decoding = false;
  let drawn = [];

  function setStatus(text, live) {
    statusEl.textContent = text;
    statusEl.classList.toggle("live", !!live);
  }

  async function drawLoop() {
    if (decoding || !pending) return;
    decoding = true;
    const data = pending;
    pending = null;
    try {
      const bmp = await createImageBitmap(new Blob([data], { type: "image/jpeg" }));
      if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
        canvas.width = bmp.width;
        canvas.height = bmp.height;
      }
      ctx.drawImage(bmp, 0, 0);
      bmp.close();
      drawn.push(performance.now());
    } catch (e) {
      console.warn("decode failed", e);
    } finally {
      decoding = false;
      if (pending) drawLoop();
    }
  }

  function connect() {
    setStatus("connecting…");
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      backoff = 500;
      setStatus("waiting for frames…");
      pollHealth();
    };
    ws.onmessage = (ev) => {
      pending = ev.data;
      drawLoop();
    };
    ws.onclose = () => {
      setStatus("disconnected, retrying…");
      setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, 5000);
    };
  }

  setInterval(() => {
    const now = performance.now();
    drawn = drawn.filter((t) => now - t <= 1000);
    fpsEl.textContent = String(drawn.length);
  }, 250);

  const STATE_TEXT = {
    no_sim: "waiting for sim…",
    warming: "rendering: shaders compiling (first start takes minutes)…",
    stalled: "sim connected, frames stalled",
    live: "live",
  };

  async function pollHealth() {
    try {
      const h = await (await fetch("/api/health", { cache: "no-store" })).json();
      ageEl.textContent = h.frame_age_s == null ? "no frames" : h.frame_age_s.toFixed(1) + " s";
      setStatus(STATE_TEXT[h.state] || h.state, h.state === "live");
    } catch {
      ageEl.textContent = "–";
    }
  }
  setInterval(pollHealth, 2000);
  pollHealth();
  connect();
})();
