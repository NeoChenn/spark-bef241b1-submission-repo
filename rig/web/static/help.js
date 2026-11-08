(() => {
  const H = HUD, { esc } = H, MAC = /Mac|iP/.test(navigator.platform), MOD = MAC ? "⌘" : "Ctrl";
  const SECTIONS = [
    ["Demo", (c) => ["guide", "start", "pause", "presenter", "closeall", "labels"].includes(c.id)],
    ["Panels", (c) => c.group === "Panels"],
    ["Views", (c) => c.group === "Views"],
    ["Argon", (c) => ["order", "intent", "u.link"].includes(c.id)],
    ["Cameras", (c) => c.group === "Cameras"],
    ["Unscripted changes", (c) => c.group === "Unscripted change" && c.id !== "u.link"],
  ];
  const STATIC = {
    Replay: [["&t=30", "Start the replay 30 s in (URL)"], ["&speed=2", "Play at 2× (URL)"], [MOD + " K → restart", "Restart replay"]],
    Search: [[MOD + " K", "Command palette"], ["/", "Command palette"], [MOD + " ↵", "Run, keep the palette open"], ["Esc", "Close palette or view"]],
    Help: [["H / ?", "Open or close this list"]],
  };
  const key = (k) => esc(k.length === 1 ? k.toUpperCase() : k).replace(/^⇧(.)$/, "⇧ $1");
  const row = ([k, t]) => `<li><kbd>${key(k)}</kbd><span>${esc(t)}</span></li>`;
  const sec = (title, rows) => (rows.length ? `<section><h3>${esc(title)}</h3><ul>${rows.map(row).join("")}</ul></section>` : "");
  function body() {
    const all = H.commands().filter((c) => c.key && c.id !== "keys");
    const order = (a, b) => (a.key.length - b.key.length) || a.key.localeCompare(b.key);
    return SECTIONS.map(([t, f]) => sec(t, all.filter(f).sort(t === "Demo" || t === "Argon" ? () => 0 : order).map((c) => [c.key, c.label.replace(/…$/, "")]))).join("")
      + Object.entries(STATIC).map(([t, rows]) => sec(t, rows)).join("");
  }
  const el = document.createElement("div");
  el.id = "help";
  el.setAttribute("role", "dialog");
  el.setAttribute("aria-label", "Controls");
  el.innerHTML = `<div class="hp glass hl"><header><b>Controls</b><span>Every key in this demo</span><kbd>H</kbd></header><div class="hp-b"></div></div>`;
  const tip = document.createElement("div");
  tip.id = "help-tip";
  tip.innerHTML = `New here? Press <kbd>T</kbd> Guide for a 2-minute walkthrough · <kbd>H</kbd> controls`;
  document.body.append(el, tip);
  let open = false;
  const fade = () => tip.classList.add("gone");
  setTimeout(fade, 12000);
  H.hideTip = fade;
  function toggle(on = !open) {
    open = on;
    if (on) { el.querySelector(".hp-b").innerHTML = body(); fade(); }
    el.classList.toggle("open", on);
  }
  el.addEventListener("pointerdown", (e) => { if (e.target === el) toggle(false); });
  tip.onclick = () => toggle(true);
  addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA/.test(e.target.tagName) || document.getElementById("palette").classList.contains("open")) return;
    const k = e.key.toLowerCase();
    if ((k === "h" && !e.shiftKey) || e.key === "?" || (open && k === "escape")) {
      e.preventDefault(); e.stopImmediatePropagation();
      toggle(k === "escape" ? false : undefined);
    } else if (open) toggle(false);
  }, true);
  const kc = H.commands().find((c) => c.id === "keys");
  if (kc) H.command({ ...kc, label: "Controls", desc: "Every control in the demo, grouped by section. H or ? opens it; H or Esc closes it.", run: () => setTimeout(() => toggle(true), 0) });
})();
