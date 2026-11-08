(() => {
  const post = (name, args) =>
    fetch("/api/event", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, args }) })
      .then(async (r) => {
        const body = await r.json().catch(() => ({}));
        dispatchEvent(new CustomEvent("rig:command-result", { detail: { name, args, ok: r.ok, body } }));
        return body;
      })
      .catch(() => null);
  const cmd = (id, label, name, args, shortcut) => ({ id, group: "Scenario", label, shortcut, run: () => post(name, args) });
  const COMMANDS = [
    cmd("drop.crate.r1", "Drop crate on R1's route", "drop_obstacle", { kind: "crate", at: "R1" }, "O"),
    cmd("drop.debris.r2", "Drop debris on R2's route", "drop_obstacle", { kind: "debris", at: "R2" }),
    cmd("drop.scaffold.r3", "Scaffold tower on R3's route", "drop_obstacle", { kind: "scaffold", at: "R3" }),
    cmd("drop.hose.r1", "Hose across R1's route", "drop_obstacle", { kind: "hose", at: "R1" }),
    cmd("drop.spill.stairs", "Oil spill on the west stairs", "drop_obstacle", { kind: "spill", at: "stairs_west_upper" }),
    cmd("close.stairs.west", "Close west stairs", "close_stairs", { id: "stairs_west_upper" }),
    cmd("close.catwalk.mid", "Close middle-deck catwalk", "close_walkway", { edge: "catwalk_mid" }),
    cmd("person.top", "Worker on the top-deck walkway", "person_at", { xy: [-1, 4], deck: "top_deck", duration: 90 }),
    cmd("weather.storm", "Weather: storm", "set_weather", { level: "storm", wind_dir: 90 }, "W"),
    cmd("weather.wind", "Weather: strong wind off the flare", "set_weather", { level: "wind", wind_dir: 0 }),
    cmd("weather.rain", "Weather: rain", "set_weather", { level: "rain" }),
    cmd("weather.fog", "Weather: fog", "set_weather", { level: "fog" }),
    cmd("weather.calm", "Weather: calm", "set_weather", { level: "calm" }, "Shift+W"),
    cmd("clear.all", "Clear all obstacles and closures", "clear", { id: "all" }, "Shift+O"),
  ];
  const matches = (sc, e) => {
    if (!sc) return false;
    const shift = sc.startsWith("Shift+");
    return shift === e.shiftKey && !e.metaKey && !e.ctrlKey && e.key.toLowerCase() === sc.replace("Shift+", "").toLowerCase();
  };
  window.RIG_COMMANDS = (window.RIG_COMMANDS || []).concat(COMMANDS);
  window.RIG_POST = post;
  window.RIG_SHORTCUT = (e) => window.RIG_COMMANDS.find((c) => matches(c.shortcut, e));
  dispatchEvent(new CustomEvent("rig:commands", { detail: COMMANDS }));
})();
