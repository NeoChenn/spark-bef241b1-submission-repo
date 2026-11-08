import json
import shutil
import subprocess
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "rig" / "web" / "static" / "sound.js"

HARNESS = r"""
const vm = require("vm"), fs = require("fs");
const made = { osc: 0, buf: 0 };
const param = () => ({ value: 0, setValueAtTime() {}, exponentialRampToValueAtTime() {}, linearRampToValueAtTime() {} });
const node = (extra = {}) => ({ connect() {}, start() {}, stop() {}, gain: param(), frequency: param(), Q: param(), threshold: param(), ratio: param(), ...extra });
class AC {
  constructor() { this.state = "suspended"; this.currentTime = 0; this.sampleRate = 8000; this.destination = node(); }
  resume() { this.state = "running"; }
  createDynamicsCompressor() { return node(); }
  createGain() { return node(); }
  createBiquadFilter() { return node(); }
  createOscillator() { made.osc++; return node(); }
  createBufferSource() { made.buf++; return node(); }
  createBuffer(c, n) { return { getChannelData: () => new Float32Array(n) }; }
}
function run(style) {
  const L = {}, timers = [], store = {};
  const hooks = { header: [], alert: [], spotlight: [] };
  const HUD = { S: { robots: [], fleet: null } };
  HUD.on = style === "events" ? function (name, f) { hooks[name].push(f); } : (f) => hooks.header.push(f);
  const ctx = {
    HUD, AudioContext: AC, console, Math, Array, Float32Array, JSON,
    localStorage: { getItem: (k) => store[k] ?? null, setItem: (k, v) => (store[k] = String(v)) },
    setInterval: (f) => timers.push(f), clearInterval: (i) => (timers[i - 1] = () => {}),
    addEventListener: (e, f) => (L[e] = L[e] || []).push(f),
    document: { getElementById: () => null, addEventListener() {} },
  };
  ctx.window = ctx;
  vm.createContext(ctx);
  vm.runInContext(fs.readFileSync(process.argv[2], "utf8"), ctx);
  const fire = (e, o) => (L[e] || []).forEach((f) => f({ preventDefault() {}, stopImmediatePropagation() {}, target: { tagName: "BODY" }, ...o }));
  const step = (dt = 5) => { ctx.Sound.boot().currentTime += dt; timers.forEach((f) => f()); };
  const r = {};
  fire("pointerdown");
  const send = (as) => (style === "events" ? as.forEach((a) => hooks.alert.forEach((f) => f(a))) : hooks.header.forEach((f) => f({ alerts: as })));
  send([{ seq: 1, kind: "gas_alarm", text: "old" }]);
  step();
  r.baseline = made.osc;
  send([{ seq: 1, kind: "gas_alarm" }, { seq: 2, kind: "gas_alarm", text: "Gas" }]);
  r.alarm = made.osc;
  send([{ seq: 2, kind: "gas_alarm" }]);
  r.dup = made.osc;
  step();
  send([{ seq: 3, kind: "approval" }]);
  r.chime = made.osc;
  HUD.S.robots = [{ id: "R1", state: "WELD" }];
  const b0 = made.buf;
  step(); step(); step();
  r.weld = ctx.Sound.weld();
  r.crackle = made.buf - b0;
  fire("keydown", { key: "m" });
  r.muted = ctx.Sound.muted();
  r.weldMuted = ctx.Sound.weld();
  r.stored = store["hud.muted"];
  step();
  send([{ seq: 4, kind: "disturbance" }]);
  r.mutedPlays = made.osc - r.chime;
  fire("keydown", { key: "m" });
  HUD.S.robots = [{ id: "R1", state: "WALK" }];
  step();
  r.weldOff = !ctx.Sound.weld();
  made.osc = 0; made.buf = 0;
  return r;
}
console.log(JSON.stringify({ header: run("header"), events: run("events") }));
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    h = tmp_path_factory.mktemp("sound") / "harness.js"
    h.write_text(HARNESS)
    out = subprocess.run([node, str(h), str(SRC)], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@pytest.mark.parametrize("style", ["header", "events"])
def test_sound_schedules_alerts_and_weld(results, style):
    r = results[style]
    if style == "header":
        assert r["baseline"] == 0
    assert r["alarm"] - r["baseline"] == 8
    assert r["dup"] == r["alarm"]
    assert r["chime"] > r["dup"]
    assert r["weld"] and r["crackle"] > 0
    assert r["muted"] is True and r["weldMuted"] is False and r["stored"] == "1"
    assert r["mutedPlays"] == 0
    assert r["weldOff"]


def test_index_loads_sound():
    html = (SRC.parent / "index.html").read_text()
    assert '<script src="/static/sound.js"></script>' in html
