import argparse
import asyncio
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PAGES = ("board", "readout", "ff", "conclusion")
RUN_FILES = {"hud": ("hud.jsonl", "live_hud.jsonl", "live_hud_1hz.jsonl"),
             "timeline": ("timeline.jsonl", "live_timeline.jsonl"),
             "summary": ("summary.json", "live_summary.json")}
KEYS = {"scrubber": "unit1", "fixer": "unit2", "leak": "G-F5", "detectors": "detectors", "lel": "lel", "led": "led",
        "robots": "robots", "battery": "battery", "rust": "rust_fraction", "force": "force_n",
        "board": "board", "beats": ("beats",), "alarm_beat": "alarm", "green_beat": "gf5_green", "argon_calls": "argon_calls", "dispatch": ("go_check", "fix_leak"), "hold": ("hold",)}
STATES = {"TRANSIT": "Walking", "SNIFF": "Sniffing at the flange", "INSPECT": "Inspecting the flange",
          "RETORQUE": "Re-torquing the bolt", "VERIFY": "Verifying", "RETURN": "Walking back", "DOCKED": "Docked",
          "SPRAY": "Spraying remover", "SOAK": "Soaking", "SCRUB": "Scrubbing", "PAINT": "Painting",
          "VERIFY_PAINT": "Checking the paint", "APPROACH": "Walking", "DETECT": "Inspecting", "HOLD": "Holding",
          "DONE": "Done", "IDLE": "Docked"}
DECISIONS = (("scrubber", ("hold",), 'Rust on the F5 line → <i class="dot u1"></i>Unit 1 held at its dock (spark rule)',
              "Unit 1 held · spark rule"),
             ("fixer", ("go_check", "fix_leak"), 'Leak at F5 → <i class="dot u2"></i>Unit 2, Fixer', "Leak at F5 → Unit 2"))
NOTE = ("scrubber", ("resume_rust",), 'Leak fixed → <i class="dot u1"></i>Unit 1, Scrubber, dispatched')
DEFAULT_RUN = REPO / "runs_sample/film"
FALLBACK_RUN = REPO / "runs_sample/fleet_demo"


def _jsonl(p: Path) -> list[dict]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _find(run: Path, kind: str) -> Path | None:
    return next((run / n for n in RUN_FILES[kind] if (run / n).is_file()), None)


def load_run(run) -> dict:
    run = Path(run)
    hp, tp, sp = _find(run, "hud"), _find(run, "timeline"), _find(run, "summary")
    heads = sorted((h for h in _jsonl(hp) if "t" in h), key=lambda h: float(h["t"])) if hp else []
    rows = [r for r in _jsonl(tp) if r.get("type") != "tick"] if tp else []
    return {"dir": str(run), "heads": heads, "rows": rows, "summary": json.loads(sp.read_text()) if sp else {}}


def _det(h: dict, did: str) -> dict | None:
    return next((d for d in h.get(KEYS["detectors"]) or [] if d.get("id") == did), None)


def _robot(h: dict, rid: str) -> dict | None:
    return next((r for r in h.get(KEYS["robots"]) or [] if r.get("id") == rid), None)


def alarm_window(R: dict) -> tuple[float | None, float | None]:
    red = green = None
    for h in R["heads"]:
        d = _det(h, KEYS["leak"])
        if not d:
            continue
        if red is None and d.get(KEYS["led"]) == "red":
            red = float(h["t"])
        elif red is not None and d.get(KEYS["led"]) == "green":
            green = float(h["t"])
            break
    return red, green


def _get(d: dict, path):
    for k in path:
        d = (d or {}).get(k)
    return d


def _knots(R: dict, span: float, sim_span) -> list[tuple[float, float]]:
    if sim_span and len(sim_span) == 3:
        s0, s1, rate = map(float, sim_span)
        u1 = min(span, (s1 - s0) / rate)
        return [(0.0, s0), (u1, s1)] + ([(span, s1)] if span > u1 + 1e-6 else [])
    if sim_span:
        return [(0.0, float(sim_span[0])), (span, float(sim_span[1]))]
    a = _get(R["summary"], KEYS["beats"] + (KEYS["alarm_beat"],)) or alarm_window(R)[0] or 6.6
    return [(0.0, 0.0), (0.6 * span, a + 2.4), (span, a + 5.4)]


def _u(knots, s: float) -> float:
    for (u0, s0), (u1, s1) in zip(knots, knots[1:]):
        if s <= s1 or (u1, s1) == knots[-1] or s1 == knots[-1][1]:
            return round(u0 + (u1 - u0) * (s - s0) / ((s1 - s0) or 1), 3)
    return knots[-1][0]


def _seg(k: str) -> str:
    return k.replace("-", "–")


def _stub_board(stub: dict) -> dict:
    b = copy.deepcopy(stub["board"])
    one = lambda v: [[0.0, v]]
    for d in b["detectors"]:
        d["lel"], d["led"] = one(d.pop("base")), one("green")
    for sg in b["segments"]:
        sg["index"], sg["coating"] = one(sg["index"]), one(sg["coating"])
    b["process"] = {k: one(v) for k, v in b["process"].items()}
    b["weather"]["wind_ms"] = one(b["weather"]["wind_ms"])
    for f in b["fleet"]:
        f["battery"] = one(f["battery"])
    return b


def _command(R: dict, rid: str, kinds) -> dict | None:
    return next((r for r in R["rows"] if r.get("type") == "command" and r.get("robot") == rid and r.get("kind") in kinds), None)


def board_data(R: dict, stub: dict, span: float = 12.0, sim_span=None) -> dict:
    b = _stub_board(stub)
    kn = _knots(R, span, sim_span)
    decs = []
    for k, (who, kinds, html, short) in enumerate(DECISIONS):
        row = _command(R, KEYS[who], kinds)
        if R["heads"] and not row:
            continue
        stub_d = next((d for d in stub["board"]["decisions"] if d["robot"] == KEYS[who]), {})
        t_run = float(row["t"]) if row else None
        if t_run is not None and not (kn[0][1] - 1e-6 <= t_run <= kn[-1][1] + 1e-6):
            continue
        decs.append({"robot": KEYS[who], "source": "argon" if (row or stub_d).get("source") == "argon" else "edge",
                     "t_run": t_run, "u": _u(kn, t_run) if t_run is not None else round((0.56 + 0.16 * k) * span, 3),
                     "short": short, "html": html})
    decs.sort(key=lambda d: d["u"])
    b["decisions"] = decs
    for f in b["fleet"]:
        f["after"] = next((i for i, d in enumerate(decs) if d["robot"] == f["id"]), None)
        if f["after"] is None:
            f.pop("state_after", None)
    calls = _get(R["summary"], (KEYS["argon_calls"],))
    if isinstance(calls, list) and calls:
        b["argon"]["calls"] = len(calls)
        b["argon"]["cost_usd"] = round(sum(float(c.get("cost_usd") or 0) for c in calls), 4)
    heads = [h for h in R["heads"] if isinstance(h.get(KEYS["board"]), dict)]
    if not heads:
        return b
    heads = [h for h in heads if kn[0][1] - 0.01 <= float(h["t"]) <= kn[-1][1] + 0.01]
    if not heads:
        return b
    first = heads[0][KEYS["board"]]
    b["modelled"] = bool(first.get("modelled"))
    b["corrosion_limit"] = first.get("corrosion_limit", b["corrosion_limit"])
    ids = [d["id"] for d in heads[0].get(KEYS["detectors"]) or []]
    b["detectors"] = [{"id": i, "flange": i.split("-")[-1], "lel": [], "led": []} for i in ids]
    b["segments"] = [{"id": _seg(k), "index": [], "coating": []} for k in first["corrosion_index"]]
    b["process"] = {k: [] for k in first["process"]}
    b["weather"] = {"wind_ms": [], "wind_from_deg": first["weather"].get("wind_from_deg", 0),
                    "sea": f"{first['weather'].get('sea_state')} · {str(first['weather'].get('sea_state_label', '')).split(' (')[0]}"}
    bat = {f["id"]: f for f in b["fleet"]}
    for f in b["fleet"]:
        f["battery"] = []
    for h in heads:
        u, bd = _u(kn, float(h["t"])), h[KEYS["board"]]
        for d in b["detectors"]:
            x = _det(h, d["id"])
            if x:
                d["lel"].append([u, round(float(x[KEYS["lel"]]), 2)])
                if not d["led"] or d["led"][-1][1] != x[KEYS["led"]]:
                    d["led"].append([u, x[KEYS["led"]]])
        for sg, k in zip(b["segments"], first["corrosion_index"]):
            sg["index"].append([u, bd["corrosion_index"][k]])
            c = bd["coating"].get(k, "")
            if not sg["coating"] or sg["coating"][-1][1] != c:
                sg["coating"].append([u, c])
        for k in b["process"]:
            b["process"][k].append([u, bd["process"][k]])
        b["weather"]["wind_ms"].append([u, bd["weather"]["wind_ms"]])
        for rid, v in (bd.get("battery") or {}).items():
            if rid in bat:
                bat[rid]["battery"].append([u, v])
    return b


def readout_data(R: dict, stub: dict, mode: str, sim_span, span: float) -> dict:
    if mode == "note":
        row = _command(R, KEYS[NOTE[0]], NOTE[1])
        if row is None and R["heads"]:
            return {"mode": mode}
        return {"mode": mode, "html": NOTE[2], "source": "argon" if (row or {"source": "argon"}).get("source") == "argon" else "edge",
                "t_run": float(row["t"]) if row else None}
    if not R["heads"] or not sim_span:
        return copy.deepcopy(stub["readout"][mode]) | {"mode": mode}
    kn = _knots(R, span, sim_span)
    rid = KEYS["fixer" if mode == "gas" else "scrubber"]
    hs = [h for h in R["heads"] if kn[0][1] - 0.5 <= float(h["t"]) <= kn[-1][1] + 0.5]
    out = {"mode": mode, "series": [], "force": [], "states": []}
    last = None
    for h in hs:
        t = _u(kn, float(h["t"])) if float(h["t"]) >= kn[0][1] else round(-(kn[0][1] - float(h["t"])), 3)
        if mode == "gas":
            d = _det(h, KEYS["leak"])
            if d:
                out["series"].append([t, float(d[KEYS["lel"]])])
        else:
            if h.get(KEYS["rust"]) is not None:
                out["series"].append([t, float(h[KEYS["rust"]])])
            if h.get(KEYS["force"]) is not None:
                out["force"].append([t, float(h[KEYS["force"]])])
        r = _robot(h, rid)
        st = STATES.get((r or {}).get("state"), ((r or {}).get("state") or "").title())
        if st and st != last:
            out["states"].append([t, st])
            last = st
    return out


def edl_shot(name: str, length=None) -> dict:
    from integration.film_edl import FILM_RUN, FilmBeats, edl, load_track
    beats = FilmBeats.load() if (FILM_RUN / "summary.json").exists() else FilmBeats()
    shots = edl(beats, track=load_track(), **({} if length is None else {"length": length}))
    s = next((x for x in shots if x.name == name), None)
    if s is None:
        raise ValueError(f"no shot {name!r}; shots are {[x.name for x in shots]}")
    return s.to_dict()


def alarm_to_fixed(R: dict) -> int | None:
    beats = _get(R["summary"], KEYS["beats"]) or {}
    a, g = beats.get(KEYS["alarm_beat"]), beats.get(KEYS["green_beat"])
    if a is None or g is None:
        a, g = alarm_window(R)
    return None if a is None or g is None else round(float(g) - float(a))


def conclusion_cards(R: dict, which: str = "long") -> list[dict]:
    n = alarm_to_fixed(R)
    alarm = [{"html": f"Alarm to fixed: <b>{n} seconds</b>."}] if n is not None else []
    piper = {"text": "On 6 July 1988, a gas leak on Piper Alpha killed 167 people."}
    close = {"lines": ["Today, a person walks up to the leak.", "It doesn't have to be that way."]}
    if which == "short":
        return alarm + [piper, close]
    return alarm + [{"text": "No person entered the gas zone."}, piper,
                    {"text": "Corrosion costs the world about $2.5 trillion a year.", "sub": "NACE IMPACT study, 2016"}, close]


def film_data(page: str, span: float, run=None, sim_span=None, mode="gas", rate=2.5, corner="br", cards="long") -> dict:
    stub = json.loads((HERE / "stub.json").read_text())
    run = Path(run) if run else (DEFAULT_RUN if _find(DEFAULT_RUN, "hud") else FALLBACK_RUN)
    R = load_run(run) if run.is_dir() else {"dir": None, "heads": [], "rows": [], "summary": {}}
    data = {"page": page, "span": span, "run": R["dir"] if R["heads"] else None}
    if page == "board":
        data["board"] = board_data(R, stub, span, sim_span)
    elif page == "readout":
        data["readout"] = readout_data(R, stub, mode, sim_span, span)
        data["corner"] = corner
    elif page == "ff":
        data["rate"] = rate
    elif page == "conclusion":
        data["cards"] = conclusion_cards(R, cards)
    return data


async def _capture(page: str, film_span, out: Path, data: dict, size, fps: int) -> list[Path]:
    from playwright.async_api import async_playwright
    out.mkdir(parents=True, exist_ok=True)
    f0, f1 = round(film_span[0] * fps), round(film_span[1] * fps)
    paths = []
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        pg = await browser.new_page(viewport={"width": size[0], "height": size[1]})
        await pg.add_init_script(f"window.FILM = {json.dumps(data)};")
        await pg.goto((HERE / f"{page}.html").as_uri())
        await pg.evaluate("Promise.all(['300 20px Inter', '400 20px Inter', '500 20px Inter', '600 20px Inter', "
                          "'400 20px \"JetBrains Mono\"', '500 20px \"JetBrains Mono\"'].map((f) => document.fonts.load(f)))"
                          ".then(() => document.fonts.ready).then(() => true)")
        if size != (1920, 1080):
            await pg.evaluate(f"document.documentElement.style.zoom = {size[0] / 1920}")
        for f in range(f0, f1):
            await pg.evaluate("t => window.render(t)", f / fps - film_span[0])
            path = out / f"{f:06d}.png"
            await pg.screenshot(path=str(path), omit_background=True)
            paths.append(path)
        await browser.close()
    return paths


def capture(page: str, film_span, out_dir, sim_span=None, run=None, mode="gas", rate=2.5, corner="br",
            size=(1920, 1080), fps: int = 30, cards: str = "long") -> list[Path]:
    if page not in PAGES:
        raise ValueError(f"page must be one of {PAGES}")
    span = float(film_span[1]) - float(film_span[0])
    data = film_data(page, span, run, sim_span, mode, rate, corner, cards)
    return asyncio.run(_capture(page, film_span, Path(out_dir), data, tuple(size), fps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", choices=PAGES)
    ap.add_argument("--film", nargs=2, type=float, default=None, metavar=("START", "END"))
    ap.add_argument("--shot", default=None)
    ap.add_argument("--length", type=float, default=None)
    ap.add_argument("--sim", nargs="+", type=float, default=None, metavar="S")
    ap.add_argument("--run", default=None)
    ap.add_argument("--mode", choices=("gas", "rust", "note"), default="gas")
    ap.add_argument("--rate", type=float, default=2.5)
    ap.add_argument("--corner", choices=("br", "tl"), default="br")
    ap.add_argument("--cards", choices=("short", "long"), default="long")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    if a.shot:
        sh = edl_shot(a.shot, a.length)
        a.film = a.film or sh["film"]
        a.sim = a.sim or (sh["sim"] and [sh["sim"][0], sh["sim"][1], sh["rate"]])
    if not a.film:
        ap.error("give --film START END or --shot NAME")
    if a.dump:
        print(json.dumps(film_data(a.page, a.film[1] - a.film[0], a.run, a.sim, a.mode, a.rate, a.corner, a.cards), indent=1))
        return
    out = a.out or str(REPO / "runs/film/overlays" / (a.shot and f"{a.page}_{a.shot}" or (a.page if a.page != "readout" else f"readout_{a.mode}")))
    paths = capture(a.page, a.film, out, a.sim, a.run, a.mode, a.rate, a.corner,
                    tuple(int(x) for x in a.size.split("x")), a.fps, a.cards)
    print(f"{len(paths)} frames -> {out}")


if __name__ == "__main__":
    main()
