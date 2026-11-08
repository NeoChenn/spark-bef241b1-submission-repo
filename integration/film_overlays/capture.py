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
        "cost": ("argon", "cost_usd"), "calls": ("argon", "calls"), "dispatch": ("go_check", "fix_leak"), "hold": ("hold",)}
STATES = {"TRANSIT": "Walking", "SNIFF": "Sniffing at the flange", "INSPECT": "Inspecting the flange",
          "RETORQUE": "Re-torquing the bolt", "VERIFY": "Verifying", "RETURN": "Walking back", "DOCKED": "Docked",
          "SPRAY": "Spraying remover", "SOAK": "Soaking", "SCRUB": "Scrubbing", "PAINT": "Painting",
          "VERIFY_PAINT": "Checking the paint", "APPROACH": "Walking", "DETECT": "Inspecting", "HOLD": "Holding",
          "DONE": "Done", "IDLE": "Docked"}
DEC_HTML = {"fixer": 'Leak at F5 → <i class="dot u2"></i>Unit 2, Fixer',
            "scrubber": 'Rust on the F5 line → <i class="dot u1"></i>Unit 1, Scrubber, after the leak is fixed (spark rule)'}
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


def board_data(R: dict, stub: dict) -> dict:
    b = copy.deepcopy(stub["board"])
    if not R["heads"]:
        for d in b["decisions"]:
            d["html"] = DEC_HTML["fixer" if d["robot"] == KEYS["fixer"] else "scrubber"]
        return b
    h0 = R["heads"][0]
    red, _ = alarm_window(R)
    base = {d["id"]: float(d.get(KEYS["lel"]) or 0) for d in h0.get(KEYS["detectors"]) or []}
    for d in b["detectors"]:
        d["base"] = base.get(d["id"], d["base"])
    leak = next((d for d in b["detectors"] if d["id"] == KEYS["leak"]), None)
    if leak is not None and red is not None:
        pts = [(float(h["t"]), float(_det(h, KEYS["leak"])[KEYS["lel"]])) for h in R["heads"]
               if _det(h, KEYS["leak"]) and red - 4.0 <= float(h["t"]) <= red + 3.0]
        if len(pts) > 1:
            t0, t1 = pts[0][0], pts[-1][0]
            leak["climb"] = [[round((t - t0) / (t1 - t0), 4), round(v, 2)] for t, v in pts]
    for f in b["fleet"]:
        r = _robot(h0, f["id"])
        if r:
            f["battery"] = r.get(KEYS["battery"], f["battery"])
    decs = []
    for who, kinds in (("fixer", KEYS["dispatch"]), ("scrubber", KEYS["hold"])):
        rid = KEYS[who]
        row = next((r for r in R["rows"] if r.get("type") == "command" and r.get("robot") == rid and r.get("kind") in kinds), None)
        row = row or next((r for r in R["rows"] if r.get("type") == "command" and r.get("robot") == rid), None)
        if row:
            stub_d = next((d for d in b["decisions"] if d["robot"] == rid), {})
            decs.append({"robot": rid, "source": "argon" if row.get("source") == "argon" else "edge", "t_run": row.get("t"),
                         "short": stub_d.get("short", ""), "html": DEC_HTML[who]})
    b["decisions"] = decs
    for f in b["fleet"]:
        idx = next((i for i, d in enumerate(decs) if d["robot"] == f["id"]), None)
        f["after"] = idx
        if idx is None:
            f.pop("state_after", None)
    return b


def readout_data(R: dict, stub: dict, mode: str, sim_span, span: float) -> dict:
    if not R["heads"] or not sim_span:
        return copy.deepcopy(stub["readout"][mode]) | {"mode": mode}
    s0, s1 = sim_span
    rid = KEYS["fixer" if mode == "gas" else "scrubber"]
    lt = lambda t: round((t - s0) / (s1 - s0) * span, 3)
    hs = [h for h in R["heads"] if s0 - 0.5 <= float(h["t"]) <= s1 + 0.5]
    out = {"mode": mode, "series": [], "force": [], "states": []}
    last = None
    for h in hs:
        t = lt(float(h["t"]))
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


def conclusion_cards(R: dict) -> list[dict]:
    cards = []
    red, green = alarm_window(R)
    if red is not None and green is not None:
        cards.append({"html": f"Alarm to fixed: <b>{round(green - red)} seconds</b>."})
    cards.append({"text": "No person entered the gas zone."})
    cost, calls = _get(R["summary"], KEYS["cost"]), _get(R["summary"], KEYS["calls"])
    if cost is not None:
        words = "about one cent" if 0.005 <= cost < 0.015 else f"${cost:.2f}"
        cards.append({"text": f"Argon's decisions cost {words}.", "sub": f"{calls} calls to gemini-4-argon" if calls else None})
    cards += [{"text": "On 6 July 1988, a gas leak on Piper Alpha killed 167 people."},
              {"text": "Corrosion costs the world about $2.5 trillion a year.", "sub": "NACE IMPACT study, 2016"},
              {"text": "Today, a person walks up to the leak."},
              {"text": "It doesn't have to be that way."}]
    return cards


def film_data(page: str, span: float, run=None, sim_span=None, mode="gas", rate=2.5, corner="br") -> dict:
    stub = json.loads((HERE / "stub.json").read_text())
    run = Path(run) if run else (DEFAULT_RUN if _find(DEFAULT_RUN, "hud") else FALLBACK_RUN)
    R = load_run(run) if run.is_dir() else {"dir": None, "heads": [], "rows": [], "summary": {}}
    data = {"page": page, "span": span, "run": R["dir"] if R["heads"] else None}
    if page == "board":
        data["board"] = board_data(R, stub)
    elif page == "readout":
        data["readout"] = readout_data(R, stub, mode, sim_span, span)
        data["corner"] = corner
    elif page == "ff":
        data["rate"] = rate
    elif page == "conclusion":
        data["cards"] = conclusion_cards(R)
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
        await pg.evaluate("document.fonts.ready.then(() => true)")
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
            size=(1920, 1080), fps: int = 30) -> list[Path]:
    if page not in PAGES:
        raise ValueError(f"page must be one of {PAGES}")
    span = float(film_span[1]) - float(film_span[0])
    data = film_data(page, span, run, sim_span, mode, rate, corner)
    return asyncio.run(_capture(page, film_span, Path(out_dir), data, tuple(size), fps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page", choices=PAGES)
    ap.add_argument("--film", nargs=2, type=float, required=True, metavar=("START", "END"))
    ap.add_argument("--sim", nargs=2, type=float, default=None, metavar=("START", "END"))
    ap.add_argument("--run", default=None)
    ap.add_argument("--mode", choices=("gas", "rust"), default="gas")
    ap.add_argument("--rate", type=float, default=2.5)
    ap.add_argument("--corner", choices=("br", "tl"), default="br")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dump", action="store_true")
    a = ap.parse_args()
    if a.dump:
        print(json.dumps(film_data(a.page, a.film[1] - a.film[0], a.run, a.sim, a.mode, a.rate, a.corner), indent=1))
        return
    out = a.out or str(REPO / "runs/film/overlays" / (a.page if a.page != "readout" else f"readout_{a.mode}"))
    paths = capture(a.page, a.film, out, a.sim, a.run, a.mode, a.rate, a.corner,
                    tuple(int(x) for x in a.size.split("x")), a.fps)
    print(f"{len(paths)} frames -> {out}")


if __name__ == "__main__":
    main()
