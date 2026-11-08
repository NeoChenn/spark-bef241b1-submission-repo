import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from integration import assemble_film as af

FFPROBE = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"


def edl_rows(scale=1.0):
    spans = [("establish", 0, 7, "cut", 1.0), ("control_room", 7, 12, "crossfade", 1.0),
             ("board", 12, 24, "crossfade", 1.0), ("unit2_job", 24, 39, "crossfade", 1.0),
             ("unit1_job", 39, 54, "crossfade", 1.0), ("walk_back", 54, 58, "crossfade", 2.5),
             ("conclusion", 58, 62, "crossfade", 1.0)]
    return [{"name": n, "film": [a * scale, b * scale], "transition": tr, "rate": r} for n, a, b, tr, r in spans]


def stub_overlays(root: Path, size, scale=1.0, step=1):
    w, h = size
    spans = {"board": (12, 24), "readout": (24, 54), "conclusion": (58, 62)}
    for name, (a, b) in spans.items():
        d = root / name
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(json.dumps({"start": a * scale}))
        for k in range(0, int(round((b - a) * scale * af.FPS)), step):
            im = Image.new("RGBA", size, (0, 0, 0, 0))
            g = ImageDraw.Draw(im)
            if name == "readout":
                g.rounded_rectangle([w - w // 4, h - h // 6, w - 16, h - 16], 12, fill=(10, 14, 20, 200))
                g.text((w - w // 4 + 12, h - h // 6 + 10), f"G-F5 {16 - k / 30:.1f} %LEL", fill=(242, 139, 130, 255))
            else:
                g.rounded_rectangle([w // 8, h // 8, w - w // 8, h - h // 8], 18, fill=(12, 16, 22, 170),
                                    outline=(46, 196, 182, 200), width=2)
                g.text((w // 8 + 20, h // 8 + 20), f"{name} {k}", fill=(238, 242, 246, 255))
            im.save(d / f"{k:05d}.png")


def stub_frames(root: Path, size, n=4):
    d = root / "frames"
    d.mkdir(parents=True)
    for i in range(n):
        a = np.zeros((size[1], size[0], 3), np.uint8)
        a[..., 0] = 40 + 50 * i
        a[:, ::8] = 230
        Image.fromarray(a).save(d / f"{i:05d}.jpg")
    return root


def test_edl_rows_parse_tolerantly():
    shots, _ = af.load_edl_rows(edl_rows())
    assert [s.name for s in shots][:3] == ["establish", "control_room", "board"]
    b = shots[2]
    assert (b.start, b.end, b.xfade, b.blur, b.readout) == (12.0, 24.0, 0.5, True, False)
    assert shots[0].xfade == 0.0 and shots[3].readout and shots[5].ff and shots[5].rate == 2.5
    assert shots[6].blur
    s = af.shot_from({"id": "x", "film_start": 3, "duration": 2, "transition": {"kind": "dissolve", "duration": 0.8}})
    assert (s.start, s.end, s.xfade) == (3.0, 5.0, 0.8)


def test_crossfade_weights_centre_on_the_cut():
    shots, _ = af.load_edl_rows(edl_rows())
    assert af.active(shots, 11.0) == [(shots[1], 1.0)]
    (a, wa), (b, wb) = af.active(shots, 12.0)
    assert (a.name, b.name) == ("control_room", "board") and abs(wa - 0.5) < 1e-9 and abs(wb - 0.5) < 1e-9
    assert af.active(shots, 6.7) == [(shots[0], 1.0)] and len(af.active(shots, 7.0)) == 2
    cut, _ = af.load_edl_rows([{"name": "a", "film": [0, 2]}, {"name": "b", "film": [2, 4], "transition": "cut"}])
    assert af.active(cut, 1.99) == [(cut[0], 1.0)] and af.active(cut, 2.0) == [(cut[1], 1.0)]


def test_chime_is_gentle_two_tone_twice_at_minus_18():
    c = af.chime()
    assert abs(af.dbfs(c) + 18.0) < 0.01
    env = np.abs(c)
    onset = np.argmax(env > 1e-4)
    assert env[onset:onset + int(0.005 * af.SR)].max() < 0.5 * env.max()
    frame = int(0.05 * af.SR)
    rms = np.array([np.sqrt(np.mean(c[i:i + frame] ** 2)) for i in range(0, len(c) - frame, frame)])
    peaks = [i for i in range(1, len(rms) - 1) if rms[i] > rms[i - 1] and rms[i] >= rms[i + 1] and rms[i] > 0.4 * rms.max()]
    assert len(peaks) >= 4
    x = af.soundtrack(30.0, 14.5)
    assert af.dbfs(x[: int(14 * af.SR)]) < -30.0 and af.dbfs(x) <= -17.5


def test_board_shot_is_blurred_and_dimmed():
    shots, _ = af.load_edl_rows(edl_rows())
    a = np.zeros((90, 160, 3), np.uint8)
    a[:, ::4] = 255
    im = Image.fromarray(a)
    out = np.asarray(af.background(shots[2], [], round(13 * af.FPS), (160, 90), im), float)
    assert out.mean() < 0.6 * a.mean() and out[:, 40:120].std() < 0.2 * a.std()


def probe(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "stream=codec_type,codec_name,width,height,pix_fmt,"
                          "avg_frame_rate:format=duration", "-of", "json", str(path)], capture_output=True, text=True,
                         check=True).stdout
    return json.loads(out)


def test_end_to_end_with_stand_ins(tmp_path):
    scale = 0.1
    (tmp_path / "edl.json").write_text(json.dumps({"inject": 1.45, "shots": edl_rows(scale)}))
    stub_frames(tmp_path / "isaac", (320, 180))
    stub_overlays(tmp_path / "ovl", (320, 180), scale)
    r = af.assemble(str(tmp_path / "edl.json"), tmp_path / "film.mp4", tmp_path / "isaac", tmp_path / "ovl",
                    (640, 360), teaser=False)
    j = probe(r["out"])
    kinds = {s["codec_type"]: s for s in j["streams"]}
    assert kinds["video"]["codec_name"] == "h264" and kinds["audio"]["codec_name"] == "aac"
    assert kinds["video"]["pix_fmt"] == "yuv420p" and kinds["video"]["avg_frame_rate"] == "30/1"
    assert abs(float(j["format"]["duration"]) - 6.2) < 0.1
    assert r["chime_at"] == 1.45 and set(r["overlays"]) == {"board", "readout", "conclusion"}
    assert "board" in r["crossfades"] and r["frames"]["board"] == 4


def test_camera_shots_edl_shape():
    from dataclasses import dataclass
    from typing import Callable, Optional

    @dataclass(frozen=True)
    class S:
        name: str
        film: tuple
        sim: Optional[tuple]
        rate: float
        camera: Callable
        t_in: str = "cut"
        t_out: str = "cut"
        two_d: bool = False
        badge: str = ""

        def to_dict(self):
            return {"name": self.name, "film": list(self.film), "sim": self.sim, "rate": self.rate, "in": self.t_in,
                    "out": self.t_out, "two_d": self.two_d, "badge": self.badge}

    cam = lambda u, t: None
    rows = [S("rig", (0.0, 7.0), (0.0, 7.0), 1.0, cam, "fade", "xfade"),
            S("control_push", (7.0, 12.0), (7.0, 12.0), 1.0, cam, "xfade", "xfade"),
            S("board", (12.0, 22.0), None, 1.0, cam, "xfade", "xfade", True),
            S("unit2_walk", (22.0, 24.0), (38.0, 40.0), 1.0, cam, "xfade", "cut"),
            S("unit1_soak", (24.0, 25.5), (74.0, 84.0), 6.7, cam, "cut", "cut", badge="7x"),
            S("walk_back", (25.5, 29.5), (98.0, 108.0), 2.5, cam, "xfade", "xfade", badge="2.5x"),
            S("conclusion", (29.5, 33.5), None, 1.0, cam, "xfade", "fade", True)]
    shots, _ = af.load_edl_rows(rows)
    by = {s.name: s for s in shots}
    assert by["rig"].fade_in and by["rig"].xfade == 0.5 and by["conclusion"].fade_out
    assert by["board"].blur and by["conclusion"].blur and not by["control_push"].blur
    assert not by["unit2_walk"].ff and by["walk_back"].ff and by["walk_back"].badge_text == "2.5×"
    assert by["unit1_soak"].ff and by["unit1_soak"].badge_text == "7×" and by["unit2_walk"].xfade == 0.5
    im = Image.new("RGB", (160, 90), (200, 200, 200))
    k0 = af.compose(by["rig"], [], [], 0, (160, 90), im)
    k1 = af.compose(by["rig"], [], [], 30, (160, 90), im)
    assert np.asarray(k0).mean() < 5 and np.asarray(k1).mean() > 190


def test_film_edl_capture_plan_chime_and_frames(tmp_path):
    shots, meta = af.load_edl(None, None, 60.0)
    long, _ = af.load_edl(None, None, None)
    assert max(s.end for s in shots) == 60.0 and max(s.end for s in long) == 66.0
    by = {s.name: s for s in shots}
    t = af.inject_time(shots, dict(meta, summary={"beats": {"alarm": 6.6}}))
    assert abs(t - (by["board"].start + 6.6 / by["board"].rate)) < 0.01
    specs = {(p, s): e for p, s, e in af.capture_specs(shots, "short")}
    board = specs[("board", "board")]
    assert board[-2:] == [f"{by['board'].start:g}", f"{by['board_hold'].end:g}"] and "--sim" in board
    ff = {s for p, s in specs if p == "ff"}
    assert ff == {s.name for s in shots if s.ff} and any(n.startswith("walk_back") for n in ff)
    assert specs[("conclusion", "conclusion")][:2] == ["--cards", "short"]
    (tmp_path / "board" / "frames").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "board" / "frames" / "00000.jpg")
    names = [s.name for s in shots]
    assert len(af.shot_frames(tmp_path, by["board_hold"], names)) == 1
    assert af.shot_frames(tmp_path, by["unit1_soak"], names) == []
    d = tmp_path / "ovl" / "ff_walk_back"
    d.mkdir(parents=True)
    Image.new("RGBA", (8, 8)).save(d / "001520.png")
    o = af.load_overlays(tmp_path / "ovl")[0]
    assert o.at(1520) is not None and o.at(1521) is None


def test_two_d_drift_moves_slowly_and_continuously():
    a = np.zeros((90, 160, 3), np.uint8)
    a[40:50, 70:90] = 255
    im = Image.fromarray(a)
    f0, f1, f9 = (np.asarray(af.drift(im, t), float) for t in (0.0, 1 / 30, 9.0))
    assert np.abs(f1 - f0).mean() < 2.0 and np.abs(f9 - f0).mean() > np.abs(f1 - f0).mean()
