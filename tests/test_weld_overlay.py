import base64
import io

import numpy as np
from PIL import Image

from policy.weld.job import make_job
from policy.weld.overlay import H, W, render, thumb


def test_overlay_renders_each_phase_and_thumb_shape():
    j = make_job("crack")
    frames = []
    j.on_frame = lambda job: frames.append(render(job)) if len(frames) < 400 and job.t % 5 < 0.06 else None
    j.run()
    img = render(j)
    assert img.shape == (H, W, 3) and img.dtype == np.uint8 and (H, W) == (240, 320)
    assert len(frames) > 5
    assert np.std(img) > 10
    th = thumb(j, seq=7)
    assert set(th) == {"jpg", "w", "h", "caption", "seq"} and th["seq"] == 7 and (th["w"], th["h"]) == (320, 240)
    im = Image.open(io.BytesIO(base64.b64decode(th["jpg"])))
    assert im.size == (320, 240) and im.format == "JPEG"
    assert "DONE" in th["caption"]


def test_overlay_marks_crack_in_red_before_repair():
    j = make_job("crack")
    j.inspect()
    j.view["x"] = float(np.mean(j.seam.crack[:, 0]))
    img = render(j).astype(int)
    red = (img[..., 0] > 180) & (img[..., 1] < 80) & (img[..., 2] < 80)
    assert red[:150].sum() > 20
