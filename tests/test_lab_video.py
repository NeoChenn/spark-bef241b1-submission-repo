import numpy as np

from integration.lab_video import FrameComposer


def test_compose_frame_size_and_inset():
    fc = FrameComposer()
    wide = np.zeros((720, 1280, 3), np.uint8)
    close = np.full((720, 1280, 3), 200, np.uint8)
    hist = [(t * 0.1, 20 + 30 * (t % 7 == 0)) for t in range(100)]
    out = fc.compose(wide, close, "SCRUB (cycle 1/3)", ["rust left: 40%", "why: x"], 45.0, hist, 10.0, speed=1.0)
    assert out.shape == (720, 1280, 3)
    assert out[100, 1100].mean() > 150
    assert out[360, 400].mean() == 0


def test_state_colours():
    fc = FrameComposer()
    assert fc._state_colour("BACK_OFF") != fc._state_colour("SCRUB")
    assert fc._state_colour("HOLD") != fc._state_colour("BACK_OFF")
