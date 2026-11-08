"""Mock bridge methods for the deck sweep (debris pickup, camera person check). Mixed into RigMockBridge.

    deck_frame() -> (rgb, CameraModel)     the body camera looking down the deck (policy/deck/scene.py)
    deck_reference() -> rgb                the clean picture of this stop (taken when the deck was known clean)
    request_deck_view(i) -> bool           step sideways / closer: a dusty view clears up
    pick_debris(u, v) -> bool              the arm picks the object whose image position is within reach of (u, v)
    body_frame() -> (rgb, CameraModel)     the body camera looking ahead, with a person if one is set
    body_range() -> (H, W) metres           its depth camera, as ground distance per pixel

In Isaac, pick_debris projects (u, v) onto the deck with the camera model and runs a grasp; the mock just removes
the object if the pixel is on it.
"""
from __future__ import annotations

from policy.deck.scene import Debris, camera_model, deck_view, person_range, person_view

PICK_RADIUS_PX = 45


class DeckMockMixin:
    def _deck_init(self, debris=(), person_m=None, deck_dust: float = 0.0):
        self.debris: list[Debris] = list(debris)
        self.stuck: set[int] = set()             # indexes of objects the gripper cannot hold (e.g. too big)
        self.picked: list[Debris] = []
        self.camera_person_m = person_m          # what the body camera sees (None = nobody)
        self.deck_dust = deck_dust
        self.deck_view_idx = 0

    def deck_frame(self):
        dust = self.deck_dust * (0.3 ** self.deck_view_idx)
        return deck_view(self.debris, dust=dust), camera_model()

    def deck_reference(self):
        """The clean picture of this patrol stop, stored when the deck was known clean (commissioning)."""
        return deck_view([])

    def request_deck_view(self, index: int) -> bool:
        self.deck_view_idx = index
        return True

    def pick_debris(self, u: float, v: float) -> bool:
        best, bd = None, PICK_RADIUS_PX
        for i, it in enumerate(self.debris):
            x, y = it.pixel()
            d = ((x - u) ** 2 + (y - v) ** 2) ** 0.5
            if d < bd:
                best, bd = i, d
        if best is None or best in self.stuck:
            return False
        self.picked.append(self.debris.pop(best))
        self.stuck = {i - (i > best) for i in self.stuck}
        return True

    def body_frame(self):
        return person_view(self.camera_person_m), camera_model()

    def body_range(self):
        """Ground distance per pixel from the depth camera, aligned with body_frame()."""
        return person_range(self.camera_person_m)
