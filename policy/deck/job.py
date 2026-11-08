"""Deck sweep: two more camera skills for the rig robot, trained like the rust and crack models.

1. Debris. At every patrol stop, once the gas check says the air is normal, the robot looks down the deck. A loose
   object (can, bottle, packaging, a tool left behind) is a trip hazard and, offshore, dropped-object / foreign-object
   debris. The robot picks it up and checks it is gone, then carries on to the rust check.

       ... GAS_CHECK (normal) -> DEBRIS_CHECK -+-> deck clear      -> DETECT (rust, unchanged)
                                               +-> debris seen     -> DEBRIS_PICKUP -> DEBRIS_VERIFY -+-> clear -> DETECT
                                               |                                                      +-> more  -> DEBRIS_PICKUP
                                               +-> unsure          -> DEBRIS_SECOND_VIEW (x2) -> report, DETECT

   Debris is not an emergency: when the robot cannot pick something up or cannot tell, it reports it (warning) and
   carries on with its patrol instead of stopping.

2. People. The team's rule pauses the robot when a person is within 3 m, using the simulator's ground truth
   (``bridge.person_distance()``). With a person model, the robot's own body camera also counts: a confident person
   in view whose rough distance (from the window height) is under 3 m pauses it too, and it only resumes when BOTH
   the camera and the bridge say clear. The camera can add pauses; it never removes one.

Bridge methods (mock in policy/leak/mock.py, Isaac to do; see integration/INTERFACE.md):
    deck_frame() -> (rgb, CameraModel)        request_deck_view(i) -> bool
    deck_reference() -> rgb: a clean picture of this stop and view (taken when the deck was known clean)
    pick_debris(u, v) -> bool                 body_frame() -> (rgb, CameraModel)
    body_range() -> (H, W) metres, optional: depth as ground distance; without it the distance comes from the box
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from policy.orchestrator import S

DECK_STATES = {S.DEBRIS_CHECK, S.DEBRIS_SECOND_VIEW, S.DEBRIS_PICKUP, S.DEBRIS_VERIFY}

DECK_ALERTS = {
    "debris_cleared": ("info", False, "Picked up {n} loose object(s) from the deck at patrol stop {stop}.",
                       "No action needed."),
    "debris_left": ("warning", False, "Loose object on the deck at patrol stop {stop} that the robot could not "
                    "clear: {why}.", "Send someone to remove it on the next walk-round."),
}


@dataclass
class DeckConfig:
    max_deck_views: int = 2          # extra views when the debris check is unsure
    max_pickups: int = 5             # per patrol stop
    pick_retries: int = 1
    look_dt: float = 1.0             # s of sim time per look
    pick_dt: float = 4.0             # s of sim time per pick
    person_every_s: float = 0.5      # how often the camera person check runs


@dataclass
class DeckJob:
    debris: object = None            # detection.rig_vision.DebrisDetector (None = no debris skill)
    person: object = None            # detection.rig_vision.PersonDetector (None = ground truth only, as before)
    config: DeckConfig = field(default_factory=DeckConfig)


class DeckSweepMixin:
    """Mixed into RigOrchestrator (policy/leak/job.py). Expects self.bridge, self.cfg, self._record, self._advance,
    self._evidence, self._alert and self.patrol_idx from the team's Orchestrator."""

    deck: Optional[DeckJob] = None

    def _deck_reset(self) -> None:
        self.deck_views = 0
        self.pickups = 0
        self.pick_tries = 0
        self.debris_target = None
        self.last_debris = None
        self.last_person = None
        self._person_t = -1e9

    # ------------------------------------------------------------- hooks
    def _deck_next(self, nxt):
        """Called where the patrol would go to DETECT after the gas check: look at the deck first."""
        if nxt == S.DETECT and self.deck is not None and self.deck.debris is not None and hasattr(self.bridge,
                                                                                                    "deck_frame"):
            self._deck_reset()
            return S.DEBRIS_CHECK
        return nxt

    def _camera_person(self):
        """(close, detection) from the body camera, re-run at most every person_every_s of sim time."""
        if self.deck is None or self.deck.person is None or not hasattr(self.bridge, "body_frame"):
            return False, None
        if self.t - self._person_t >= self.deck.config.person_every_s or self.last_person is None:
            img, cam = self.bridge.body_frame()
            rng = self.bridge.body_range() if hasattr(self.bridge, "body_range") else None
            self.last_person = self.deck.person.detect(img, cam, rng)
            self._person_t = self.t
        r = self.last_person
        close = r.found and (r.distance_m is None or r.distance_m < self.cfg.person_radius)
        return close, r

    def _camera_person_preempt(self):
        if self.state == S.PAUSE_PERSON:
            return None
        close, r = self._camera_person()
        if not close:
            return None
        self.pause_resume = self.state
        d = "?" if r.distance_m is None else f"{r.distance_m:.1f} m"
        return (S.PAUSE_PERSON, f"camera: person in view (p={r.prob:.2f}, about {d}) < {self.cfg.person_radius} m "
                "-> pause", self._evidence(person_camera=r.summary()))

    def _s_pause_person(self):
        self.bridge.advance(1.0)
        d = self.bridge.person_distance() if hasattr(self.bridge, "person_distance") else None
        close, r = self._camera_person()
        if (d is None or d >= self.cfg.person_radius) and not close:
            nxt = self.pause_resume or S.DETECT
            cam = "" if r is None else f", camera p={r.prob:.2f}"
            return (nxt, f"person clear ({'none' if d is None else f'{d:.1f} m'}{cam}) -> resume {nxt.value}",
                    self._evidence(**({} if r is None else {"person_camera": r.summary()})))
        return None, "", {}

    # ------------------------------------------------------------- states
    def _look(self):
        self._advance(self.deck.config.look_dt)
        img, cam = self.bridge.deck_frame()
        ref = self.bridge.deck_reference() if hasattr(self.bridge, "deck_reference") else None
        r = self.deck.debris.detect(img, ref)
        self.last_debris = r
        return r

    def _deck_done(self, why: str, **ev):
        if self.pickups:
            ev["alert"] = self._alert("debris_cleared", n=self.pickups, stop=self.patrol_idx)
        return S.DETECT, f"{why} -> rust check", self._evidence(debris=self.last_debris.summary(), **ev)

    def _s_debris_check(self):
        r = self._look()
        if r.found:
            best = max(r.hits, key=lambda h: h.p)
            self.debris_target = best.centre
            return (S.DEBRIS_PICKUP, f"debris on the deck (p={best.p:.2f}) at pixel "
                    f"({best.centre[0]:.0f}, {best.centre[1]:.0f}) -> pick it up", self._evidence(debris=r.summary()))
        if r.unsure:
            if self.deck_views < self.deck.config.max_deck_views:
                return (S.DEBRIS_SECOND_VIEW, f"deck unclear (max p={r.prob:.2f}) -> take view {self.deck_views + 1}",
                        self._evidence(debris=r.summary()))
            al = self._alert("debris_left", stop=self.patrol_idx, why=f"still unclear after {self.deck_views} views")
            return self._deck_done("deck still unclear, reported", alert=al)
        return self._deck_done(f"deck clear (max p={r.prob:.2f})")

    def _s_debris_second_view(self):
        self.deck_views += 1
        if hasattr(self.bridge, "request_deck_view"):
            self.bridge.request_deck_view(self.deck_views)
        return S.DEBRIS_CHECK, f"moved for deck view {self.deck_views} -> look again", self._evidence()

    def _s_debris_pickup(self):
        u, v = self.debris_target
        ok = self.bridge.pick_debris(u, v)
        self._advance(self.deck.config.pick_dt)
        if ok:
            self.pickups += 1
            self.pick_tries = 0
            return S.DEBRIS_VERIFY, f"picked up the object at ({u:.0f}, {v:.0f}) -> check the deck", self._evidence()
        self.pick_tries += 1
        if self.pick_tries <= self.deck.config.pick_retries:
            return S.DEBRIS_CHECK, f"pick at ({u:.0f}, {v:.0f}) missed -> look again and retry", self._evidence()
        al = self._alert("debris_left", stop=self.patrol_idx, why=f"pick failed {self.pick_tries} times")
        return self._deck_done("could not pick it up, reported", alert=al)

    def _s_debris_verify(self):
        r = self._look()
        if not r.found:
            return self._deck_done(f"deck clear after {self.pickups} pickup(s) (max p={r.prob:.2f})")
        if self.pickups >= self.deck.config.max_pickups:
            al = self._alert("debris_left", stop=self.patrol_idx, why=f"more than {self.pickups} objects")
            return self._deck_done("pickup limit reached, reported", alert=al)
        best = max(r.hits, key=lambda h: h.p)
        self.debris_target = best.centre
        return (S.DEBRIS_PICKUP, f"more debris (p={best.p:.2f}) at ({best.centre[0]:.0f}, {best.centre[1]:.0f}) "
                "-> pick it up", self._evidence(debris=r.summary()))
