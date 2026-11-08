from __future__ import annotations

from collections import defaultdict, deque
from datetime import date

INTRO_END = date(2026, 12, 1)
INTRO = (2.0, 10.0)
STANDARD = (4.0, 20.0)
CACHED_FRACTION = 0.05


def price(day: date) -> tuple[float, float]:
    return INTRO if day < INTRO_END else STANDARD


def cost_usd(tokens_in: int, tokens_out: int, tokens_thought: int, day: date, cached: int = 0) -> float:
    pin, pout = price(day)
    fresh = max(0, tokens_in - cached)
    return (fresh * pin + cached * pin * CACHED_FRACTION + (tokens_out + tokens_thought) * pout) / 1e6


class Budget:
    def __init__(self, cap_usd_day: float = 5.0):
        self.cap = cap_usd_day
        self.by_day: dict[date, float] = defaultdict(float)

    def spend(self, usd: float, day: date) -> None:
        self.by_day[day] += usd

    def spent(self, day: date) -> float:
        return self.by_day.get(day, 0.0)

    def allows(self, day: date) -> bool:
        return self.spent(day) < self.cap


class RateLimiter:
    def __init__(self, interval_s: float = 6.0, alarm_gap_s: float = 1.5, per_minute: int = 12):
        self.interval_s, self.alarm_gap_s, self.per_minute = interval_s, alarm_gap_s, per_minute
        self.marks: deque = deque()

    def due(self, now: float, alarm: bool = False) -> bool:
        while self.marks and now - self.marks[0] >= 60.0:
            self.marks.popleft()
        if len(self.marks) >= self.per_minute:
            return False
        if not self.marks:
            return True
        return now - self.marks[-1] >= (self.alarm_gap_s if alarm else self.interval_s)

    def mark(self, now: float) -> None:
        self.marks.append(now)
