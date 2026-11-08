class Schedule:
    def __init__(self, rate: float = 2.0):
        self.period = 1.0 / rate
        self.last: dict[str, float] = {}

    def due(self, ids, t: float):
        ready = [(self.last.get(i, -1e9), k, i) for k, i in enumerate(ids) if t - self.last.get(i, -1e9) >= self.period]
        if not ready:
            return None
        _, _, rid = min(ready)
        self.last[rid] = t
        return rid


class Demand:
    def __init__(self, rate: float = 4.0, keepalive: float = 45.0, warmup: float = 0.5):
        self.period, self.keepalive, self.warmup = 1.0 / min(rate, 5.0), keepalive, warmup
        self.active = None
        self.until = self.next = 0.0

    def request(self, rid, on: bool, now: float):
        if not on:
            if self.active is None or (rid and rid != self.active):
                return None
            self.active = None
            return "stop"
        if not rid:
            return None
        self.until = now + self.keepalive
        if rid == self.active:
            return None
        action = "switch" if self.active else "start"
        self.active, self.next = rid, now + self.warmup
        return action

    def expire(self, now: float) -> bool:
        if self.active is not None and now > self.until:
            self.active = None
            return True
        return False

    def due(self, now: float) -> bool:
        return self.active is not None and now >= self.next

    def captured(self, now: float) -> None:
        self.next = now + self.period
