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
