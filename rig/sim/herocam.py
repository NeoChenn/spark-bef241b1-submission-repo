import math

DECKS = (20.0, 31.25, 42.5)
MAIN = {"gas": "R2", "patrol": "R3", "repair": "R1"}
MODES = ("flyby", "wide", "chase")
ROBOTS_ROOT = "/World/Obstacles/Robots"
ROBOT_Z = 0.45
APERTURE = 20.955
ASPECT = 16 / 9

BACK = 3.5
HEIGHT = 1.8
AHEAD = 1.2
LOOK_Z = 0.5
PIVOT_Z = 1.0
MIN_BACK = 1.0
CHASE_FOCAL = 18.0
OFFSETS = (0.0, 40.0, -40.0, 80.0, -80.0, 180.0)
CHECK_S = 0.25
CUT_M = 8.0
W_POS, W_YAW, W_OUT = 4.0, 3.0, 1.5

WIDE_FOCAL = 24.0
WIDE_VIEW = {42.5: (-35.0, 42.0), 31.25: (-35.0, 22.0), 20.0: (-35.0, 22.0)}
WIDE_D = (14.0, 80.0)
WIDE_MIN_SPAN = 6.0
W_WIDE = 1.2


def spring(x, v, target, omega, dt):
    e = x - target
    k = math.exp(-omega * dt)
    a = v + omega * e
    return target + (e + a * dt) * k, (v - omega * a * dt) * k


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def deck_of(z):
    return max((d for d in DECKS if d <= z + 0.6), default=DECKS[0])


def ceiling_of(deck):
    return min((d for d in DECKS if d > deck + 1.0), default=None)


def view_dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def fit_distance(points, center, az, el, focal, aperture=APERTURE, aspect=ASPECT, margin=0.85):
    b = view_dir(az, el)
    f = (-b[0], -b[1], -b[2])
    n = math.hypot(f[0], f[1]) or 1.0
    r = (f[1] / n, -f[0] / n, 0.0)
    u = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
    tx = aperture / 2 / focal * margin
    ty = tx / aspect
    d = 0.0
    for p in points:
        q = (p[0] - center[0], p[1] - center[1], p[2] - center[2])
        dot = lambda a: q[0] * a[0] + q[1] * a[1] + q[2] * a[2]
        d = max(d, abs(dot(r)) / tx - dot(f), abs(dot(u)) / ty - dot(f))
    return d


def wide_shot(points, deck, focal=WIDE_FOCAL):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = WIDE_MIN_SPAN / 2
    pts = list(points) + [(cx - half, cy - half, deck), (cx + half, cy + half, deck)]
    center = (cx, cy, deck + LOOK_Z)
    az, el = WIDE_VIEW.get(deck, WIDE_VIEW[DECKS[-1]])
    top = ceiling_of(deck)
    d = WIDE_D[0]
    for _ in range(4):
        d = min(max(fit_distance(pts, center, az, el, focal), WIDE_D[0]), WIDE_D[1])
        if top is None or center[2] + d * math.sin(math.radians(el)) <= top - 1.5:
            break
        el = math.degrees(math.asin(max(min((top - 1.5 - center[2]) / d, 1.0), 0.05)))
    return center, d, az, el


class HeroCam:
    def __init__(self):
        self.mode, self.robot, self.main = "flyby", None, None
        self.poses = None
        self.blocked = None
        self.paths, self.fleet_poses = {}, {}
        self.key, self.last = None, None
        self.s = {}

    def meta(self):
        return {"mode": self.mode, "robot": self.robot if self.mode == "chase" else None}

    def _set(self, mode, robot=None):
        self.mode, self.robot = mode, robot if mode == "chase" else None
        self.key = None

    def event(self, name, args):
        args = args or {}
        if name == "camera":
            mode = args.get("mode")
            if mode not in MODES:
                return False
            self._set(mode, args.get("robot") or self.robot or self.main or "R1")
            return True
        if name == "select":
            robot = args.get("robot")
            self._set("chase" if robot else "wide", robot)
            return True
        if name == "scenario":
            sid = args.get("id")
            if sid == "flyby":
                self.main = None
                self._set("flyby")
                return True
            if sid in MAIN:
                self.main = MAIN[sid]
                self._set("wide")
                return True
            return False
        if name == "fleet":
            self.fleet(args)
        return False

    def fleet(self, args):
        for r in args.get("robots") or []:
            self.paths[r["id"]] = [tuple(p) for p in r.get("path") or []]
            p = r.get("pose")
            if p:
                self.fleet_poses[r["id"]] = (p["x"], p["y"], p["z"], p.get("yaw", 0.0))

    def _poses(self):
        got = self.poses() if self.poses is not None else None
        return got or self.fleet_poses

    def at(self, t):
        dt = 0.0 if self.last is None else min(max(t - self.last, 0.0), 0.5)
        self.last = t
        if self.mode == "flyby":
            return None
        poses = self._poses()
        if self.mode == "chase" and self.robot in poses:
            return self._chase(poses[self.robot], dt, t)
        if not poses:
            return None
        return self._wide(poses, dt)

    def _ray(self, pivot, yaw, back):
        eye = (pivot[0] - math.cos(yaw) * back, pivot[1] - math.sin(yaw) * back, pivot[2] - PIVOT_Z + HEIGHT)
        if self.blocked is None:
            return back
        span = math.dist(pivot, eye)
        hit = self.blocked(pivot, eye)
        if hit is None or hit >= span:
            return back
        return max(MIN_BACK, back * (hit - 0.3) / span)

    def _pick(self, pivot, yaw):
        cur = self.s["off"]
        clear = lambda o: self._ray(pivot, yaw + math.radians(o), BACK) >= BACK * 0.95
        if clear(cur) and (cur == 0.0 or not clear(0.0)):
            return cur
        return next((o for o in OFFSETS if clear(o)), cur)

    def _chase(self, pose, dt, t):
        x, y, z, yaw = pose
        s = self.s
        key = ("chase", self.robot)
        if self.key != key or math.dist(s["p"], (x, y, z)) > CUT_M:
            self.key = key
            s.clear()
            s.update(p=[x, y, z], pv=[0.0, 0.0, 0.0], off=0.0, yaw=yaw, yv=0.0, d=BACK, dv=0.0, check=-math.inf)
            s["off"] = self._pick((x, y, z + PIVOT_Z), yaw)
            s["yaw"] = yaw + math.radians(s["off"])
            s["d"] = self._ray((x, y, z + PIVOT_Z), s["yaw"], BACK)
            s["check"] = t + CHECK_S
        elif t >= s["check"]:
            s["check"] = t + CHECK_S
            s["off"] = self._pick((x, y, z + PIVOT_Z), yaw)
        for i, v in enumerate((x, y, z)):
            s["p"][i], s["pv"][i] = spring(s["p"][i], s["pv"][i], v, W_POS, dt)
        goal = yaw + math.radians(s["off"])
        e, s["yv"] = spring(wrap(s["yaw"] - goal), s["yv"], 0.0, W_YAW, dt)
        s["yaw"] = goal + e
        px, py, pz = s["p"]
        want = self._ray((px, py, pz + PIVOT_Z), s["yaw"], BACK)
        if want < s["d"]:
            s["d"], s["dv"] = want, 0.0
        else:
            s["d"], s["dv"] = spring(s["d"], s["dv"], want, W_OUT, dt)
        d = s["d"]
        c, n = math.cos(s["yaw"]), math.sin(s["yaw"])
        ez = pz + HEIGHT
        top = ceiling_of(deck_of(pz))
        if top is not None:
            ez = min(ez, top - 0.6)
        eye = (px - c * d, py - n * d, ez)
        look = (px + c * AHEAD, py + n * AHEAD, pz + LOOK_Z)
        return eye, look, CHASE_FOCAL

    def _deck(self, poses):
        if self.main in poses:
            return deck_of(poses[self.main][2])
        decks = [deck_of(p[2]) for p in poses.values()]
        return max(set(decks), key=lambda d: (decks.count(d), d))

    def _wide(self, poses, dt):
        deck = self._deck(poses)
        pts = [(p[0], p[1], deck) for p in poses.values() if deck_of(p[2]) == deck]
        for path in self.paths.values():
            pts += [(p[0], p[1], deck) for p in path if abs(p[2] - deck) < 1.5]
        center, d, az, el = wide_shot(pts, deck)
        goal = list(center) + [d, el]
        s = self.s
        key = ("wide", deck)
        if self.key != key:
            self.key = key
            s.clear()
            s.update(w=list(goal), wv=[0.0] * len(goal))
        for i, g in enumerate(goal):
            s["w"][i], s["wv"][i] = spring(s["w"][i], s["wv"][i], g, W_WIDE, dt)
        cx, cy, cz, d, el = s["w"]
        b = view_dir(az, el)
        eye = (cx + d * b[0], cy + d * b[1], cz + d * b[2])
        return eye, (cx, cy, cz), WIDE_FOCAL


def robot_poses(stage, root=ROBOTS_ROOT):
    from pxr import Usd, UsdGeom

    def poses():
        prim = stage.GetPrimAtPath(root)
        if not prim.IsValid():
            return {}
        out = {}
        for p in prim.GetChildren():
            m = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            o = m.ExtractTranslation()
            out[p.GetName()] = (o[0], o[1], o[2] - ROBOT_Z, math.atan2(m[0][1], m[0][0]))
        return out

    return poses


def physx_ray():
    try:
        import carb
        from omni.physx import get_physx_scene_query_interface

        query = get_physx_scene_query_interface()
    except Exception as exc:
        print(f"rig sim: hero camera without scene queries ({exc})", flush=True)
        return None

    def blocked(a, b):
        span = math.dist(a, b)
        if span < 1e-6:
            return None
        d = tuple((q - p) / span for p, q in zip(a, b))
        hit = query.raycast_closest(carb.Float3(*a), carb.Float3(*d), span)
        return float(hit["distance"]) if hit and hit.get("hit") else None

    return blocked
