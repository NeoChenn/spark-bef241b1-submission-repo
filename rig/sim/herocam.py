import math

DECKS = (20.0, 31.25, 42.5)
ROBOTS_ROOT = "/World/Obstacles/Robots"
ROBOT_Z = 0.45
APERTURE = 20.955
ASPECT = 16 / 9

FOCAL = {"deck": 24.0, "zoom": 28.0, "obstacle": 24.0}
DIST = {"deck": 20.0, "zoom": 4.0, "obstacle": 14.0}
ELEV = {"deck": 30.0, "zoom": 35.0, "obstacle": 30.0}
AZ = (-35.0, 55.0, 145.0, -125.0)
LOOK_Z = 0.5
DECK_AT = {"gas": (13.5, -8.5, 42.5), "patrol": (11.0, -6.0, 42.5), "repair": (18.75, -5.75, 42.5)}


def view_dir(az, el):
    a, e = math.radians(az), math.radians(el)
    return (math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e))


def deck_of(z):
    return max((d for d in DECKS if d <= z + 0.6), default=DECKS[0])


def ceiling_of(deck):
    return min((d for d in DECKS if d > deck + 1.0), default=None)


def fixed_shot(at, kind, blocked=None, lens=None):
    lens = lens or {}
    look = (at[0], at[1], at[2] + LOOK_Z)
    d, el, focal = lens.get("dist", DIST[kind]), lens.get("elev", ELEV[kind]), lens.get("focal", FOCAL[kind])
    if lens.get("eye"):
        return tuple(float(v) for v in lens["eye"]), look, focal
    top = ceiling_of(deck_of(at[2]))
    if top is not None:
        el = math.degrees(math.asin(max(min((top - 1.0 - look[2]) / d, math.sin(math.radians(el))), 0.05)))
    best = None
    for az in tuple(lens.get("az") or ()) + AZ:
        b = view_dir(az, el)
        eye = (look[0] + d * b[0], look[1] + d * b[1], look[2] + d * b[2])
        hit = blocked(look, eye) if blocked is not None else None
        if hit is None or hit >= d:
            return eye, look, focal
        if best is None or hit > best[0]:
            best = (hit, b)
    k = max(best[0] - 0.5, 1.5)
    b = best[1]
    return (look[0] + k * b[0], look[1] + k * b[1], look[2] + k * b[2]), look, focal


class HeroCam:
    def __init__(self):
        self.mode, self.view, self.sid = "flyby", None, None
        self.at_xyz, self.shot = None, None
        self.lens = None
        self.poses = None
        self.blocked = None

    def meta(self):
        return {"mode": self.mode, "robot": None, "view": self.view}

    def _cut(self, mode, at=None, view=None, lens=None):
        self.mode, self.view, self.at_xyz, self.shot, self.lens = mode, view, at, None, lens

    def event(self, name, args):
        args = args or {}
        if name == "camera":
            mode = args.get("mode")
            at = args.get("at")
            if mode in ("flyby", "rig"):
                self._cut(mode)
                return True
            if mode == "deck" and at is None and self.sid in DECK_AT:
                at = DECK_AT[self.sid]
            if mode not in FOCAL or at is None:
                return False
            self._cut(mode, tuple(float(v) for v in at[:3]), args.get("view"), args.get("lens"))
            return True
        if name == "scenario":
            sid = args.get("id")
            if sid == "flyby":
                self.sid = None
                self._cut("flyby")
                return True
            if sid in DECK_AT:
                self.sid = sid
                self._cut("deck", DECK_AT[sid])
                return True
        return False

    def at(self, t):
        if self.at_xyz is None or self.mode not in FOCAL:
            return None
        if self.shot is None:
            self.shot = fixed_shot(self.at_xyz, self.mode, self.blocked, self.lens)
        return self.shot


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


def first_hit(*blockers):
    live = [b for b in blockers if b is not None]

    def blocked(a, b):
        hits = [h for h in (f(a, b) for f in live) if h is not None]
        return min(hits) if hits else None

    return blocked


def mesh_ray(build, cell=40.0):
    cache = {}

    def blocked(a, b):
        span = math.dist(a, b)
        if span < 1e-6:
            return None
        key = tuple(round(c / cell) for c in a)
        if key not in cache:
            cache[key] = build(tuple(k * cell for k in key))
        caster = cache[key]
        if caster is None:
            return None
        d = [(q - p) / span for p, q in zip(a, b)]
        t = float(caster.cast(a, [d], span)[0])
        return t if 0.0 < t < span else None

    return blocked


def stage_ray(stage, half=(60.0, 60.0, 12.0)):
    def build(center):
        try:
            from rig.sim.sensors import stage_meshes
            from rig.views.raycast import WarpCaster

            verts, tris = stage_meshes(stage, center, half=half)
            return WarpCaster(verts, tris) if len(tris) else None
        except Exception as exc:
            print(f"rig sim: hero camera mesh rays off ({exc})", flush=True)
            return None

    return mesh_ray(build)
