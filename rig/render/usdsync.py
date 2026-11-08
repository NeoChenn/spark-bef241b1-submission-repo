import numpy as np
from pxr import Gf, Sdf, Tf, Usd

from rig.render.state import StateLog, under

SKIP_PREFIXES = ("/Render", "/OmniverseKit", "/OmniKit", "/RigStream", "/RigRender", "/World/Cinematic/ocean")
SKIP_NAMES = ("points", "normals", "velocities", "accelerations", "extent")
MAX_ARRAY = 20000
MAX_SPEC = 256 * 1024
DIGITS = 6


def _round(x):
    return round(float(x), DIGITS)


def encode_value(v):
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return _round(v)
    if isinstance(v, Sdf.AssetPath):
        return v.path
    if isinstance(v, Sdf.Path):
        return str(v)
    if isinstance(v, (Gf.Quatf, Gf.Quatd, Gf.Quath)):
        return [_round(v.GetReal()), *(_round(c) for c in v.GetImaginary())]
    if isinstance(v, (Gf.Matrix4d, Gf.Matrix4f, Gf.Matrix3d, Gf.Matrix3f, Gf.Matrix2d, Gf.Matrix2f)):
        return np.round(np.asarray(v, np.float64), DIGITS).reshape(-1).tolist()
    if hasattr(v, "__len__"):
        if len(v) > MAX_ARRAY:
            raise ValueError("array too large")
        if len(v) and isinstance(v[0], (str, Sdf.AssetPath, Sdf.Path)):
            return [encode_value(x) for x in v]
        arr = np.asarray(v)
        if arr.dtype.kind in "fc":
            return np.round(arr.astype(np.float64), DIGITS).tolist()
        return arr.tolist()
    return v


def decode_value(kind: str, value):
    vt = Sdf.ValueTypeNames.Find(kind)
    cls = vt.type.pythonClass
    if value is None:
        return None
    if kind in ("asset",):
        return Sdf.AssetPath(value)
    if kind == "asset[]":
        return Sdf.AssetPathArray([Sdf.AssetPath(x) for x in value])
    if cls is None:
        return value
    if vt.isArray:
        return cls([tuple(x) for x in value]) if value and isinstance(value[0], list) else cls(value)
    if kind.startswith("quat"):
        return cls(value[0], *value[1:])
    if isinstance(value, list):
        return cls(*value)
    return cls(value)


def skipped(path: str, prefixes=SKIP_PREFIXES) -> bool:
    return any(under(path, p) for p in prefixes)


def export_spec(stage, path: str) -> str | None:
    layer = stage.GetEditTarget().GetLayer()
    if not layer.GetPrimAtPath(path):
        return None
    anon = Sdf.Layer.CreateAnonymous(".usda")
    Sdf.CreatePrimInLayer(anon, path)
    Sdf.CopySpec(layer, path, anon, path)
    text = anon.ExportToString()
    return text if len(text) <= MAX_SPEC else None


class StageCapture:
    def __init__(self, stage, skip_prefixes=SKIP_PREFIXES, skip_names=SKIP_NAMES) -> None:
        self.stage = stage
        self.skip_prefixes, self.skip_names = tuple(skip_prefixes), set(skip_names)
        self.log = StateLog()
        self.dropped = 0
        self._props: set[str] = set()
        self._prims: set[str] = set()
        self._key = Tf.Notice.Register(Usd.Notice.ObjectsChanged, self._on_change, stage)

    def close(self) -> None:
        self._key.Revoke()

    def _on_change(self, notice, stage) -> None:
        for p in notice.GetResyncedPaths():
            (self._props if p.IsPropertyPath() else self._prims).add(str(p))
        for p in notice.GetChangedInfoOnlyPaths():
            if p.IsPropertyPath():
                self._props.add(str(p))

    def _prop(self, path: str):
        sp = Sdf.Path(path)
        prop = self.stage.GetPropertyAtPath(sp)
        name = sp.name
        if name in self.skip_names or skipped(path, self.skip_prefixes):
            return None
        prim_path = str(sp.GetPrimPath())
        if not prop:
            return None
        if isinstance(prop, Usd.Relationship):
            return [prim_path, name, "rel", [str(t) for t in prop.GetTargets()]]
        try:
            value = encode_value(prop.Get())
        except ValueError:
            self.dropped += 1
            return None
        return [prim_path, name, str(prop.GetTypeName()), value]

    def collect(self) -> dict:
        prims, props = sorted(self._prims), sorted(self._props)
        self._prims, self._props = set(), set()
        specs = []
        for path in prims:
            if path == "/" or skipped(path, self.skip_prefixes) or any(under(path, s[0]) for s in specs):
                continue
            prim = self.stage.GetPrimAtPath(path)
            if not prim:
                specs.append([path, None])
                continue
            text = export_spec(self.stage, path)
            if text is None:
                self.dropped += 1
                continue
            specs.append([path, text])
        attrs = []
        for path in props:
            if any(under(path.split(".")[0], s[0]) for s in specs):
                continue
            entry = self._prop(path)
            if entry is not None:
                attrs.append(entry)
        delta = {"specs": specs, "attrs": attrs}
        self.log.merge(delta)
        return delta

    def keyframe(self) -> dict:
        return {k: v for k, v in self.log.message(True).items() if k in ("specs", "attrs", "key")}


class StageApplier:
    def __init__(self, stage, layer=None) -> None:
        self.stage = stage
        self.layer = layer or stage.GetSessionLayer()
        self._specs: dict[tuple[str, str], object] = {}
        self.applied = 0
        self.failed = 0

    def _attr_spec(self, path: str, name: str, kind: str):
        key = (path, name)
        spec = self._specs.get(key)
        if spec is not None and not spec.expired:
            return spec
        prim_spec = Sdf.CreatePrimInLayer(self.layer, path)
        if kind == "rel":
            spec = prim_spec.relationships.get(name) or Sdf.RelationshipSpec(prim_spec, name, custom=False)
        else:
            spec = prim_spec.attributes.get(name) or Sdf.AttributeSpec(prim_spec, name, Sdf.ValueTypeNames.Find(kind))
        self._specs[key] = spec
        return spec

    def _spec(self, path: str, usda: str | None) -> None:
        self._specs = {k: v for k, v in self._specs.items() if not under(k[0], path)}
        if usda is None:
            Sdf.CreatePrimInLayer(self.layer, path).active = False
            return
        src = Sdf.Layer.CreateAnonymous(".usda")
        src.ImportFromString(usda)
        Sdf.CreatePrimInLayer(self.layer, path)
        Sdf.CopySpec(src, path, self.layer, path)
        self.layer.GetPrimAtPath(path).active = True

    def apply(self, msg: dict) -> int:
        n = 0
        with Sdf.ChangeBlock():
            for path, usda in msg.get("specs") or ():
                self._spec(path, usda)
                n += 1
            for path, name, kind, value in msg.get("attrs") or ():
                try:
                    spec = self._attr_spec(path, name, kind)
                    if kind == "rel":
                        spec.targetPathList.explicitItems = [Sdf.Path(t) for t in value]
                    else:
                        spec.default = decode_value(kind, value)
                    n += 1
                except Exception:
                    self.failed += 1
        self.applied += n
        return n
