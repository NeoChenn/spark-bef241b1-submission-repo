import copy

DECK_Z = 42.5
ZONE_TRANSLATION = (18.75, -5.75, DECK_Z)
DECK_HALF = (8.0, 5.0)
AXIS_DISTANCE = 0.5
TRUNK_HEIGHT = 0.32
SEGMENT_LENGTH = 0.8


def to_world(p, t=ZONE_TRANSLATION) -> tuple:
    return tuple(float(a) + float(b) for a, b in zip(p, t))


def zone_footprint(t=ZONE_TRANSLATION, half=DECK_HALF) -> tuple:
    return (t[0] - half[0], t[0] + half[0], t[1] - half[1], t[1] + half[1])


def _r(v) -> list:
    return [round(float(c), 4) for c in v]


def merged_meta(deck_meta: dict, t=ZONE_TRANSLATION, cameras: dict | None = None, extra: dict | None = None) -> dict:
    out = copy.deepcopy(deck_meta)
    pipe, damage = deck_meta["main_pipe"], deck_meta["damage"]
    dx = float(damage["position_m"][0])
    origin = to_world((dx, pipe["y_m"], pipe["z_m"]), t)
    robot = to_world((dx, pipe["y_m"] + AXIS_DISTANCE, TRUNK_HEIGHT), t)
    out["units"] = "metres, Z up. Top-level main_pipe/damage/work_zone_m are zone-local (deck at z=0, same as rig_deck_meta.json); world = local + zone_to_world.translation"
    out["zone_to_world"] = {"translation": _r(t), "yaw": 0.0}
    out["world"] = {
        "deck_z_m": float(t[2]),
        "main_pipe": {
            "axis": [1.0, 0.0, 0.0],
            "ref": [0.0, 1.0, 0.0],
            "radius_m": float(pipe["radius_m"]),
            "y_m": round(float(pipe["y_m"]) + t[1], 4),
            "z_m": round(float(pipe["z_m"]) + t[2], 4),
            "x_range_m": _r(v + t[0] for v in pipe["x_range_m"]),
        },
        "segment_origin_m": _r(origin),
        "segment_length_m": SEGMENT_LENGTH,
        "damage": {**damage, "position_m": _r(to_world(damage["position_m"], t))},
        "robot_spawn": {"position_m": _r(robot), "yaw_rad": 0.0, "trunk_height_above_deck_m": TRUNK_HEIGHT},
        "work_zone_m": {
            "x": _r(v + t[0] for v in deck_meta["work_zone_m"]["x"]),
            "y": _r(v + t[1] for v in deck_meta["work_zone_m"]["y"]),
        },
        "footprint_m": _r(zone_footprint(t)),
    }
    if cameras is not None:
        out["cameras"] = cameras
    if extra:
        out.update(extra)
    return out
