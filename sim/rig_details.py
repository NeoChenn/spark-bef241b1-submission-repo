"""Add inspection detail to the full rig: pipes, valves, gauges, props, markings.

Import after SimulationApp has started. Gauge readings are saved with their
positions, so renders can serve as labeled data for gauge-reading work.
"""

from __future__ import annotations

import math
import random

import numpy as np
from pxr import Gf, Sdf, UsdGeom, UsdShade

from scene_builder import GENERATED, SceneBuilder, rel

TOP_DECK_Z = 42.5
GAUGE_RANGE_BAR = 16.0
HELIDECK_CIRCLE_RADIUS = 8.0

# Top deck loop in metres. route_blockages re-checks it against the geometry at build time.
PATROL_ROUTE = [
    (-21.0, 20.5), (-10.0, 20.5), (-10.0, 11.0), (-1.0, 11.0), (-1.0, -4.5), (24.0, -4.5), (24.0, -9.0),
    (9.0, -9.0), (9.0, -4.5), (-1.0, -4.5), (-1.0, 11.0), (-21.0, 11.0), (-21.0, 20.5),
]


def image_material(stage, name: str, image, roughness: float = 0.35) -> UsdShade.Material:
    path = f"/World/Looks/{name}"
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    reader = UsdShade.Shader.Define(stage, f"{path}/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    node = UsdShade.Shader.Define(stage, f"{path}/image")
    node.CreateIdAttr("UsdUVTexture")
    node.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(image))
    node.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    )
    node.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        node.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    material.CreateSurfaceOutput().ConnectToSource(shader.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return material


def face_disc(b: SceneBuilder, parent, name, center, facing, radius, material):
    """Round disc facing a horizontal direction, mapped so +Z is up on the texture."""
    normal = np.array((facing[0], facing[1], 0.0))
    right = np.array((-facing[1], facing[0], 0.0))
    up = np.array((0.0, 0.0, 1.0))
    origin = np.array(center, dtype=float)
    segments = 36
    points, normals, uvs, indices = [tuple(origin)], [tuple(normal)], [(0.5, 0.5)], []
    for step in range(segments):
        angle = math.tau * step / segments
        cos, sin = math.cos(angle), math.sin(angle)
        points.append(tuple(origin + (right * cos + up * sin) * radius))
        normals.append(tuple(normal))
        uvs.append((0.5 + cos / 2, 0.5 + sin / 2))
        indices.extend((0, step + 1, (step + 1) % segments + 1))
    b._mesh(b.unique(parent, name), points, normals, uvs, [3] * segments, indices, material, None)


def flat_ring(b: SceneBuilder, path, center, inner, outer, material):
    segments = 96
    points, normals, uvs, indices, counts = [], [], [], [], []
    for step in range(segments + 1):
        angle = math.tau * step / segments
        for radius in (inner, outer):
            points.append((center[0] + math.cos(angle) * radius, center[1] + math.sin(angle) * radius, center[2]))
            normals.append((0, 0, 1))
            uvs.append((step / segments * 40, radius))
    for step in range(segments):
        a = step * 2
        indices.extend((a, a + 1, a + 3, a + 2))
        counts.append(4)
    b._mesh(path, points, normals, uvs, counts, indices, material, None)


def add_helideck_markings(b: SceneBuilder, center) -> None:
    white = b.plain("marking_white", (0.92, 0.92, 0.9), 0.6)
    yellow = b.plain("marking_yellow", (0.92, 0.72, 0.05), 0.6)
    x, y, z = center[0], center[1], center[2] + 0.015
    flat_ring(b, "/World/Details/helideck_circle", (x, y, z), HELIDECK_CIRCLE_RADIUS - 0.5, HELIDECK_CIRCLE_RADIUS, yellow)
    for offset_x, offset_y, size_x, size_y in ((-1.7, 0, 0.9, 6.0), (1.7, 0, 0.9, 6.0), (0, 0, 2.5, 0.9)):
        b.box("/World/Details", "helideck_h", (x + offset_x, y + offset_y, z), (size_x, size_y, 0.02), white, collide=None)


def add_windows(b: SceneBuilder, low, high, name: str, glass) -> None:
    width, height, spacing, floor = 1.6, 1.15, 3.0, 3.2
    points, normals, uvs, indices = [], [], [], []

    def add(corner, along, normal):
        base = len(points)
        for du, dv in ((0, 0), (width, 0), (width, height), (0, height)):
            points.append(tuple(np.array(corner) + np.array(along) * du + np.array((0, 0, dv))))
            normals.append(normal)
            uvs.append((du / width, dv / height))
        indices.extend((base, base + 1, base + 2, base + 3))

    levels = np.arange(low[2] + 1.5, high[2] - 1.6, floor)
    for z in levels:
        for y, normal in ((low[1] - 0.03, (0, -1, 0)), (high[1] + 0.03, (0, 1, 0))):
            for x in np.arange(low[0] + 1.2, high[0] - 2.4, spacing):
                add((x, y, z), (1, 0, 0), normal)
        for x, normal in ((low[0] - 0.03, (-1, 0, 0)), (high[0] + 0.03, (1, 0, 0))):
            for y in np.arange(low[1] + 1.2, high[1] - 2.4, spacing):
                add((x, y, z), (0, 1, 0), normal)
    if points:
        b._mesh(f"/World/Details/windows_{name}", points, normals, uvs, [4] * (len(points) // 4), indices, glass, None)


def _inside_any(triangles: np.ndarray, point: np.ndarray, margin: float) -> bool:
    """True if a square around ``point`` is covered by the 2D triangles."""
    for offset in ((-margin, -margin), (margin, -margin), (margin, margin), (-margin, margin)):
        target = point + np.array(offset)
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        sides = [
            (q[:, 0] - p[:, 0]) * (target[1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (target[0] - p[:, 0])
            for p, q in ((a, b), (b, c), (c, a))
        ]
        positive = (sides[0] >= 0) & (sides[1] >= 0) & (sides[2] >= 0)
        negative = (sides[0] <= 0) & (sides[1] <= 0) & (sides[2] <= 0)
        if not (positive | negative).any():
            return False
    return True


def add_facade_windows(b: SceneBuilder, triangles: np.ndarray, normals: np.ndarray, glass) -> int:
    """Add window rows to large flat walls. Triangles and normals are in world space."""
    planes: dict[tuple, list[np.ndarray]] = {}
    for triangle, normal in zip(triangles, normals):
        axis = 0 if abs(normal[0]) > abs(normal[1]) else 1
        key = (axis, normal[axis] > 0, round(float(triangle[0][axis]), 1))
        planes.setdefault(key, []).append(triangle)

    points, face_normals, uvs, indices = [], [], [], []
    width, height, spacing, floor = 1.5, 1.1, 4.5, 3.6
    for (axis, positive, offset), group in planes.items():
        corners = np.concatenate(group)
        other = 1 - axis
        flat_triangles = np.stack(group)[:, :, [other, 2]]
        along_low, along_high = corners[:, other].min(), corners[:, other].max()
        z_low, z_high = corners[:, 2].min(), corners[:, 2].max()
        if z_high - z_low < 4.5 or along_high - along_low < 6:
            continue
        outward = 0.03 if positive else -0.03
        normal = [0.0, 0.0, 0.0]
        normal[axis] = 1.0 if positive else -1.0
        for z in np.arange(z_low + 1.6, z_high - 1.8, floor):
            for along in np.arange(along_low + 1.5, along_high - 2.8, spacing):
                window = np.array(
                    [(along, z), (along + width, z), (along + width, z + height), (along, z + height)]
                )
                if not all(_inside_any(flat_triangles, point, margin=0.25) for point in window):
                    continue
                base = len(points)
                for du, dv in ((0, 0), (width, 0), (width, height), (0, height)):
                    corner = [0.0, 0.0, z + dv]
                    corner[axis] = offset + outward
                    corner[other] = along + du
                    points.append(tuple(corner))
                    face_normals.append(tuple(normal))
                    uvs.append((du / width, dv / height))
                indices.extend((base, base + 1, base + 2, base + 3))
    if points:
        b._mesh(
            "/World/Details/facade_windows", points, face_normals, uvs,
            [4] * (len(points) // 4), indices, glass, None,
        )
    return len(points) // 4


def add_gauge(b, dial, steel, black, red, name, pipe_top, facing, value):
    """Stem-mounted gauge facing a walkway. ``value`` runs from 0 to 1 across the dial."""
    parent = "/World/Details/Gauges"
    x, y, z = pipe_top
    head_z = z + 0.34
    b.tube(parent, "stem", (x, y, z), (x, y, head_z), 0.012, steel, segments=10)
    back = (x - facing[0] * 0.03, y - facing[1] * 0.03, head_z)
    front = (x + facing[0] * 0.03, y + facing[1] * 0.03, head_z)
    b.tube(parent, "body", back, front, 0.085, black, segments=28)
    face_disc(b, parent, "dial", front, facing, 0.085, dial)

    # Dial runs clockwise from lower left (-135 deg) to lower right (+135 deg), seen from the front.
    angle = math.radians(135.0 - 270.0 * value)
    right = (-facing[1], facing[0])
    tip_right, tip_up = math.cos(angle) * 0.068, math.sin(angle) * 0.068
    lift = 0.006
    start = (front[0] + facing[0] * lift, front[1] + facing[1] * lift, head_z)
    end = (start[0] - right[0] * tip_right, start[1] - right[1] * tip_right, head_z + tip_up)
    b.tube(parent, "needle", start, end, 0.0035, red, segments=6)
    return {
        "id": name,
        "type": "gauge",
        "position_m": [round(front[0], 3), round(front[1], 3), round(head_z, 3)],
        "facing": [facing[0], facing[1], 0],
        "reading_bar": round(value * GAUGE_RANGE_BAR, 2),
        "range_bar": [0.0, GAUGE_RANGE_BAR],
    }


def add_splash_zone(b: SceneBuilder, leg_triangles: np.ndarray) -> None:
    """Dark tide band where the four concrete legs meet the sea."""
    band = b.plain("splash_zone", (0.06, 0.08, 0.07), 0.7)
    points = leg_triangles.reshape(-1, 3)
    middle = (points[:, :2].min(0) + points[:, :2].max(0)) / 2
    for east in (False, True):
        for north in (False, True):
            leg = points[((points[:, 0] > middle[0]) == east) & ((points[:, 1] > middle[1]) == north)]
            if not len(leg):
                continue
            top_z, bottom_z = leg[:, 2].max(), leg[:, 2].min()
            top = leg[leg[:, 2] > top_z - 0.5, :2]
            bottom = leg[leg[:, 2] < bottom_z + 0.5, :2]
            top_center = (top.min(0) + top.max(0)) / 2
            bottom_center = (bottom.min(0) + bottom.max(0)) / 2
            top_radius = float((top.max(0) - top.min(0)).max()) / 2
            bottom_radius = float((bottom.max(0) - bottom.min(0)).max()) / 2
            waterline = top_z / (top_z - bottom_z)
            center = top_center + (bottom_center - top_center) * waterline
            radius = top_radius + (bottom_radius - top_radius) * waterline + 0.05
            b.tube(
                "/World/Details", "splash_band",
                (center[0], center[1], -0.5), (center[0], center[1], 2.4), radius, band, segments=48,
            )


def add_process_area(b: SceneBuilder, stage, seed: int) -> list[dict]:
    rng = random.Random(seed)
    blue = b.plain("line_water", (0.09, 0.3, 0.55), 0.42, 0.2)
    yellow = b.plain("line_gas", (0.86, 0.66, 0.08), 0.42, 0.1)
    orange = b.plain("line_process", (0.72, 0.27, 0.06), 0.42, 0.2)
    steel = b.plain("support_steel", (0.3, 0.33, 0.36), 0.5, 0.5)
    black = b.plain("gauge_body", (0.04, 0.04, 0.045), 0.4, 0.3)
    red = b.plain("gauge_needle", (0.75, 0.05, 0.04), 0.4)
    dial = image_material(stage, "gauge_dial", GENERATED / "gauge_dial.png")
    deck = TOP_DECK_Z
    points: list[dict] = []

    def line(y, z, radius, material, x_start, x_end):
        b.tube("/World/Details/Pipes", "line", (x_start, y, z), (x_end, y, z), radius, material, collide="none")
        for end in (x_start, x_end):
            b.flange("/World/Details/Pipes", end, y, z, radius, material)

    def support(x, y, z, depth):
        b.box("/World/Details/Pipes", "support", (x, y, (deck + z) / 2), (0.14, depth, z - deck), steel)
        b.box("/World/Details/Pipes", "support_base", (x, y, deck + 0.02), (0.4, depth + 0.2, 0.04), steel)

    line(24.3, 43.6, 0.16, blue, -25.0, 0.5)
    line(23.4, 43.35, 0.11, yellow, -25.0, 0.5)
    line(-10.7, 43.4, 0.12, orange, 15.5, 35.0)
    for x in (-23.0, -18.0, -13.0, -8.0, -3.0):
        support(x, 23.85, 43.2, 1.6)
    for x in (17.0, 23.0, 29.0, 34.0):
        support(x, -10.7, 43.25, 0.5)

    for index, (x, y, z, scale, part) in enumerate(
        (
            (-20.5, 24.3, 43.6, 3.2, "pipe_valve_large_metal"),
            (-5.5, 24.3, 43.6, 3.2, "pipe_valve_large_metal"),
            (-15.5, 23.4, 43.35, 2.2, "pipe_valve_small_1_metal"),
            (-1.0, 23.4, 43.35, 2.2, "pipe_valve_small_1_metal"),
            (20.0, -10.7, 43.4, 2.4, "pipe_valve_small_1_metal"),
            (31.5, -10.7, 43.4, 2.4, "pipe_valve_small_1_metal"),
        ),
        start=1,
    ):
        b.asset("/World/Details/Pipes", "modular_pipes", (x, y, z), rotate_y=90, part=part, scale=scale, on_deck=False)
        points.append({"id": f"valve_{index:02d}", "type": "valve", "position_m": [x, y, z]})

    for index, (x, y, z, facing) in enumerate(
        (
            (-23.5, 23.4, 43.46, (0, -1)),
            (-17.5, 23.4, 43.46, (0, -1)),
            (-12.0, 23.4, 43.46, (0, -1)),
            (-6.5, 23.4, 43.46, (0, -1)),
            (17.8, -10.7, 43.52, (0, 1)),
            (24.5, -10.7, 43.52, (0, 1)),
            (33.0, -10.7, 43.52, (0, 1)),
        ),
        start=1,
    ):
        value = rng.uniform(0.2, 0.8)
        points.append(add_gauge(b, dial, steel, black, red, f"gauge_{index:02d}", (x, y, z), facing, value))
    return points


def add_props(b: SceneBuilder) -> None:
    deck = TOP_DECK_Z
    for asset, x, y, rotate in (
        ("portable_generator", -21.5, 3.5, 20),
        ("utility_box_02", -15.5, 2.0, 0),
        ("utility_box_01", -14.3, 2.0, 0),
        ("power_box_01", -12.6, 2.2, 0),
        ("worn_metal_rack", -24.8, 8.5, 90),
        ("metal_toolbox", -18.5, 6.5, 25),
        ("portable_welding_cart", -24.0, 13.5, 75),
        ("propane_tank", -24.8, 16.5, 0),
        ("propane_tank", -24.2, 17.0, 40),
        ("barrel_03", -24.5, 5.2, 0),
        ("barrel_03", -23.8, 4.2, 60),
        ("korean_fire_extinguisher_01", -10.8, 3.0, 200),
        ("korean_fire_extinguisher_01", 5.0, -3.8, 120),
        ("barrel_03", 33.0, 22.0, 15),
        ("barrel_03", 33.8, 20.9, 80),
    ):
        b.asset("/World/Details/Props", asset, (x, y, deck), rotate_z=rotate)


def add_walkway_lines(b: SceneBuilder, route) -> None:
    yellow = b.plain("walkway_line", (0.9, 0.68, 0.05), 0.6)
    half = 0.95
    z = TOP_DECK_Z + 0.008
    drawn = set()
    for start, end in zip(route, route[1:]):
        key = tuple(sorted((start, end)))
        if key in drawn:
            continue
        drawn.add(key)
        center = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        length = math.dist(start, end) + 2 * half
        if start[1] == end[1]:
            for offset in (-half, half):
                b.box("/World/Details/Walkway", "line", (center[0], center[1] + offset, z), (length, 0.1, 0.01), yellow, collide=None)
        elif start[0] == end[0]:
            for offset in (-half, half):
                b.box("/World/Details/Walkway", "line", (center[0] + offset, center[1], z), (0.1, length, 0.01), yellow, collide=None)


def route_blockages(world_triangles: np.ndarray, route, clearance: float = 0.45) -> list[list[float]]:
    """Return route samples within ``clearance`` of geometry at robot height."""
    low, high = TOP_DECK_Z + 0.15, TOP_DECK_Z + 0.9
    z_low = world_triangles[:, :, 2].min(1)
    z_high = world_triangles[:, :, 2].max(1)
    nearby = world_triangles[(z_low < high) & (z_high > low)]
    rng = np.random.default_rng(0)
    samples = []
    for triangle in nearby:
        area = np.linalg.norm(np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])) / 2
        weights = rng.dirichlet((1, 1, 1), min(1500, max(4, int(area * 12))))
        samples.append(weights @ triangle)
    cloud = np.concatenate(samples)
    cloud = cloud[(cloud[:, 2] > low) & (cloud[:, 2] < high)][:, :2]
    blocked = []
    for start, end in zip(route, route[1:]):
        steps = max(2, int(math.dist(start, end) / 0.5))
        for t in np.linspace(0, 1, steps):
            point = np.array(start) + (np.array(end) - np.array(start)) * t
            if (np.abs(cloud - point).max(axis=1) < clearance).any():
                blocked.append([round(float(point[0]), 1), round(float(point[1]), 1)])
    return blocked
