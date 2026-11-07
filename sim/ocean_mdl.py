"""Generate the OceanWater MDL with a baked Gerstner wave set, so shading normals match the displaced mesh."""

from __future__ import annotations

from collections.abc import Sequence

from sim.ocean_waves import Waves

MODULE = "OceanWater"

TEMPLATE = """mdl 1.7;

import ::df::*;
import ::state::*;
import ::math::*;
import ::base::*;
import ::anno::*;

const int WAVE_COUNT = @WAVE_COUNT@;
const float4[WAVE_COUNT] WAVE_A = float4[](@WAVE_A@);
const float4[WAVE_COUNT] WAVE_B = float4[](@WAVE_B@);
const int LEG_COUNT = @LEG_COUNT@;
const float3[@LEG_SLOTS@] LEGS = float3[](@LEGS@);

struct ocean_sample {
    float3 n;
    float h;
    float jac;
    float lost;
};

ocean_sample ocean_eval(float2 p, float t, float footprint)
{
    float sx = 0.0, sy = 0.0, sz = 0.0, jxx = 0.0, jyy = 0.0, jxy = 0.0, h = 0.0, lost = 0.0;
    for (int i = 0; i < WAVE_COUNT; ++i) {
        float4 a = WAVE_A[i];
        float4 b = WAVE_B[i];
        float w = math::smoothstep(2.0, 4.0, 6.2831853 / a.z / footprint);
        float amp = a.w * w;
        float th = a.z * (a.x * p.x + a.y * p.y) - (b.x * t - b.y);
        float s = math::sin(th);
        float c = math::cos(th);
        float ka = a.z * amp;
        sx += a.x * ka * c;
        sy += a.y * ka * c;
        float qs = b.z * ka * s;
        sz += qs;
        jxx += qs * a.x * a.x;
        jyy += qs * a.y * a.y;
        jxy += qs * a.x * a.y;
        h += a.z > 0.25 ? amp * s : 0.0;
        float slope = a.z * a.w;
        lost += 0.5 * slope * slope * (1.0 - w * w);
    }
    return ocean_sample(math::normalize(float3(-sx, -sy, 1.0 - sz)), h, (1.0 - jxx) * (1.0 - jyy) - jxy * jxy, lost);
}

float noise(float2 p, float t, uniform float size, uniform int levels)
{
    return base::perlin_noise_texture(
        uvw: base::texture_coordinate_info(position: float3(p.x, p.y, t), tangent_u: float3(1, 0, 0), tangent_v: float3(0, 1, 0)),
        size: size,
        noise_levels: levels
    ).mono;
}

float fog_transmittance(float3 cam, float3 p, float sea_level, float density, float scale_height, float haze_distance)
{
    float zc = math::max(cam.z - sea_level, 0.0);
    float zp = math::max(p.z - sea_level, 0.0);
    float dz = zc - zp;
    float ep = math::exp(-zp / scale_height);
    float mean = math::abs(dz) > 1e-3 ? scale_height * (ep - math::exp(-zc / scale_height)) / dz : ep;
    float len = math::length(cam - p);
    float tau = density * len * mean + (haze_distance > 0.0 ? len / haze_distance : 0.0);
    return math::exp(-tau);
}

float leg_foam(float2 p, float t)
{
    float f = 0.0;
    for (int i = 0; i < LEG_COUNT; ++i) {
        float3 l = LEGS[i];
        float d = math::max(math::length(p - float2(l.x, l.y)) - l.z, 0.0);
        f = math::max(f, math::exp(-d / (0.8 + 0.6 * l.z)));
    }
    return f;
}

export material OceanWater(
    float time = 0.0,
    color deep_color = color(0.004, 0.016, 0.03),
    color crest_color = color(0.02, 0.09, 0.08),
    color foam_color = color(0.82),
    float roughness = 0.03,
    float crest_height = 0.6,
    float foam_threshold = 0.85,
    float foam_softness = 0.08,
    float foam_amount = 1.0,
    float leg_foam_amount = 1.0,
    float3 camera_position = float3(0.0, 0.0, 40.0),
    float pixel_angle = 0.001,
    float min_footprint = 0.02,
    float haze_distance = 0.0,
    color haze_radiance = color(0.0),
    float sea_level = 0.0,
    float fog_density = 0.0,
    float fog_scale_height = 5.0
)
= let {
    float2 p = float2(state::texture_coordinate(0).x, state::texture_coordinate(0).y);
    float3 pw = state::transform_point(state::coordinate_internal, state::coordinate_world, state::position());
    float3 to_cam = camera_position - pw;
    float dist = math::length(to_cam);
    float stretch = math::sqrt(1.0 / math::max(math::abs(to_cam.z) / math::max(dist, 1e-3), 0.05));
    ocean_sample o = ocean_eval(p, time, math::max(min_footprint, dist * pixel_angle * stretch));
    float alpha = math::min(math::sqrt(roughness * roughness + 2.0 * o.lost), 0.5);
    float n1 = noise(p, time * 0.15, 3.0, 4);
    float n2 = noise(p * 0.21 + float2(time * 0.4, 0.0), time * 0.05, 3.0, 3);
    float n3 = noise(p, time * 0.3, 0.5, 2);
    float crest_foam = math::saturate((foam_threshold - o.jac) / foam_softness + 0.5);
    float foam = math::saturate(foam_amount * crest_foam * math::smoothstep(0.3, 0.8, n1 + 0.35 * n2) * (0.6 + 0.4 * n3)
        + leg_foam_amount * leg_foam(p, time) * (0.25 + 0.75 * math::smoothstep(0.3, 0.7, n1)) * (0.5 + 0.5 * n3));
    float crest = math::pow(math::saturate(o.h / crest_height), 1.5);
    color body = math::lerp(deep_color, crest_color, crest);
    float3 n = state::transform_normal(state::coordinate_world, state::coordinate_internal, o.n);
    bsdf water = df::fresnel_layer(
        ior: 1.333,
        layer: df::microfacet_ggx_smith_bsdf(roughness_u: alpha, roughness_v: alpha, tint: color(1.0), mode: df::scatter_reflect),
        base: df::diffuse_reflection_bsdf(tint: body)
    );
    bsdf surf = df::weighted_layer(weight: foam, layer: df::diffuse_reflection_bsdf(tint: foam_color), base: water);
    float haze = 1.0 - fog_transmittance(camera_position, pw, sea_level, fog_density, fog_scale_height, haze_distance);
} in material(
    surface: material_surface(
        scattering: df::weighted_layer(weight: 1.0 - haze, layer: surf, base: bsdf()),
        emission: material_emission(emission: df::diffuse_edf(), intensity: haze_radiance * (haze * math::PI), mode: intensity_radiant_exitance)
    ),
    geometry: material_geometry(normal: n)
);

export material SplashZone(
    float sea_level = 0.0,
    float band = 2.5,
    color wet_color = color(0.035, 0.035, 0.03),
    color growth_color = color(0.025, 0.04, 0.02),
    float roughness = 0.12
)
= let {
    float3 pw = state::transform_point(state::coordinate_internal, state::coordinate_world, state::position());
    float h = pw.z - sea_level;
    float edge = band * (0.55 + 0.6 * noise(float2(pw.x * 0.7 + pw.y * 0.7, pw.z * 1.5), 0.0, 0.8, 3));
    float streak = noise(float2((pw.x - pw.y) * 3.0, pw.z * 0.15), 0.0, 0.4, 2);
    float top = edge + 1.2 * math::smoothstep(0.55, 0.8, streak);
    float growth = 1.0 - math::smoothstep(0.0, 0.6, h - 0.4 * band * noise(float2(pw.x + pw.y, pw.z), 0.0, 0.3, 3));
    color tint = math::lerp(wet_color, growth_color, growth);
    float alpha = math::lerp(roughness, 0.4, growth);
} in material(
    surface: material_surface(
        scattering: df::fresnel_layer(
            ior: 1.33,
            layer: df::microfacet_ggx_smith_bsdf(roughness_u: alpha, roughness_v: alpha, tint: color(1.0), mode: df::scatter_reflect),
            base: df::diffuse_reflection_bsdf(tint: tint)
        )
    ),
    geometry: material_geometry(cutout_opacity: h < top ? 1.0 : 0.0)
);

"""


def _f(v: float) -> str:
    return f"{float(v):.7g}"


def build_mdl(w: Waves, legs: Sequence[tuple[float, float, float]] = ()) -> str:
    a = ",\n    ".join(
        f"float4({_f(d[0])}, {_f(d[1])}, {_f(k)}, {_f(amp)})" for d, k, amp in zip(w.dir, w.k, w.amp)
    )
    b = ",\n    ".join(f"float4({_f(om)}, {_f(ph)}, {_f(q)}, 0)" for om, ph, q in zip(w.omega, w.phase, w.q))
    slots = list(legs) or [(0.0, 0.0, 0.0)]
    leg_src = ", ".join(f"float3({_f(x)}, {_f(y)}, {_f(r)})" for x, y, r in slots)
    return (
        TEMPLATE.replace("@WAVE_COUNT@", str(len(w.k)))
        .replace("@WAVE_A@", a)
        .replace("@WAVE_B@", b)
        .replace("@LEG_COUNT@", str(len(legs)))
        .replace("@LEG_SLOTS@", str(len(slots)))
        .replace("@LEGS@", leg_src)
    )
