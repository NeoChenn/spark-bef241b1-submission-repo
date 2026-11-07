# Rig assets

Everything here is listed in `manifest.json` (schema v2: `id`, `kind`, `source`, `url`, `licence`,
`resolution`, `path`, plus `category`, `note`, `remote`). Licences are in `LICENSES.md`, generated
from the manifest.

```
python sim/tools/fetch_assets.py                    # download / verify everything
python sim/tools/fetch_assets.py --kind hdri        # one kind
python sim/tools/fetch_assets.py --only kloofendal_48d_partly_cloudy_puresky_8k   # hero 8k sky (75 MB, gitignored)
python sim/tools/fetch_assets.py --licences         # rewrite LICENSES.md
```

Files live in `third_party/<path>`:

- `textures/<id>/` holds `diff.jpg` (sRGB), `nor_gl.jpg` (OpenGL normal), `rough.jpg`, plus `metal.jpg`,
  `ao.jpg`, `disp.jpg`, `opacity.jpg` when the set has them. All raw except `diff`. Same names for Poly
  Haven and ambientCG sets.
- `hdris/<id>/<id>_4k.hdr` (lat-long). The three legacy skies stay at `hdris/<id>.hdr`.
- `models/<id>/<id>_1k.usdc` with a `textures/` folder (Poly Haven, metres, Z up). Their USD files carry
  absolute texture paths; repoint them at the local folder as `build_rig_deck.SceneBuilder.asset` does.
- `decals/iso7010_<code>.png`, 1280 px RGBA safety signs.
- `remote: true` models are NVIDIA Omniverse USDs loaded straight from the public S3 asset root by URL
  (reachable from watson). They are not downloaded or committed.

`sim/tools/pbr_material.py` builds a UsdPreviewSurface from a texture folder (tiling, tint, opacity
cut-out). `sim/tools/render_asset_sheets.py {swatches,hdris,props}` renders the check sheets in
`sim/output/assets/`.

## Catalogue

Textures are 2k (legacy sets 1k), checked in `sim/output/assets/swatches_*.png`.

| Use | Texture ids |
| --- | --- |
| Open grating walkways (opacity cut-out) | MetalWalkway004 (galvanised bar grating), MetalWalkway007 (heavy), MetalWalkway010 (slotted plank), MetalWalkway014 (rusty perforated), metal_grate_rusty |
| Tread / checker plate | DiamondPlate001 (bare), DiamondPlate006A (yellow), DiamondPlate006C (worn yellow), DiamondPlate008C (dark worn), metal_plate |
| Painted steel, clean | Paint006 (safety yellow), Paint004 (white; tint for grey), Paint001 (orange-red), Paint003 (bright green), blue_metal_plate |
| Painted steel, weathered | PaintedMetal009 (yellow), PaintedMetal010 (white), PaintedMetal008 (orange), PaintedMetal006 (green), PaintedMetal014 (grey-blue), PaintedMetal016 (hazard stripes), container_side, green_metal_rust, rusty_painted_metal |
| Galvanised / cladding / pipe jacketing | corrugated_iron_02, Metal050C (rough aluminium), MetalPlates001 (riveted brushed plates) |
| Heavy rust | Rust007, Rust004, Metal053C, MetalPlates010, rusty_metal_04, metal_plate_02, rust_coarse_01, rusty_metal_02 |
| Salt / water staining | Metal021 (streaked steel), SurfaceImperfections011 (stain overlay with opacity) |
| Concrete | Concrete042C (damaged, rusty rebar), Concrete034 (smooth), concrete_floor_03 |
| Rubber matting | Rubber004 (black), rubberized_track (red) |
| Anti-slip deck / helideck | Asphalt033 (dark grit), Asphalt031 (light grit; tint green with `tint=` for the helideck) |
| Mesh / nets | Fence008A (welded mesh), Fence003 (chain-link; wires are sub-pixel beyond a few metres), Net004A (rope net, helideck net) |
| Signs / tape | Sign002, Sign008, Sign009, Sign012, Sign018, Sign020 (yellow triangle warnings on metal), Sticker001 (red/white barrier tape) |

Safety-sign decals (`decals/`, public domain ISO 7010): E007 muster point, P002 no smoking, M014 hard hat,
M009 gloves, M004 eye protection, M008 safety boots, M010 ear protection, E003 first aid, E011 eyewash,
E012 safety shower, F001 extinguisher, F002 fire hose reel, E001 exit, W001 general, W012 electricity,
W021 flammable.

Skies (`hdris/`, 4k, checked in `sim/output/assets/hdris.png`). The `_puresky` versions have no ground,
so the ocean surface supplies the lower half:

| Mood | HDRI ids |
| --- | --- |
| Clear day | kloofendal_43d_clear_puresky, qwantani_noon_puresky (sun in map) |
| Partly cloudy | kloofendal_48d_partly_cloudy_puresky (hero; 8k entry `_8k`), aristea_wreck_puresky |
| Golden hour / sunset | qwantani_sunset_puresky, industrial_sunset_02_puresky |
| Overcast North Sea | kloofendal_overcast_puresky, overcast_soil_puresky, mud_road_puresky |
| Dusk / night | qwantani_dusk_2_puresky, qwantani_night_puresky |

Poly Haven skies are exposure-normalised, so night and dusk come out as bright as day at the same dome
intensity (measured mean sky radiance is within 2x across all of them). Turn the dome intensity down by
eye when using them. Clear, noon and sunset maps contain the sun disc and cast hard shadows on their
own; the overcast maps have no sun peak.

Props (checked in `sim/output/assets/props_*.png`, sizes in metres in the labels):

| Use | Poly Haven (local USD) | NVIDIA (remote USD) |
| --- | --- | --- |
| Life-saving | lifebuoy, life_jacket, ocean_buoy | nv_eyewash_shower |
| Fire | fire_alarm, fire_hydrant, korean_fire_extinguisher_01 | nv_fire_cabinet |
| Lifting | overhead_crane | nv_jib_crane |
| Lighting | industrial_caged_sconce, mounted_fluorescent_lights, security_light, caged_hanging_light | nv_worklight (tripod flood) |
| Access / railing | ladder_sectioned_01, modular_chainlink_fence, modular_fire_escape | nv_guard_rail, nv_rolling_scaffold, nv_safety_bollard |
| Cable trays / electrical | security_camera_01, utility_box_01/02, power_box_01, modular_electric_cables | nv_cable_tray, nv_cable_tray_corner |
| Pipes / HVAC | modular_industrial_pipes_01, modular_pipes, modular_airduct_circular_01, modular_airduct_rectangular_01, exterior_aircon_unit | |
| Gas / storage | small_lpg_tank, propane_tank, Barrel_01, barrel_03, industrial_storage_cart | nv_gas_cylinder_truck, nv_ibc_tank, nv_steel_drum |
| Tools / misc | tool_cart, metal_tool_chest, hand_truck, pipe_wrench, garden_hose_wall_mounted_01, WetFloorSign_01 | nv_hse_board, nv_safety_tape |

Not found as free downloads: liferaft canisters / lifeboat capsules, wheel valves, pressure gauges,
winches, helideck lights, shipping containers as models (use `container_side` on a box). NVIDIA SimReady
extinguisher, suppression cylinders, first-aid cabinet, tripod flood light, hose reel, siren and junction
boxes loaded without visible geometry or at the wrong scale and were dropped.
