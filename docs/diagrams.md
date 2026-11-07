# Rust repair robot: diagrams

GitHub renders these Mermaid diagrams. Every number comes from the code on branch `rust-detection`
(`policy/envs/scrub_env_cfg.py`, `detection/models.py`, `detection/RESULTS.md`).

1. [System overview](#1-system-overview)
2. [Repair loop (state machine)](#2-repair-loop-state-machine)
3. [Adaptation: re-planning on an unscripted change](#3-adaptation-re-planning-on-an-unscripted-change)
4. [RL training loop (scrub policy)](#4-rl-training-loop-scrub-policy)
5. [Observation and reward](#5-observation-and-reward)
6. [Rust physics model](#6-rust-physics-model)
7. [CV model architecture](#7-cv-model-architecture)
8. [From pixels to a scrub target](#8-from-pixels-to-a-scrub-target)
9. [Data pipeline](#9-data-pipeline)
10. [Where everything runs](#10-where-everything-runs)

---

## 1. System overview

Perception decides everything the robot does. The detector sees rust, the orchestrator decides what to do, and the
skills act. Each action changes the world, and the camera checks the result.

```mermaid
flowchart LR
    subgraph World["Rig scene (Isaac Sim)"]
        PIPE["Pipe + rust patch"]
        CAM["Wrist / repair_closeup camera"]
        FT["Brush contact force"]
    end

    subgraph Perception
        CLS["Rust classifier<br/>MobileNetV4-S + LoRA<br/>1.1 ms CPU"]
        LOC["Localizer<br/>patch heatmap → mask → pipe grid"]
    end

    subgraph Brain["Orchestrator (state machine)"]
        SM["Decide next state<br/>+ log the evidence"]
    end

    subgraph Skills
        SPRAY["Spray<br/>coverage path"]
        SCRUB["Scrub<br/>PPO policy"]
        PAINT["Paint<br/>coverage path + check"]
    end

    subgraph Body["Go2 + lift mast + UR3"]
        LIFT["Lift 0–0.45 m"]
        ARM["UR3 diff-IK on brush_tip"]
    end

    CAM --> CLS --> LOC --> SM
    FT --> SM
    SM --> SPRAY & SCRUB & PAINT
    SM -- "set height from rust z" --> LIFT
    SPRAY & SCRUB & PAINT --> ARM
    LIFT --> ARM --> PIPE
    PIPE -. "re-detect after every step" .-> CAM
    SM --> LOG[("decisions.jsonl<br/>+ HUD overlay")]
```

## 2. Repair loop (state machine)

`policy/orchestrator.py`. Every transition writes a `DecisionRecord` with the detector confidence, rust fraction,
patch ids and force behind it.

```mermaid
stateDiagram-v2
    [*] --> PATROL
    PATROL --> DETECT
    DETECT --> SECOND_VIEW: unsure (p in 0.35–0.65)
    SECOND_VIEW --> DETECT: max 2 extra views
    SECOND_VIEW --> ESCALATE: still unsure
    DETECT --> APPROACH: rust found
    DETECT --> PATROL: clean
    APPROACH --> SPRAY: lift + arm in reach
    SPRAY --> SOAK
    SOAK --> SPRAY: NEW patch seen (re-plan)
    SOAK --> SCRUB: 30 s since last spray
    SCRUB --> BACK_OFF: force > 40 N
    BACK_OFF --> SCRUB: retract 5 cm, press × 0.7
    SCRUB --> VERIFY
    VERIFY --> SCRUB: rust > 5% (max 3 cycles)
    VERIFY --> ESCALATE: still rusty after 3 cycles
    VERIFY --> PAINT: clean
    PAINT --> VERIFY_PAINT
    VERIFY_PAINT --> PAINT: coverage < 98%
    VERIFY_PAINT --> DONE
    ESCALATE --> [*]: work order + evidence to a human
    DONE --> [*]

    state "Safety (any state)" as SAFE {
        HOLD: camera lost → hold position
        PAUSE_PERSON: person < 3 m → pause
    }
```

## 3. Adaptation: re-planning on an unscripted change

What the judges see when a new patch appears mid-run. This trace is real output from `python -m policy.demo_mock`.

```mermaid
sequenceDiagram
    autonumber
    participant Cam as Camera
    participant Det as Detector + localizer
    participant Orc as Orchestrator
    participant Arm as Arm skills
    participant W as World

    Cam->>Det: frame
    Det->>Orc: patch #0, conf 0.98, 58 cells
    Orc->>Arm: SPRAY patch #0 (40 waypoints, 0.15 m standoff)
    Note over Orc: SOAK, re-detect every 5 s
    W-->>W: unscripted: new rust patch #1 appears
    Cam->>Det: frame (t = 15 s)
    Det->>Orc: NEW patch #1, conf 0.90
    Orc->>Orc: RE-PLAN: spray #1, extend soak
    Orc->>Arm: SPRAY patch #1
    Orc->>Arm: SCRUB #0 + #1 (PPO policy)
    Arm->>Orc: force 65 N > 40 N
    Orc->>Arm: BACK_OFF, retry at 0.7 × press
    Cam->>Det: VERIFY frame
    Det->>Orc: rust 0%, conf 0.01
    Orc->>Arm: PAINT, coverage 100%, DONE
```

## 4. RL training loop (scrub policy)

`policy/envs/scrub_env.py` and `policy/train_scrub.py`. rsl_rl PPO in Isaac Lab 2.3 on a Modal L40S: 2,048
parallel envs at about 45k env-steps/s. The task is solved by iteration ~20.

```mermaid
flowchart TB
    subgraph Envs["Isaac Lab DirectRLEnv × 2048 (GPU)"]
        direction TB
        RESET["Reset + domain randomisation<br/>pipe z 1.1 ± 0.05 m, r 0.15 ± 0.03 m, yaw ± 0.1 rad<br/>brush stiffness 1.5–4 kN/m, abrasion ± 30%<br/>1–3 rust blobs, lift 0.38–0.45 m"]
        OBS["Observation, 327-D"]
        IK["Differential IK on brush_tip<br/>→ 6 UR3 joint targets"]
        PHYS["PhysX 120 Hz × 4 decimation<br/>= 30 Hz control, 15 s episodes"]
        CONTACT["Analytic brush–pipe contact<br/>penetration × k → normal force"]
        RUST["RustState.abrade()<br/>removes rust on the grid"]
        EVENT["Mid-episode event (p = 0.3)<br/>new rust patch at 3–10 s"]
        REW["Reward"]
    end

    subgraph Agent["PPO (rsl_rl)"]
        ACTOR["Actor MLP 512-256-128<br/>Gaussian policy"]
        CRITIC["Critic MLP 512-256-128"]
        BUF["Rollout buffer<br/>24 steps × 2048 envs"]
        UPD["PPO update<br/>clip 0.2, γ 0.99, λ 0.95<br/>lr 3e-4, entropy 0.005"]
    end

    RESET --> OBS
    OBS --> ACTOR
    ACTOR -- "action 4-D:<br/>Δθ, Δz, Δnormal, force setpoint 5–45 N" --> IK
    IK --> PHYS --> CONTACT --> RUST --> REW
    EVENT -.-> RUST
    RUST --> OBS
    OBS --> CRITIC
    REW --> BUF
    OBS --> BUF
    BUF --> UPD
    UPD -- "new weights" --> ACTOR & CRITIC
    REW -- "done: rust < 5% or timeout" --> RESET

    UPD -. "every 50 iters" .-> CKPT[("model_N.pt → policy.onnx<br/>Modal volume rust-ckpt")]
```

**Learning curve** (`scrub_lift_v1`, `progress.jsonl`)

| Iteration | Mean reward | Rust removed | Success | Episode max force |
|---|---|---|---|---|
| 0 | −0.8 | 0% | 0% | – |
| 10 | 19.5 | 40% | 0% | – |
| 20 | 53 | 96% | 100% | – |
| 100 | 70.5 | 95% | 100% | – |
| 735 | 74.4 | 95% | 100% | 46 N |

Eval on the exact rig geometry, with a new patch injected every episode: **99.2% success, 5.9 s mean time to clean.**

## 5. Observation and reward

```mermaid
pie showData
    title Observation, 327 dims
    "Local rust patch 9×9" : 81
    "Local sprayed mask 9×9" : 81
    "Coarse rust map 8×16" : 128
    "Arm joints pos + vel" : 12
    "Brush pose, vel, gap, surface coords" : 10
    "Target offset, force, setpoint, pipe r" : 6
    "Nearest rust, fraction left, flags" : 5
    "Previous action" : 4
```

The rust maps are grids on the pipe surface (rows = axial z, columns = angle θ). In deployment the same grid is
filled from the camera by the localizer (section 8), so the policy's input doesn't change.

```mermaid
flowchart LR
    subgraph Plus["Rewards"]
        R1["+50 × rust removed this step"]
        R2["+20 completion bonus<br/>(rust left < 5%)"]
        R3["+0.02 contact in 10–40 N band"]
    end
    subgraph Minus["Penalties"]
        P1["−0.1 per 10 N above 40 N"]
        P2["−0.05 scrubbing unsprayed area"]
        P3["−0.01 action rate"]
        P4["−0.2 per metre to nearest rust"]
    end
    Plus --> SUM(("Σ reward"))
    Minus --> SUM
```

**Known issue:** the over-force penalty is too weak, so mean episode-max force is about 46–52 N. Scrub v2 raises it
5× and clamps the setpoint below 38 N.

## 6. Rust physics model

`policy/envs/rust_state.py`, pure torch, batched over envs. This is why the spray and soak steps matter to the
scrub policy.

```mermaid
flowchart LR
    F["Normal force F"] --> GATE{"F ≥ 5 N?"}
    V["Slide speed v"] --> RATE
    GATE -- no --> ZERO["no removal"]
    GATE -- yes --> RATE["removal ∝ F × v × dt<br/>(Archard-like wear)"]
    SPR{"Cell sprayed?"} --> MULT
    SOAKQ{"Soaked ≥ 30 s?"} --> MULT["× 4 sprayed + soaked<br/>× 1.5 sprayed only<br/>× 1 dry"]
    MULT --> RATE
    RATE --> GRID[("Rust grid<br/>level 0–1 per cell")]
    GRID --> TEX["to_texture() → rust overlay<br/>visibly disappears in the sim"]
    F -- "> 40 N" --> FLAG["excess-force flag<br/>→ penalty / BACK_OFF"]
```

## 7. CV model architecture

`detection/models.py`. MobileNetV4-Conv-Small, ImageNet-pretrained (timm), with **LoRA adapters on its 28 pointwise
(1×1) convolutions**. Only 0.26M of 2.49M parameters train. Shown with real channel counts and strides.

```mermaid
flowchart TB
    IN["RGB crop 224 × 224 × 3<br/>ImageNet mean/std"] --> STEM

    subgraph Backbone["MobileNetV4-Conv-Small backbone (frozen + LoRA)"]
        STEM["Conv 3×3 s2 → 32 ch · /2"]
        S0["Stage 0: 2 × Conv-BN-Act → 32 ch · /4"]
        S1["Stage 1: 2 × Conv-BN-Act → 64 ch · /8"]
        S2["Stage 2: 6 × Universal Inverted Bottleneck → 96 ch · /16"]
        S3["Stage 3: 6 × Universal Inverted Bottleneck → 128 ch · /32"]
        S4["Conv 1×1 → 960 ch"]
        STEM --> S0 --> S1 --> S2 --> S3 --> S4
    end

    S4 --> GAP["Global average pool"]
    GAP --> HEAD["Conv 1×1 960 → 1280 + BN-Act"]
    HEAD --> FC["Linear 1280 → 1<br/>(fully trainable)"]
    FC --> LOGIT["logit"]
    LOGIT --> TEMP["÷ T = 2.607<br/>(temperature, fitted on val)"]
    TEMP --> SIG["sigmoid → p_rust"]
    SIG --> DEC{"p_rust"}
    DEC -- "≥ 0.476" --> RUSTY["RUST"]
    DEC -- "< 0.476" --> CLEAN["CLEAN"]
    DEC -- "0.35–0.65" --> UNSURE["UNSURE → take a second view"]

    classDef frozen fill:#e8eef5,stroke:#5a7896
    classDef train fill:#fde9dc,stroke:#cf4b17
    class STEM,S0,S1,S2,S3,S4,GAP,HEAD frozen
    class FC train
```

**LoRA inside each 1×1 conv.** The pretrained weight W₀ stays frozen. A low-rank update is learned beside it:

```mermaid
flowchart LR
    X["x<br/>(C_in channels)"] --> W0["W₀ 1×1 conv<br/>frozen"]
    X --> A["A: C_in → 16<br/>(rank r = 16)"]
    A --> DROP["dropout 0.1"] --> B["B: 16 → C_out<br/>init 0"]
    B --> S["× α/r = 32/16"]
    W0 --> ADD(("+"))
    S --> ADD
    ADD --> Y["y = W₀x + (α/r)·BAx"]
    classDef frozen fill:#e8eef5,stroke:#5a7896
    classDef train fill:#fde9dc,stroke:#cf4b17
    class W0 frozen
    class A,B train
```

**Training:** BCEWithLogits with `pos_weight = n_clean / n_rust`, AdamW + cosine LR, AMP, RandomResizedCrop + flips
+ mild colour jitter (small hue shift, because rust is defined by colour), early stopping on val PR-AUC. The threshold
is the highest that keeps val recall ≥ 0.95 on both the real and the synthetic subsets, because missing rust is the
costly error.

**Model comparison** (same data, real-photo test set)

| Model | Trainable / total params | Test F1 | AUROC | CPU latency (1 thread) |
|---|---|---|---|---|
| **MobileNetV4-S + LoRA (deployed)** | 0.26M / 2.5M | **0.977** (rust_v2) | 0.998 | **2.6 ms** |
| MobileNetV4-S full fine-tune | 2.5M / 2.5M | 0.972 | 0.995 | 4.3 ms |
| ResNet-18 + LoRA | – / 11.7M | 0.968 | 0.995 | 40.7 ms |
| DINOv2 ViT-S/14 + LoRA | – / 22M | 0.976 | 0.999 | 145 ms |

Sim-domain check (held-out renders of the rig scene): rust_v1 model 10/13 → **rust_v2 model 13/13**.

## 8. From pixels to a scrub target

`detection/localize.py`. The classifier gives a yes/no per crop, so sliding it over the frame produces a map of
where the rust is.

```mermaid
flowchart LR
    FRAME["Camera frame<br/>e.g. 640 × 360"] --> TILE["Sliding patches<br/>96 px, stride 48"]
    TILE --> BATCH["Batched ONNX classifier<br/>calibrated p per patch"]
    BATCH --> HEAT["Heatmap"]
    HEAT --> MASK["Threshold → binary mask"]
    MASK --> CC["Connected components<br/>bbox, centroid, area, mean conf"]
    HEAT --> UNS{"Any patch in 0.35–0.65<br/>not at a blob edge?"}
    UNS -- yes --> SV["unsure → SECOND_VIEW"]
    CC --> RAY["Ray–cylinder intersection<br/>camera intrinsics + pipe axis/radius"]
    RAY --> GRID[("Pipe-surface grid (θ, z)<br/>same layout as RustState")]
    GRID --> ORC["Orchestrator: spray / scrub / paint targets"]
    GRID --> POL["Scrub policy observation"]
```

## 9. Data pipeline

`detection/prepare_data.py` and `detection/synth_sim.py`.

```mermaid
flowchart LR
    subgraph Real["Real photos (Hugging Face)"]
        H1["BinKhoaLe1812/Corrosion_Rust<br/>rust / no rust · MIT"]
        H2["Francesco/corrosion-bi3q3 (RF100)<br/>box crops + background crops · CC BY 4.0"]
        H3["iamcloud/Corrosion_Dataset<br/>rust only · Apache-2.0"]
    end
    H1 & H2 & H3 --> DEDUP["Perceptual-hash dedupe<br/>Hamming ≤ 4 → same group"]
    DEDUP --> SPLIT["Split 70/15/15 by group_id<br/>(no leakage)"]
    SPLIT --> V1[("rust_v1<br/>5,894 imgs · 37% rust")]

    subgraph Synth["Sim-style synthetic (scene's own Poly Haven textures)"]
        T1["rust_coarse_01, rusty_metal_02,<br/>metal_grate_rusty → rust"]
        T2["metal_plate, blue_metal_plate,<br/>paints, full_rig renders → clean"]
        AUG["Pipe shading + highlight, exposure,<br/>white balance, blur, noise, JPEG;<br/>rust blobs pasted on painted pipe"]
        T1 & T2 --> AUG
    end
    AUG --> V2[("rust_v2<br/>v1 + ~3,400 crops<br/>4×4 texture-region split")]
    V1 --> V2
    V2 --> TRAIN["Train MobileNetV4-S + LoRA"]
    HELD["preview_* renders<br/>(never trained on)"] --> SIMTEST["Held-out sim test: 13/13"]
    TRAIN --> SIMTEST
```

## 10. Where everything runs

```mermaid
flowchart TB
    subgraph Modal["Modal cloud (training only)"]
        A10["A10G: detector training<br/>~minutes per run"]
        L40["L40S: Isaac Sim 5.1 + Isaac Lab PPO<br/>2048 envs · ~45k steps/s<br/>(no rendering: gVisor has no Vulkan)"]
        VOL[("Volumes: rust-data, rust-ckpt, rig-assets")]
        A10 <--> VOL
        L40 <--> VOL
    end

    subgraph GB10["Sim team's GB10 (aarch64)"]
        ISAAC["Isaac Sim 6.2 rig scene<br/>RTX render"]
        STREAM["Camera → JPEG → FastAPI → browser"]
        ISAAC --> STREAM
    end

    subgraph Edge["On the robot (edge budget)"]
        DET["Rust classifier ONNX<br/>1.1 ms · 4 CPU threads"]
        POLICY["Scrub policy ONNX<br/>MLP, under 1 ms"]
        ORCH["Orchestrator<br/>pure Python"]
    end

    VOL -- "model.onnx + meta.json<br/>policy.onnx" --> Edge
    VOL -- "URDF, policy, logged rollouts" --> GB10
    ISAAC -- "camera frames" --> DET
    ORCH -- "arm + lift commands" --> ISAAC
```

No remote calls are needed at run time. Every decision runs on the device, which is what the autonomy criterion asks for.
