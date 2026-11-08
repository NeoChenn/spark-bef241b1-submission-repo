# Deck sweep: picking up debris and seeing people

Two more camera skills on the same robot and the same plan engine as rust repair and the gas-leak job. Both detectors
are trained the same way as the rust and crack models. Numbers and limits:
[`detection/RIG_VISION_RESULTS.md`](../detection/RIG_VISION_RESULTS.md). Bridge methods:
[`integration/INTERFACE.md`](../integration/INTERFACE.md) section 7.

## Where it sits in a patrol stop

```mermaid
flowchart TD
    P["Arrive at patrol stop"] --> G{"Gas"}
    G -- "above 0.05 LEL" --> L["Leak job"]
    G -- "normal" --> C["DEBRIS_CHECK<br/>look down the deck"]
    L -- "fixed" --> C
    C -- "deck clear" --> R["DETECT<br/>rust job (unchanged)"]
    C -- "unsure" --> V["DEBRIS_SECOND_VIEW<br/>(up to 2)"] --> C
    C -- "debris" --> K["DEBRIS_PICKUP<br/>arm picks at the object"]
    K --> F["DEBRIS_VERIFY<br/>look again"]
    F -- "more" --> K
    F -- "clear" --> R
    K -- "cannot lift it" --> W["report (warning)<br/>and carry on"] --> R
```

Debris is not an emergency: if the robot cannot lift something or cannot tell, it reports it and keeps patrolling.

## How it sees debris

```mermaid
flowchart LR
    I["Camera frame<br/>at this stop"] --> CH["Change check<br/>against the clean reference<br/>(ignores shadows and light)"]
    REF["Clean reference<br/>of this stop"] --> CH
    CH --> WIN["Only windows that changed"]
    WIN --> M["Debris model<br/>MobileNetV4 + LoRA<br/>(trained like rust)"]
    M -- "litter" --> AIM["Aim at the middle<br/>of the changed pixels"] --> ARM["Arm picks it up"]
    M -- "not litter<br/>(boots, a hose, a puddle)" --> X["ignore"]
```

Why the reference: the model alone fired on fire extinguishers and gas bottles it had never seen from that angle
(every empty deck flagged). With the clean reference of each stop, fixed equipment is never a candidate: 73 of 75
objects found and 0 false alarms, under random lighting and shadows.

## How it sees people

```mermaid
flowchart LR
    B["Body camera frame"] --> PM["Person model<br/>(trained like rust)"]
    D["Depth camera"] --> DIST["Distance of what stands up<br/>off the deck inside the detection"]
    PM -- "person" --> DIST
    DIST -- "under 3 m" --> PAUSE["PAUSE_PERSON"]
    GT["bridge.person_distance()<br/>(team's rule, unchanged)"] -- "under 3 m" --> PAUSE
    PAUSE -- "both clear" --> RES["resume"]
```

The model says *who*, depth says *how far*: 40 of 40 pause decisions right on the held-out deck. The camera can add a
pause; it never removes one.

## Try it

```
python -m policy.leak.mock --leak none --debris litter --person-at 2
python -m detection.eval_rig_vision
```
