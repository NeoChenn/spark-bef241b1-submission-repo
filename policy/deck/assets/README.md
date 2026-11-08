# Mock sprites for the deck sweep

Held-out **test-split** crops only (the detectors never trained on them).

| File | Source | Licence |
|---|---|---|
| `debris_*.png` | Hugging Face `alexNova/MRS-Trash-Detection` (from ModelScope `YizheDev/trash_detection`), test split, cut out with `detection.prepare_rig_data.object_alpha` | Apache-2.0 |
| `person_*.jpg` | Hugging Face `VincentGOURBIN/ppe-detection` (Roboflow `vincentspace/ppe-detection-1-cniwr` v2), test split by source photo | CC BY 4.0 |

Drawn drink cans come from `detection.prepare_rig_data.draw_can` (no file).
