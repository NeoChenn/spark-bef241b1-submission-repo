"""Model factory for the binary rust classifier: timm backbone + optional LoRA (peft).

Single-logit output (logit > 0 => rust before thresholding / calibration).
"""
from __future__ import annotations

import torch
import torch.nn as nn

PRIMARY = "mobilenetv4_conv_small"
MODELS = ["mobilenetv4_conv_small", "resnet18", "vit_small_patch14_dinov2"]
INPUT_SIZE = 224

LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.1


def _create(name: str, pretrained: bool) -> nn.Module:
    kw = {"img_size": INPUT_SIZE} if "vit" in name else {}
    return __import__("timm").create_model(name, pretrained=pretrained, num_classes=1, **kw)


def head_name(model: nn.Module) -> str:
    """Top-level name of the classifier module (e.g. 'classifier', 'fc', 'head')."""
    return model.default_cfg.get("classifier", "head")


def find_lora_targets(model: nn.Module, name: str) -> list[str]:
    """Module names to wrap with LoRA.

    ViT: attention qkv + proj Linear layers.
    CNN: 1x1 pointwise Conv2d (groups == 1). If a CNN has too few of those (resnet18 has only 3
    downsample 1x1s) we fall back to every non-stem, non-depthwise Conv2d so LoRA has capacity.
    """
    head = head_name(model)
    if "vit" in name:
        return [n for n, m in model.named_modules()
                if isinstance(m, nn.Linear) and (n.endswith("attn.qkv") or n.endswith("attn.proj"))]
    convs = [(n, m) for n, m in model.named_modules()
             if isinstance(m, nn.Conv2d) and m.groups == 1 and not n.startswith(head)]
    pw = [n for n, m in convs if m.kernel_size == (1, 1)]
    if len(pw) >= 10:
        return pw
    stem = convs[0][0] if convs else None
    return [n for n, _ in convs if n != stem]


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def build_model(name: str = PRIMARY, lora: bool = True, pretrained: bool = True, verbose: bool = True):
    """Return (model, meta). With lora=True the model is a peft PeftModel."""
    base = _create(name, pretrained)
    cfg = base.default_cfg
    meta = {
        "model_name": name,
        "input_size": [3, INPUT_SIZE, INPUT_SIZE],
        "mean": list(cfg.get("mean", (0.485, 0.456, 0.406))),
        "std": list(cfg.get("std", (0.229, 0.224, 0.225))),
        "head": head_name(base),
        "lora": lora,
    }
    if not lora:
        model = base
    else:
        from peft import LoraConfig, get_peft_model

        targets = find_lora_targets(base, name)
        conf = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                          target_modules=targets, modules_to_save=[meta["head"]], bias="none")
        model = get_peft_model(base, conf)
        # CNNs: let BatchNorm affine params adapt too (tiny param count, big help for domain shift).
        if "vit" not in name:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    for p in m.parameters(recurse=False):
                        p.requires_grad = True
        meta["lora_targets"] = len(targets)
        meta["lora_cfg"] = {"r": LORA_R, "alpha": LORA_ALPHA, "dropout": LORA_DROPOUT}
    tr, tot = count_params(model)
    meta["trainable_params"], meta["total_params"] = tr, tot
    if verbose:
        print(f"[models] {name} lora={lora} targets={meta.get('lora_targets', '-')} "
              f"trainable={tr:,} / total={tot:,} ({100 * tr / tot:.2f}%)")
    return model, meta


def to_plain(model: nn.Module) -> nn.Module:
    """Merge LoRA weights (if any) and return a plain timm module suitable for export."""
    if hasattr(model, "merge_and_unload"):
        return model.merge_and_unload()
    return model
