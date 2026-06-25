import os
import torch

try:
    from .architecture import GenericVanillaBaseline, GenericBioSiamese, GenericGenerativeLM
except ImportError as e:
    print(f"[Factory] architecture module import failed: {e}")


try:
    from .hyenadna_loader import load_hyenadna_lora
except ImportError as e:
    print(f"[Factory] hyenadna_loader import failed: {e}")

try:
    from .ntv3_loader import load_ntv3_lora
except ImportError as e:
    print(f"[Factory] ntv3_loader import failed: {e}")

try:
    from .caduceus_loader import load_caduceus
except ImportError as e:
    print(f"[Factory] caduceus_loader import failed: {e}")


def build_model(model_type, arch_type, model_dir, lora_r=16, lora_dropout=0.15, num_classes=2, device="cuda"):
    """
    Build a downstream GeoAlign model from a local backbone checkpoint.

    Args:
        model_type (str): "hyenadna", "ntv3", or "caduceus".
        arch_type (str): "baseline", "siamese"/"ours", or "tapt".
        model_dir (str): Local HuggingFace-style backbone directory.
        lora_r (int): LoRA rank for backbones that use PEFT.
        lora_dropout (float): LoRA dropout.
        num_classes (int): Number of output classes or regression targets.
        device (str): Target device.
    """
    model_type = model_type.lower()
    arch_type = arch_type.lower()

    print(f"[Factory] Loading backbone: {model_type.upper()} (LoRA rank={lora_r})")
    if model_type == "hyenadna":
        base_model = load_hyenadna_lora(model_dir=model_dir, r=lora_r, lora_dropout=lora_dropout, device=device)
    elif model_type == "ntv3":

        base_model = load_ntv3_lora(model_dir=model_dir, r=lora_r, lora_dropout=lora_dropout, device=device)
    elif model_type == "caduceus":

        base_model = load_caduceus(model_dir=model_dir, device=device)
    else:
        raise ValueError(f"Unsupported backbone type: {model_type}")

    print(f"[Factory] Attaching downstream architecture: {arch_type.upper()}")
    if arch_type == "baseline":
        model = GenericVanillaBaseline(base_model, model_type=model_type, num_classes=num_classes)
    elif arch_type in ["siamese", "ours"]:
        model = GenericBioSiamese(base_model, model_type=model_type, num_classes=num_classes)
    elif arch_type == "tapt":
        model = GenericGenerativeLM(base_model, model_type=model_type)
    else:
        raise ValueError(f"Unsupported downstream architecture: {arch_type}")

    model.to(device)
    print("[Factory] Model is ready.")
    return model
