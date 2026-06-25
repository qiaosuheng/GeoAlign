import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM
from peft import get_peft_model, LoraConfig
import types


def load_ntv3_lora(model_dir, r=16, lora_dropout=0.15, device="cuda"):
    print(f" [Loader] Loading NTv3_8M Foundation Model from {model_dir}...")







    model = AutoModelForMaskedLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        local_files_only=True
    )


    if not hasattr(model, "get_input_embeddings"):
        def custom_get_input_embeddings(self):
            for module in self.modules():
                if isinstance(module, nn.Embedding):
                    return module
            raise AttributeError("  Embedding ")

        model.get_input_embeddings = types.MethodType(custom_get_input_embeddings, model)



    try:
        model.gradient_checkpointing_enable()
        print(" [Loader] Gradient Checkpointing ")
    except Exception as e:
        print(f" [Loader]  Gradient Checkpointing: {e}")

    # ==========================================================

    # ==========================================================
    ntv3_targets = ["linear", "mha_output", "fc1", "fc2"]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=ntv3_targets,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None
    )


    model = get_peft_model(model, peft_config)


    model.print_trainable_parameters()
    model.to(device)

    return model