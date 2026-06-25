import torch
import torch.nn as nn
from transformers import AutoModel
from peft import get_peft_model, LoraConfig
import types


def load_hyenadna_lora(model_dir, r=16, lora_dropout=0.15, device="cuda"):
    print(f" [Loader] Loading HyenaDNA from {model_dir}...")


    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # ==========================================================


    # ==========================================================
    def custom_get_input_embeddings(self):

        for module in self.modules():
            if isinstance(module, nn.Embedding):
                return module
        raise AttributeError("  Embedding ")


    model.get_input_embeddings = types.MethodType(custom_get_input_embeddings, model)


    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


    peft_config = LoraConfig(
        r=r,
        lora_alpha=r * 2,
        target_modules=["in_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type=None
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    return model