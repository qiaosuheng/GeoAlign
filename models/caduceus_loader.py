import torch
import torch.nn as nn
from transformers import AutoModel
import types


def load_caduceus(model_dir, device="cuda"):
    """
     Caduceus ( Mamba + RCPS) 
    Caduceus  Mamba SSM  DNA  RC  (RCPS)
    d_model=256, n_layer=16, vocab_size=16, float32

    
    -  Mamba (BiMamba) + RC  (RCPS)
    -  Tokenizer (A=7, C=8, G=9, T=10, N=11)
    -  gradient checkpointing
    - 
    """
    print(f" [Loader] Loading Caduceus from {model_dir}...")


    model = AutoModel.from_pretrained(
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


    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" [Loader] : {total_params}, : {trainable_params}")
    model.to(device)

    return model
