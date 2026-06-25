import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.model_select import build_model
    from data_loader.dataset_select import build_dataset
except ImportError as e:
    print("[ImportError] Please check project paths:", e)
    raise


MODEL_PATHS = {
    "hyenadna": os.path.join(PROJECT_ROOT, "models", "hyenadna-small-32k-seqlen-hf"),
    "ntv3": os.path.join(PROJECT_ROOT, "models", "NTv3_8M_pre"),
    "caduceus": os.path.join(PROJECT_ROOT, "models", "caduceus-ps_seqlen-131k_d_model-256_n_layer-16"),
}


# =============================================================================
# Configuration
# =============================================================================
# Server usage:
#   1. Edit this CONFIG block.
#   2. Run: python train_classification/train_tapt.py
CONFIG = {
    # Corpus built by code/data/build_splice_tapt_corpus.py.
    "corpus_path": os.path.join(PROJECT_ROOT, "data", "tapt_splice_family", "train.parquet"),

    # Backbone.
    "arch_type": "tapt",
    "model_type": "hyenadna",  # "hyenadna" or "ntv3"
    "lora_r": 16,
    "lora_dropout": 0.15,

    # Optimization.
    "seed": 42,
    "lr": 1e-4,
    "batch_size": 4,
    "grad_accum": 32,
    "epochs": 2,
    "seq_len": 1024,
    "weight_decay": 0.01,
    "warmup_ratio": 0.10,
    "max_grad_norm": 1.0,

    # Runtime.
    "num_workers": 4,
    "pin_memory": True,
    "use_amp": True,
    "amp_dtype": "bf16",  # "bf16" or "fp16"
    "deterministic": False,
    "save_every_epoch": True,
    "dense_log_until_update": 100,
    "dense_log_every_update": 5,
    "log_every_update": 100,

    # Output.
    "experiment_tag": "histone_family",
}


def build_experiment_name():
    return "tapt_{model}_r={rank}_{tag}_seed={seed}".format(
        model=CONFIG["model_type"],
        rank=CONFIG["lora_r"],
        tag=CONFIG["experiment_tag"],
        seed=CONFIG["seed"],
    )


CONFIG["save_dir"] = os.path.join(PROJECT_ROOT, "checkpoints", build_experiment_name())


def seed_everything(seed, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as exc:
            print("[WARN] deterministic algorithms are not fully available:", exc)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id):
    worker_seed = (CONFIG["seed"] + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def save_config(save_dir):
    path = os.path.join(save_dir, "config.json")
    serializable = {}
    for key, value in CONFIG.items():
        serializable[key] = str(value) if not isinstance(value, (int, float, bool, str, list, dict, type(None))) else value
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def get_amp_context(device):
    if (not CONFIG["use_amp"]) or device.type != "cuda":
        return nullcontext()
    dtype = torch.bfloat16 if CONFIG["amp_dtype"].lower() == "bf16" else torch.float16
    return torch.amp.autocast("cuda", dtype=dtype)


def compute_lm_loss(logits, labels, model_type, criterion):
    model_type = model_type.lower()

    if model_type in ["hyenadna"]:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    if model_type in ["ntv3", "caduceus"]:
        return criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    raise ValueError("Unsupported TAPT model_type: {}".format(model_type))


def checkpoint_state_dict(model):
    # Downstream fine-tuning scripts load this into model.base_model.
    if hasattr(model, "base_model"):
        return model.base_model.state_dict()
    return model.state_dict()


def append_csv_row(path, header, row):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def should_log_update(update_step):
    if update_step <= CONFIG["dense_log_until_update"]:
        return update_step % CONFIG["dense_log_every_update"] == 0
    return update_step % CONFIG["log_every_update"] == 0


def train_foundation():
    seed_everything(CONFIG["seed"], CONFIG["deterministic"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    save_config(CONFIG["save_dir"])

    print("[TAPT] model_type:", CONFIG["model_type"])
    print("[TAPT] corpus:", CONFIG["corpus_path"])
    print("[TAPT] save_dir:", CONFIG["save_dir"])
    print("[TAPT] seed:", CONFIG["seed"])

    dataset = build_dataset(
        data_path=CONFIG["corpus_path"],
        arch_type=CONFIG["arch_type"],
        model_type=CONFIG["model_type"],
        max_length=CONFIG["seq_len"],
    )

    generator = torch.Generator()
    generator.manual_seed(CONFIG["seed"])
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=CONFIG["pin_memory"] and device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    print("[TAPT] samples:", len(dataset))
    print("[TAPT] batches_per_epoch:", len(loader))

    model_type = CONFIG["model_type"].lower()
    model_dir = MODEL_PATHS.get(model_type)
    if model_dir is None:
        raise ValueError("Missing model path for model_type={}".format(model_type))

    model = build_model(
        model_type=model_type,
        arch_type=CONFIG["arch_type"],
        model_dir=model_dir,
        lora_r=CONFIG["lora_r"],
        lora_dropout=CONFIG["lora_dropout"],
        device=device,
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print("[TAPT] trainable parameters: {:,} / {:,}".format(trainable_params, all_params))

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    updates_per_epoch = int(math.ceil(float(len(loader)) / float(CONFIG["grad_accum"])))
    total_updates = max(1, updates_per_epoch * CONFIG["epochs"])
    warmup_updates = int(CONFIG["warmup_ratio"] * total_updates)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
    )

    detailed_log = os.path.join(CONFIG["save_dir"], "tapt_loss_detailed.csv")
    epoch_log = os.path.join(CONFIG["save_dir"], "tapt_loss_epoch.csv")
    detailed_header = ["epoch", "batch_step", "update_step", "loss", "lr"]
    epoch_header = ["epoch", "mean_loss", "updates", "lr"]

    print("[TAPT] start training")
    global_update = 0

    for epoch in range(CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad()

        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        accum_loss_sum = 0.0
        accum_count = 0

        pbar = tqdm(loader, desc="Epoch {}/{}".format(epoch + 1, CONFIG["epochs"]))
        for batch_step, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            with get_amp_context(device):
                logits = model(input_ids, attention_mask=attention_mask)
                loss = compute_lm_loss(logits, labels, model_type, criterion)

            raw_loss = float(loss.detach().cpu().item())
            (loss / CONFIG["grad_accum"]).backward()

            epoch_loss_sum += raw_loss
            epoch_loss_count += 1
            accum_loss_sum += raw_loss
            accum_count += 1

            should_update = accum_count == CONFIG["grad_accum"] or batch_step == len(loader)
            if should_update:
                if CONFIG["max_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                global_update += 1
                update_loss = accum_loss_sum / float(accum_count)
                lr = scheduler.get_last_lr()[0]

                if should_log_update(global_update):
                    append_csv_row(
                        detailed_log,
                        detailed_header,
                        {
                            "epoch": epoch + 1,
                            "batch_step": batch_step,
                            "update_step": global_update,
                            "loss": update_loss,
                            "lr": lr,
                        },
                    )

                pbar.set_postfix({"loss": "{:.4f}".format(update_loss), "lr": "{:.2e}".format(lr)})
                accum_loss_sum = 0.0
                accum_count = 0

        mean_epoch_loss = epoch_loss_sum / float(max(1, epoch_loss_count))
        current_lr = scheduler.get_last_lr()[0]
        append_csv_row(
            epoch_log,
            epoch_header,
            {
                "epoch": epoch + 1,
                "mean_loss": mean_epoch_loss,
                "updates": global_update,
                "lr": current_lr,
            },
        )

        if CONFIG["save_every_epoch"]:
            save_path = os.path.join(CONFIG["save_dir"], "tapt_{}_epoch_{}.pth".format(model_type, epoch + 1))
            torch.save(checkpoint_state_dict(model), save_path)
            print("[TAPT] saved:", save_path)

        last_path = os.path.join(CONFIG["save_dir"], "tapt_{}_last.pth".format(model_type))
        torch.save(checkpoint_state_dict(model), last_path)
        print("[TAPT] epoch {} mean_loss={:.6f}".format(epoch + 1, mean_epoch_loss))

    print("[TAPT] done")
    print("[TAPT] detailed loss log:", detailed_log)
    print("[TAPT] epoch loss log:", epoch_log)
    print("[TAPT] final checkpoint:", os.path.join(CONFIG["save_dir"], "tapt_{}_last.pth".format(model_type)))


if __name__ == "__main__":
    train_foundation()
