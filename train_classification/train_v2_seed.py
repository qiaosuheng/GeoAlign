import sys
import os
import gc

# ==========================================

# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score
from scipy.stats import pearsonr
import csv
import pandas as pd
import numpy as np
import random
from transformers import get_cosine_schedule_with_warmup

# ==========================================

# ==========================================
try:
    from models.model_select import build_model
    from data_loader.dataset_select import build_dataset
    from data_loader.tokenizer import get_tokenizer
except ImportError as e:
    print(f" : {e}")

# ==========================================

# ==========================================
task_name = "splice_sites_acceptors"

MODEL_PATHS = {
    "hyenadna": os.path.join(PROJECT_ROOT, "models", "hyenadna-small-32k-seqlen-hf"),
    "ntv3": os.path.join(PROJECT_ROOT, "models", "NTv3_8M_pre"),
    "caduceus": os.path.join(PROJECT_ROOT, "models", "caduceus-ps_seqlen-131k_d_model-256_n_layer-16")
}


SEEDS_TO_RUN = [40, 41, 42, 43, 44 ]

CONFIG = {

    "arch_type": "siamese",
    "model_type": "caduceus",


    "foundation_ckpt": None,
    "train_path": os.path.join(PROJECT_ROOT, "data", task_name, "train_split.parquet"),
    "test_path": os.path.join(PROJECT_ROOT, "data", task_name, "val.parquet"),

    "batch_size": 16,
    "grad_accumulation_steps": 8,
    "val_batch_size": 64,
    "lora_r": 16,

    "val_check_interval": 100,

    "lr_base": 2e-4,
    "lr_head": 2e-4,
    "weight_decay": 0.05,
    "epochs": 20,
    "warmup_ratio": 0.10,
    "label_smoothing": 0.1,


    "lambda_align": 1.0,


    "use_rc": True,
    "use_shift": False,

    "seq_len": 1024,
    "num_classes": 2,
}


# ==========================================

# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"\n Random Seed set to: {seed}")


def center_pad_or_crop(seq_str, max_len=1024):
    curr_len = len(seq_str)
    if curr_len > max_len:
        start = (curr_len - max_len) // 2
        return seq_str[start: start + max_len]
    elif curr_len < max_len:
        pad_total = max_len - curr_len
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return "N" * pad_left + seq_str + "N" * pad_right
    else:
        return seq_str


def get_rc_sequence(seq):
    rc_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return seq.translate(rc_map)[::-1]


def calculate_rc_consistency_siamese(model, test_path, device, model_type="hyenadna", num_samples=1000, seq_len=1024,
                                     seed=42):
    model.eval()
    try:
        df = pd.read_parquet(test_path)
        cols = df.columns.tolist()
        seq_col = next((c for c in cols if 'seq' in c.lower()), cols[0])
        if len(df) > num_samples:
            df = df.sample(num_samples, random_state=seed)
        sequences = df[seq_col].tolist()
    except Exception:
        return 0.0

    tokenizer = get_tokenizer(model_type, max_length=seq_len)
    probs_fwd, probs_rc = [], []
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_raw = sequences[i: i + batch_size]
            batch_str_fwd = [center_pad_or_crop(s, seq_len) for s in batch_raw]
            batch_str_rc = [get_rc_sequence(s) for s in batch_str_fwd]

            tokens_fwd = tokenizer(batch_str_fwd, return_tensors="pt", padding=False, truncation=True)
            tokens_rc = tokenizer(batch_str_rc, return_tensors="pt", padding=False, truncation=True)

            ids_fwd = tokens_fwd["input_ids"].to(device)
            mask_fwd = tokens_fwd["attention_mask"].to(device)
            ids_rc = tokens_rc["input_ids"].to(device)
            mask_rc = tokens_rc["attention_mask"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits_fwd, _ = model.forward_one_branch(ids_fwd, mask_fwd)
                logits_rc, _ = model.forward_one_branch(ids_rc, mask_rc)

                p_fwd = F.softmax(logits_fwd.float(), dim=1).cpu().numpy()
                p_rc = F.softmax(logits_rc.float(), dim=1).cpu().numpy()

            probs_fwd.append(p_fwd)
            probs_rc.append(p_rc)

    if len(probs_fwd) == 0:
        return 0.0

    probs_fwd = np.concatenate(probs_fwd, axis=0)
    probs_rc = np.concatenate(probs_rc, axis=0)

    if probs_fwd.shape[0] <= 1:
        return 0.0

    class_corrs = []
    for c in range(probs_fwd.shape[1]):
        if np.std(probs_fwd[:, c]) > 0 and np.std(probs_rc[:, c]) > 0:
            class_corrs.append(pearsonr(probs_fwd[:, c], probs_rc[:, c])[0])

    return float(np.mean(class_corrs)) if class_corrs else 0.0


def validate_siamese(model, test_loader, device, criterion_cls):
    model.eval()
    val_preds, val_labels = [], []
    val_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=" Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _ = model.forward_one_branch(input_ids, mask)
                loss = criterion_cls(logits.float(), labels)

            val_loss += loss.item()
            preds = torch.argmax(logits.float(), 1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
            batch_count += 1

    return matthews_corrcoef(val_labels, val_preds), accuracy_score(val_labels,
                                                                    val_preds), val_loss / batch_count if batch_count > 0 else 0.0


# ==========================================

# ==========================================
def train_single_seed(current_seed):
    """ Seed """
    set_seed(current_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    experiment_name = f"step2_{CONFIG['arch_type']}_{CONFIG['model_type']}_lambda={CONFIG['lambda_align']}_seed={current_seed}_shift={CONFIG['use_shift']}_rc={CONFIG['use_rc']}"
    save_dir = os.path.join(PROJECT_ROOT, "checkpoints", task_name, experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    print(
        f" Training Seed: {current_seed} | Arch: {CONFIG['arch_type'].upper()} | Model: {CONFIG['model_type'].upper()}")
    print(f" Saving logs to: {save_dir}")

    # ==========================================

    # ==========================================
    print("\n" + "=" * 40)
    print(f" [Augmentation Strategy]")
    print(f"    RC : {' ' if CONFIG['use_rc'] else ' '}")
    print(f"    Shift : {' ' if CONFIG['use_shift'] else ' '}")
    print("=" * 40 + "\n")


    # ==========================================

    # ==========================================
    train_ds = build_dataset(CONFIG["train_path"], CONFIG["arch_type"], CONFIG["model_type"], CONFIG["seq_len"],
                             use_rc=CONFIG["use_rc"], use_shift=CONFIG["use_shift"])
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=8, pin_memory=True)

    test_ds = build_dataset(CONFIG["test_path"], CONFIG["arch_type"], CONFIG["model_type"], CONFIG["seq_len"],
                            use_rc=CONFIG["use_rc"], use_shift=CONFIG["use_shift"])
    test_loader = DataLoader(test_ds, batch_size=CONFIG["val_batch_size"], shuffle=False, num_workers=4,
                             pin_memory=True)

    # ==========================================

    # ==========================================
    model_dir = MODEL_PATHS.get(CONFIG["model_type"])
    model = build_model(
        model_type=CONFIG["model_type"],
        arch_type=CONFIG["arch_type"],
        model_dir=model_dir,
        lora_r=CONFIG["lora_r"],
        num_classes=CONFIG["num_classes"],
        device=device
    )

    if CONFIG["foundation_ckpt"] and os.path.exists(CONFIG["foundation_ckpt"]):
        state_dict = torch.load(CONFIG["foundation_ckpt"], map_location=device)
        try:
            model.base_model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f" Failed to load TAPT weights: {e}")

    # ==========================================
    # ==========================================

    # ==========================================

    # ==========================================
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("head" in n or "unembed" in n)]
    base_params = [p for n, p in model.named_parameters() if p.requires_grad and not ("head" in n or "unembed" in n)]
    optimizer = optim.AdamW(
        [{'params': base_params, 'lr': CONFIG['lr_base']}, {'params': head_params, 'lr': CONFIG['lr_head']}],
        weight_decay=CONFIG['weight_decay'])

    total_steps = (len(train_loader) // CONFIG["grad_accumulation_steps"]) * CONFIG["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(CONFIG["warmup_ratio"] * total_steps), num_training_steps=total_steps
    )

    criterion_cls = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    criterion_align = nn.MSELoss()


    log_file = os.path.join(save_dir, "log_detailed.csv")
    with open(log_file, "w") as f:
        csv.writer(f).writerow(
            ["Epoch", "Global_Step", "Train_Loss", "Align_Loss", "Val_Loss", "Val_MCC", "Val_Acc", "Train_Acc", "LR",
             "RC_Pearson"])

    best_mcc = -1.0
    global_step = 0

    print(f" Start Training Seed {current_seed}...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        stats = {"correct": 0, "n": 0}
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"S{current_seed}-Ep{epoch + 1}/{CONFIG['epochs']}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            input_ids_pair = batch["input_ids_pair"].to(device)
            mask = batch["attention_mask"].to(device)
            mask_pair = batch["attention_mask_pair"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits_anchor, logits_pair, feat_anchor, feat_pair = model(
                    input_ids, input_ids_pair, mask, mask_pair, return_feats=True
                )
                loss_cls = (criterion_cls(logits_anchor.float(), labels) + criterion_cls(logits_pair.float(),
                                                                                         labels)) / 2
                loss_align = criterion_align(feat_anchor.float(), feat_pair.float())
                loss = (loss_cls + CONFIG["lambda_align"] * loss_align) / CONFIG["grad_accumulation_steps"]

                preds = torch.argmax(logits_anchor, dim=1)
                stats["correct"] += (preds == labels).sum().item()
                stats["n"] += labels.size(0)

            loss.backward()

            current_train_loss_val = loss.item() * CONFIG["grad_accumulation_steps"]
            current_align_loss_val = loss_align.item()

            if (step + 1) % CONFIG["grad_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"Loss": f"{current_train_loss_val:.3f}", "Align": f"{current_align_loss_val:.3f}"})

                if global_step % CONFIG["val_check_interval"] == 0:
                    val_mcc, val_acc, val_loss = validate_siamese(model, test_loader, device, criterion_cls)
                    rc_pearson = calculate_rc_consistency_siamese(model, CONFIG["test_path"], device,
                                                                  model_type=CONFIG["model_type"], num_samples=1000,
                                                                  seed=current_seed)

                    with open(log_file, "a") as f:
                        csv.writer(f).writerow([
                            epoch + 1, global_step, f"{current_train_loss_val:.4f}", f"{current_align_loss_val:.4f}",
                            f"{val_loss:.4f}", f"{val_mcc:.4f}", f"{val_acc:.4f}",
                            f"{stats['correct'] / stats['n']:.4f}", f"{current_lr:.1e}", f"{rc_pearson:.4f}"
                        ])

                    if val_mcc > best_mcc:
                        best_mcc = val_mcc
                        torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                        pbar.set_postfix_str(f" New Best: {best_mcc:.4f}")


        del input_ids, input_ids_pair, mask, mask_pair, labels, logits_anchor, logits_pair, feat_anchor, feat_pair, loss
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    print(f"\n Seed {current_seed} Done! Best MCC: {best_mcc:.4f}")

    # ==========================================

    # ==========================================
    print(f" Clearing memory for Seed {current_seed}...")

    del model
    del optimizer
    del scheduler
    del criterion_cls
    del criterion_align

    del train_loader
    del test_loader
    del train_ds
    del test_ds


    gc.collect()


    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.ipc_collect()

    print(f" Memory cleared. Ready for next seed.\n")


if __name__ == "__main__":
    print(f" Starting Auto-Batch Training for {len(SEEDS_TO_RUN)} seeds: {SEEDS_TO_RUN}")
    print("=" * 60)

    for seed in SEEDS_TO_RUN:
        try:
            train_single_seed(seed)
        except Exception as e:
            print(f"\n : Seed {seed} ! : {e}")

            torch.cuda.empty_cache()
            continue

    print("\n ")