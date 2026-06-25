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
from scipy.stats import pearsonr, spearmanr
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
task_name = "Drosophila"

MODEL_PATHS = {
    "hyenadna": os.path.join(PROJECT_ROOT, "models", "hyenadna-small-32k-seqlen-hf"),
    "ntv3": os.path.join(PROJECT_ROOT, "models", "NTv3_8M_pre"),
    "caduceus": os.path.join(PROJECT_ROOT, "models", "caduceus-ps_seqlen-131k_d_model-256_n_layer-16")
}


TARGET_COLS = ["Dev_log2_enrichment", "Hk_log2_enrichment"]
NUM_TARGETS = len(TARGET_COLS)


SEEDS = [40, 41, 42, 43, 44]

CONFIG = {



    "arch_type": "baseline",
    "model_type": "caduceus",


    "foundation_ckpt": None,

    "train_path": os.path.join(PROJECT_ROOT, "data", task_name, "train.parquet"),
    "test_path": os.path.join(PROJECT_ROOT, "data", task_name, "val.parquet"),

    "batch_size": 32,
    "grad_accumulation_steps": 4,
    "lora_r": 16,
    "lr_base": 2e-4,
    "lr_head": 2e-4,
    "weight_decay": 0.05,
    "epochs": 5,
    "warmup_ratio": 0.10,
    "seq_len": 1024,
    "num_targets": NUM_TARGETS,

    "val_check_interval": 500,

    "aug_rc": True,
    "aug_shift": True,
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
    print(f" Random Seed set to: {seed}")


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


def calculate_rc_consistency_regression(model, val_df, device, model_type="hyenadna",
                                        num_samples=1000, seq_len=1024, target_cols=None):
    """
     RC  Pearson 
    
    """
    model.eval()
    try:
        cols = val_df.columns.tolist()
        seq_col = next((c for c in cols if 'seq' in c.lower()), cols[0])

        if len(val_df) > num_samples:
            df_sample = val_df.sample(num_samples, random_state=CONFIG["seed"])
        else:
            df_sample = val_df
        sequences = df_sample[seq_col].tolist()
    except Exception as e:
        print(f" RC Consistency Eval Error: {e}")
        return 0.0

    tokenizer = get_tokenizer(model_type, max_length=seq_len)
    preds_fwd = {i: [] for i in range(NUM_TARGETS)}
    preds_rc = {i: [] for i in range(NUM_TARGETS)}
    batch_size = 16

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
                out_fwd = model(ids_fwd, attention_mask=mask_fwd).float().cpu().numpy()
                out_rc = model(ids_rc, attention_mask=mask_rc).float().cpu().numpy()

            for t in range(NUM_TARGETS):
                preds_fwd[t].extend(out_fwd[:, t])
                preds_rc[t].extend(out_rc[:, t])

    pearson_values = []
    for t in range(NUM_TARGETS):
        if len(preds_fwd[t]) > 1:
            r, _ = pearsonr(preds_fwd[t], preds_rc[t])
            pearson_values.append(r)
    return np.mean(pearson_values) if pearson_values else 0.0


def validate_regression(model, test_loader, device, criterion):
    """
     MSE, Pearson-r, Spearman- (per target)
    """
    model.eval()
    all_preds = []
    all_labels = []
    val_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=" Validating", leave=False):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)  # [Batch, NUM_TARGETS]

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask=mask)
                loss = criterion(preds.float(), labels)

            val_loss += loss.item()
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            batch_count += 1

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = {}
    pearson_list = []
    for t in range(NUM_TARGETS):
        col_name = TARGET_COLS[t] if TARGET_COLS else f"target_{t}"
        p_r, _ = pearsonr(all_labels[:, t], all_preds[:, t])
        s_r, _ = spearmanr(all_labels[:, t], all_preds[:, t])
        mse = np.mean((all_labels[:, t] - all_preds[:, t]) ** 2)
        metrics[col_name] = {"pearson": p_r, "spearman": s_r, "mse": mse}
        pearson_list.append(p_r)

    avg_pearson = np.mean(pearson_list)
    avg_loss = val_loss / batch_count if batch_count > 0 else 0.0

    return metrics, avg_pearson, avg_loss



def train(current_seed, val_df_in_memory):

    CONFIG["seed"] = current_seed
    experiment_name = f"reg_baseline_{CONFIG['arch_type']}_{CONFIG['model_type']}_seed={current_seed}_aug_shift={CONFIG['aug_shift']}_rc={CONFIG['aug_rc']}"
    CONFIG["save_dir"] = os.path.join(PROJECT_ROOT, "checkpoints", task_name, experiment_name)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print(f" [SEED {current_seed}] Regression Training (Baseline)")
    print(f"   Arch: {CONFIG['arch_type'].upper()} | Model: {CONFIG['model_type'].upper()}")
    print(f"   Logs: {CONFIG['save_dir']}")
    print("=" * 60)


    train_ds = build_dataset(
        CONFIG["train_path"], CONFIG["arch_type"], CONFIG["model_type"], CONFIG["seq_len"],
        task_type="regression", target_cols=TARGET_COLS,
        aug_rc=CONFIG["aug_rc"], aug_shift=CONFIG["aug_shift"]
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)

    test_ds = build_dataset(
        CONFIG["test_path"], CONFIG["arch_type"], CONFIG["model_type"], CONFIG["seq_len"],
        task_type="regression", target_cols=TARGET_COLS
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)


    model_dir = MODEL_PATHS.get(CONFIG["model_type"])
    if not model_dir:
        raise ValueError(f" : {CONFIG['model_type']} MODEL_PATHS ")

    model = build_model(
        model_type=CONFIG["model_type"],
        arch_type=CONFIG["arch_type"],
        model_dir=model_dir,
        lora_r=CONFIG["lora_r"],
        num_classes=CONFIG["num_targets"],
        device=device
    )


    if CONFIG["foundation_ckpt"] and os.path.exists(CONFIG["foundation_ckpt"]):
        print(f" Loading TAPT Weights: {CONFIG['foundation_ckpt']}")
        state_dict = torch.load(CONFIG["foundation_ckpt"], map_location=device)
        try:
            model.base_model.load_state_dict(state_dict, strict=True)
            print(" TAPT Weights loaded successfully into base_model.")
        except Exception as e:
            print(f" Failed to load TAPT weights: {e}")
    else:
        pass



    head_params = [p for n, p in model.named_parameters() if p.requires_grad and ("head" in n or "unembed" in n)]
    base_params = [p for n, p in model.named_parameters() if p.requires_grad and not ("head" in n or "unembed" in n)]
    optimizer = optim.AdamW(
        [{'params': base_params, 'lr': CONFIG['lr_base']}, {'params': head_params, 'lr': CONFIG['lr_head']}],
        weight_decay=CONFIG['weight_decay'])

    total_steps = (len(train_loader) // CONFIG["grad_accumulation_steps"]) * CONFIG["epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(CONFIG["warmup_ratio"] * total_steps), num_training_steps=total_steps
    )

    criterion = nn.MSELoss()


    log_file = os.path.join(CONFIG["save_dir"], "log_detailed.csv")
    header = ["Epoch", "Global_Step", "Train_Loss", "Val_Loss", "Val_Pearson_Avg"]
    for col in TARGET_COLS:
        header.extend([f"Pearson_{col}", f"Spearman_{col}", f"MSE_{col}"])
    header.extend(["LR", "RC_Pearson"])
    with open(log_file, "w") as f:
        csv.writer(f).writerow(header)

    best_pearson = -1.0
    global_step = 0

    print(" Start Training...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        running_loss = 0.0
        running_n = 0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Ep {epoch + 1}/{CONFIG['epochs']}")

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds = model(input_ids, attention_mask=mask)
                loss = criterion(preds.float(), labels) / CONFIG["grad_accumulation_steps"]

            loss.backward()

            current_train_loss_val = loss.item() * CONFIG["grad_accumulation_steps"]

            # --- Optimizer Step ---
            if (step + 1) % CONFIG["grad_accumulation_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                pbar.set_postfix({"Loss": f"{current_train_loss_val:.4f}", "LR": f"{current_lr:.1e}"})

                if global_step % CONFIG["val_check_interval"] == 0:
                    metrics, avg_pearson, val_loss = validate_regression(model, test_loader, device, criterion)

                    rc_pearson = calculate_rc_consistency_regression(
                        model, val_df_in_memory, device,
                        model_type=CONFIG["model_type"], num_samples=500
                    )

                    row = [epoch + 1, global_step, f"{current_train_loss_val:.4f}", f"{val_loss:.4f}",
                           f"{avg_pearson:.4f}"]
                    for col in TARGET_COLS:
                        m = metrics[col]
                        row.extend([f"{m['pearson']:.4f}", f"{m['spearman']:.4f}", f"{m['mse']:.4f}"])
                    row.extend([f"{current_lr:.1e}", f"{rc_pearson:.4f}"])

                    with open(log_file, "a") as f:
                        csv.writer(f).writerow(row)

                    metric_str = " | ".join([f"{col}: r={metrics[col]['pearson']:.4f}" for col in TARGET_COLS])
                    print(f"\n    Step {global_step} | Val Loss: {val_loss:.4f} | {metric_str} | RC: {rc_pearson:.4f}")

                    if avg_pearson > best_pearson:
                        best_pearson = avg_pearson
                        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
                        pbar.set_postfix_str(f" Best Pearson: {best_pearson:.4f}")

        # --- Epoch End ---
        rc_pearson_full = calculate_rc_consistency_regression(
            model, val_df_in_memory, device,
            model_type=CONFIG["model_type"], num_samples=2000
        )

        with open(log_file, "a") as f:
            row = [epoch + 1, "End", f"{current_train_loss_val:.4f}", "0.0", "0.0"]
            for _ in TARGET_COLS:
                row.extend(["0.0", "0.0", "0.0"])
            row.extend([f"{current_lr:.1e}", f"{rc_pearson_full:.4f}"])
            csv.writer(f).writerow(row)

    torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "final_model.pth"))
    print(f" [SEED {current_seed}] Done! Best Avg Pearson: {best_pearson:.4f}\n")


    del model
    del optimizer
    del train_loader
    del test_loader
    return True


if __name__ == "__main__":

    print(" Loading Validation Data into memory for multi-seed eval...")
    val_df_in_memory = pd.read_parquet(CONFIG["test_path"])
    print(f" Validation Data Loaded! Rows: {len(val_df_in_memory)}\n")


    for seed in SEEDS:
        try:
            train(seed, val_df_in_memory)
        except Exception as e:
            print(f" Error during Seed {seed}: {e}")
        finally:
            # ==========================================

            # ==========================================
            print(f" Finishing Seed {seed}, initiating hardware cleanup...")


            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


            gc.collect()

            print(f" Hardware reset complete. Ready for next seed.\n")

    print(" All seeds training sessions completed safely.")
