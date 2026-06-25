import random

import pandas as pd
import torch
from torch.utils.data import Dataset

try:
    from .tokenizer import get_tokenizer
except ImportError:
    from tokenizer import get_tokenizer


SEQ_CANDIDATES = ("seq", "sequence", "Sequence", "dna", "DNA")


class DatasetTAPT(Dataset):
    def __init__(self, file_path, max_length=1024, model_type="hyenadna"):
        self.tokenizer = get_tokenizer(model_type, max_length=max_length)
        self.max_length = max_length
        self.model_type = model_type.lower()
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        self.masked_lm = self.model_type in ["ntv3", "caduceus"]
        self.mask_token_id = self._resolve_mask_token_id()

        try:
            self.df = pd.read_parquet(file_path)
        except Exception as e:
            raise FileNotFoundError("Failed to read parquet: {}\n{}".format(file_path, e))

        self.seq_col = self._infer_seq_col(self.df)
        self.rc_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")

    def _infer_seq_col(self, df):
        for col in SEQ_CANDIDATES:
            if col in df.columns:
                return col
        raise ValueError(
            "Cannot infer sequence column. Expected one of {}, got {}".format(
                SEQ_CANDIDATES, list(df.columns)
            )
        )

    def _resolve_mask_token_id(self):
        if not self.masked_lm:
            return None

        mask_token_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_token_id is not None:
            return mask_token_id

        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            for token in ("<mask>", "[MASK]", "MASK"):
                try:
                    token_id = self.tokenizer.convert_tokens_to_ids(token)
                except Exception:
                    token_id = None
                if token_id is not None and token_id != getattr(self.tokenizer, "unk_token_id", None):
                    return token_id

        if self.model_type == "ntv3":
            return 2
        if self.model_type == "caduceus":
            return 3
        raise ValueError("Cannot resolve mask_token_id for {}".format(self.model_type))

    def __len__(self):
        return len(self.df)

    def _normalize_sequence(self, seq):
        seq = str(seq).upper()
        return "".join(base if base in "ACGTN" else "N" for base in seq)

    def get_rc(self, seq):
        return seq.translate(self.rc_map)[::-1]

    def apply_smart_shift(self, seq, shift_val):
        if shift_val == 0:
            return seq
        if shift_val > 0:
            return "N" * shift_val + seq[:-shift_val]
        shift_abs = abs(shift_val)
        return seq[shift_abs:] + "N" * shift_abs

    def center_pad_or_crop(self, seq):
        curr_len = len(seq)
        if curr_len > self.max_length:
            start = (curr_len - self.max_length) // 2
            return seq[start:start + self.max_length]
        if curr_len < self.max_length:
            pad_total = self.max_length - curr_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return "N" * pad_left + seq + "N" * pad_right
        return seq

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_str = self._normalize_sequence(row[self.seq_col])
        seq_str = self.center_pad_or_crop(seq_str)

        # TAPT uses the same simple perturbation exposure as the old script.
        # if random.random() < 0.5:
        #     seq_str = self.apply_smart_shift(seq_str, random.randint(-5, 5))
        # if random.random() < 0.3:
        #     seq_str = self.get_rc(seq_str)

        token_ids = self.tokenizer.encode(seq_str)
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        elif len(token_ids) < self.max_length:
            token_ids = token_ids + [self.pad_token_id] * (self.max_length - len(token_ids))

        input_ids = torch.tensor(token_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.pad_token_id] = -100
        attention_mask = (input_ids != self.pad_token_id).long()

        if self.masked_lm:
            probability_matrix = torch.full(labels.shape, 0.15)
            probability_matrix.masked_fill_(labels == -100, value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()

            labels[~masked_indices] = -100
            input_ids[masked_indices] = self.mask_token_id
            attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }
