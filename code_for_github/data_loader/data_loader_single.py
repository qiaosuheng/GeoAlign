import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random


try:
    from .tokenizer import get_tokenizer
except ImportError:
    print(" Warning: Could not import Tokenizer.")


class DatasetBaseline(Dataset):
    """
     Baseline Dataset ()

    
    1.  (Single Stream):  input_ids input_ids_pair
    2.  (Optional Augmentation):  aug_rc / aug_shift 
        (RC /  / RC+)
        Baseline 
    3.  (Center Padding): 
    4.  :  (Classification)   (Regression)
    """

    def __init__(self, file_path, max_length=1024, model_type="hyenadna",
                 task_type="classification", target_cols=None,
                 aug_rc=False, aug_shift=False):
        """
        Args:
            aug_rc (bool):  RC () 
            aug_shift (bool):  (Shift) 
             (RC + Shift)
        """
        self.tokenizer = get_tokenizer(model_type, max_length=max_length)
        self.max_length = max_length

        self.task_type = task_type.lower()
        self.target_cols = target_cols
        self.aug_rc = aug_rc
        self.aug_shift = aug_shift

        self.pad_token_id = getattr(self.tokenizer, 'pad_token_id', 0)
        if self.pad_token_id is None:
            self.pad_token_id = 0

        try:
            self.df = pd.read_parquet(file_path)
        except Exception as e:
            raise FileNotFoundError(f" Parquet : {file_path}\n: {e}")


        columns = self.df.columns.tolist()
        self.seq_col = next((c for c in columns if 'seq' in c.lower()), columns[0])

        # ==========================================

        # ==========================================
        if self.task_type == "classification":
            self.label_col = next((c for c in columns if 'label' in c.lower() or 'target' in c.lower()), None)
            if not self.label_col and len(columns) > 1:
                self.label_col = columns[1]
        elif self.task_type == "regression":
            if not self.target_cols or not isinstance(self.target_cols, list):
                raise ValueError("  target_cols  ['Dev_log2fc', 'Hk_log2fc']")


        self.rc_map = str.maketrans("ACGTNacgtn", "TGCANtgcan")


        aug_status = []
        if self.aug_rc: aug_status.append("RC")
        if self.aug_shift: aug_status.append("Shift")
        if aug_status:
            print(f"    : {' + '.join(aug_status)}")
        else:
            print(f"     ( Baseline )")

    def __len__(self):
        return len(self.df)

    def get_rc(self, seq):
        """"""
        return seq.translate(self.rc_map)[::-1]

    def _get_complex_shift_val(self):
        """ ( DatasetDual )"""
        p = random.random()
        if p < 0.7:
            return random.randint(-5, 5)
        else:
            if random.random() < 0.5:
                return random.randint(-10, -6)
            else:
                return random.randint(6, 10)

    def apply_smart_shift(self, seq, shift_val):
        """ ( DatasetDual )"""
        if shift_val == 0: return seq
        if shift_val > 0:
            return "N" * shift_val + seq[:-shift_val]
        else:
            s = abs(shift_val)
            return seq[s:] + "N" * s

    def pad_and_mask(self, token_ids):
        """ Padding """
        curr_len = len(token_ids)
        if curr_len > self.max_length:
            input_ids = token_ids[:self.max_length]
            attention_mask = [1] * self.max_length
        else:
            pad_len = self.max_length - curr_len
            input_ids = token_ids + [self.pad_token_id] * pad_len
            attention_mask = [1] * curr_len + [0] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attention_mask, dtype=torch.long)

    def _apply_augmentation(self, seq_str):
        """
          ( DatasetDual )

         aug_rc  aug_shift 
        -  aug_shift: 50% 
        -  aug_rc:    50%  RC 
        - :     
                        ~33% , ~33% RC, ~33% RC+
        """
        if self.aug_rc and self.aug_shift:

            rand_val = random.random()
            if rand_val < 0.33:

                shift_val = self._get_complex_shift_val()
                seq_str = self.apply_smart_shift(seq_str, shift_val)
            elif rand_val < 0.66:

                seq_str = self.get_rc(seq_str)
            else:

                shift_val = self._get_complex_shift_val()
                seq_str = self.apply_smart_shift(self.get_rc(seq_str), shift_val)

        elif self.aug_rc:

            if random.random() < 0.5:
                seq_str = self.get_rc(seq_str)

        elif self.aug_shift:

            if random.random() < 0.5:
                shift_val = self._get_complex_shift_val()
                seq_str = self.apply_smart_shift(seq_str, shift_val)

        return seq_str

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_str = row[self.seq_col]

        # ==========================================

        # ==========================================
        if self.task_type == "classification":
            try:
                label = int(row[self.label_col])
            except:
                label = -1
            label_tensor = torch.tensor(label, dtype=torch.long)
        else:

            try:
                values = [float(row[col]) for col in self.target_cols]
            except Exception as e:
                raise ValueError(f"  {idx}  {self.target_cols}: {e}")

            label_tensor = torch.tensor(values, dtype=torch.float32)

        # ==========================================

        # ==========================================
        curr_len = len(seq_str)
        if curr_len > self.max_length:
            start = (curr_len - self.max_length) // 2
            seq_str = seq_str[start: start + self.max_length]
        elif curr_len < self.max_length:
            pad_total = self.max_length - curr_len
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            seq_str = "N" * pad_left + seq_str + "N" * pad_right

        # ==========================================

        # ==========================================
        seq_str = self._apply_augmentation(seq_str)

        # Tokenize
        tokens = self.tokenizer.encode(seq_str)
        input_ids, attention_mask = self.pad_and_mask(tokens)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor
        }
