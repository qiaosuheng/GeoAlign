import torch
import torch.nn as nn

class hyenadnaTokenizer:
    """
     HyenaDNA  Tokenizer
     AutoTokenizer  [EOS] 
     Shift  100% 
    """
    def __init__(self, max_length=1024):

        self.pad_token_id = 4
        self.max_length = max_length

        self.vocab_map = {'A': 7, 'C': 8, 'G': 9, 'T': 10, 'N': 11}

    def encode(self, text):

        return [self.vocab_map.get(c, 11) for c in text.upper()]

    def __call__(self, text_list, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        if isinstance(text_list, str): text_list = [text_list]
        max_len = max_length if max_length else self.max_length
        batch_ids = []
        batch_masks = []

        for text in text_list:
            ids = self.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding and len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_masks, dtype=torch.long)
            }
        return batch_ids

class ntv3Tokenizer:
    """
     NTv3 (Nucleotide Transformer v3)  Tokenizer
     <cls>  <bos>  RC 
    """
    def __init__(self, max_length=1024):
        self.pad_token_id = 1
        self.unk_token_id = 0
        self.mask_token_id = 2
        self.max_length = max_length

        self.vocab_map = {'A': 6, 'T': 7, 'C': 8, 'G': 9, 'N': 10}

    def encode(self, text):

        return [self.vocab_map.get(c, 10) for c in text.upper()]

    def __call__(self, text_list, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        if isinstance(text_list, str): text_list = [text_list]
        max_len = max_length if max_length else self.max_length
        batch_ids = []
        batch_masks = []

        for text in text_list:
            ids = self.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding and len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_masks, dtype=torch.long)
            }
        return batch_ids


class caduceusTokenizer:
    """
     Caduceus  Tokenizer
     [CLS] / [SEP]  RC 
    :
      [CLS]=0, [SEP]=1, [BOS]=2, [MASK]=3, [PAD]=4, [RESERVED]=5, [UNK]=6
      A=7, C=8, G=9, T=10, N=11
    """
    def __init__(self, max_length=1024):
        self.pad_token_id = 4
        self.unk_token_id = 6
        self.mask_token_id = 3
        self.max_length = max_length

        self.vocab_map = {'A': 7, 'C': 8, 'G': 9, 'T': 10, 'N': 11}

    def encode(self, text):

        return [self.vocab_map.get(c, 11) for c in text.upper()]

    def __call__(self, text_list, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        if isinstance(text_list, str): text_list = [text_list]
        max_len = max_length if max_length else self.max_length
        batch_ids = []
        batch_masks = []

        for text in text_list:
            ids = self.encode(text)
            if len(ids) > max_len:
                ids = ids[:max_len]
            mask = [1] * len(ids)
            if padding and len(ids) < max_len:
                pad_len = max_len - len(ids)
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            batch_ids.append(ids)
            batch_masks.append(mask)

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_masks, dtype=torch.long)
            }
        return batch_ids



# =================================================================

# =================================================================
def get_tokenizer(model_type, max_length=1024):
    """
     Tokenizer DataLoader 
    """
    model_type = model_type.lower()
    if model_type == "hyenadna":
        return hyenadnaTokenizer(max_length=max_length)
    elif model_type in ["ntv3", "nt"]:
        return ntv3Tokenizer(max_length=max_length)
    elif model_type == "caduceus":
        return caduceusTokenizer(max_length=max_length)
    else:
        raise ValueError(f" : {model_type}")

