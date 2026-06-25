import os

try:

    from .data_loader_single import DatasetBaseline
    from .data_loader_dual import DatasetDual
    from .data_loader_tapt import DatasetTAPT
except ImportError as e:
    print(f" [Data Factory]  data_loader : {e}")


def build_dataset(data_path, arch_type, model_type="hyenadna", max_length=1024, use_rc=True, use_shift=True,
                  task_type="classification", target_cols=None,
                  aug_rc=False, aug_shift=False, **kwargs):
    """
      (Dataset Factory)
     arch_type  TAPT  DataLoader
     model_type  Tokenizer

    Args:
        data_path (str):  (Parquet / Fasta )
        arch_type (str): "baseline" (), "siamese" (),  "tapt"
        model_type (str): Backbone tokenizer type, for example "hyenadna", "ntv3", or "caduceus".
        max_length (int): 
        use_rc (bool): []  RC () Splice Site  False
        use_shift (bool): []  (Shift) 
        task_type (str): "classification" ()  "regression" ( DeepSTARR)
        target_cols (list):  ['Dev_log2fc', 'Hk_log2fc']
        aug_rc (bool): [ baseline]  RC 
        aug_shift (bool): [ baseline] 
    """
    arch_type = arch_type.lower()
    model_type = model_type.lower()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f" [Data Factory] : {data_path}")

    task_info = f" (Classification)" if task_type == "classification" else f" (Regression) Targets: {target_cols}"


    if arch_type == "baseline":
        aug_parts = []
        if aug_rc: aug_parts.append("RC")
        if aug_shift: aug_parts.append("Shift")
        aug_info = f" | : {'+'.join(aug_parts)}" if aug_parts else ""
        print(f" [Data Factory]  | : {model_type.upper()} | : {task_info}{aug_info}")
        dataset = DatasetBaseline(
            file_path=data_path, model_type=model_type, max_length=max_length,
            task_type=task_type, target_cols=target_cols,
            aug_rc=aug_rc, aug_shift=aug_shift, **kwargs
        )


    elif arch_type in ["siamese", "ours", "dual"]:
        rc_status = "" if use_rc else ""
        shift_status = "" if use_shift else ""
        print(f" [Data Factory]  | : {model_type.upper()} | RC: {rc_status} | Shift: {shift_status} | : {task_info}")
        dataset = DatasetDual(
            file_path=data_path, model_type=model_type, max_length=max_length,
            use_rc=use_rc, use_shift=use_shift,
            task_type=task_type, target_cols=target_cols, **kwargs
        )


    elif arch_type == "tapt":
        print(f" [Data Factory]  TAPT  | : {model_type.upper()}")
        dataset = DatasetTAPT(file_path=data_path, model_type=model_type, max_length=max_length, **kwargs)

    else:
        raise ValueError(f" [Data Factory] : {arch_type}")

    return dataset
