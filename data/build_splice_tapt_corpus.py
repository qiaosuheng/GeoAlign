from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


DEFAULT_SPLICE_TASKS = [
    "splice_sites_donors",
    "splice_sites_acceptors",
    "splice_sites_all",
]


# =============================================================================
# Configuration
# =============================================================================
# Server usage:
#   1. Edit this CONFIG block if needed.
#   2. Run: python data/build_splice_tapt_corpus.py
CONFIG = {
    # NT downstream benchmark root. It should contain splice_sites_* folders.
    "data_root": Path("/root/autodl-tmp/general/data"),

    # Output parquet for DatasetTAPT. It must contain a seq column.
    "output": Path("/root/autodl-tmp/general/data/tapt_splice_family/train.parquet"),

    # Splice-family tasks pooled for unlabeled TAPT.
    "tasks": DEFAULT_SPLICE_TASKS,

    # Prefer train_split.parquet when available; otherwise fallback to train.parquet.
    "train_split_candidates": ("train_split.parquet", "train.parquet"),

    # Held-out splits used only for overlap removal, never for TAPT training.
    "heldout_split_candidates": ("val.parquet", "valid.parquet", "validation.parquet", "test.parquet"),

    # False is recommended because donor/acceptor may overlap with splice_all.
    "keep_duplicates": False,

    # True is recommended to avoid any exact overlap with val/test sequences.
    "drop_heldout_overlap": True,

    # True saves only seq; False keeps source_task/source_split/source_index/seq_len for audit.
    "minimal": False,

    # True raises if any requested task is missing; False skips and warns.
    "require_all": False,
}


SEQ_CANDIDATES = ("seq", "sequence", "Sequence", "dna", "DNA")


def infer_seq_col(df: pd.DataFrame, task: str) -> str:
    for col in SEQ_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        "{}: cannot infer sequence column. Expected one of {}, got columns={}".format(
            task, SEQ_CANDIDATES, list(df.columns)
        )
    )


def normalize_sequence(seq: str) -> str:
    seq = str(seq).upper()
    return "".join(base if base in "ACGTN" else "N" for base in seq)


def find_first_existing(task_dir: Path, candidates: Tuple[str, ...]) -> Optional[Path]:
    for name in candidates:
        path = task_dir / name
        if path.exists():
            return path
    return None


def load_task_train(data_root: Path, task: str) -> pd.DataFrame:
    task_dir = data_root / task
    if not task_dir.exists():
        raise FileNotFoundError("{}: missing task directory: {}".format(task, task_dir))

    train_path = find_first_existing(task_dir, CONFIG["train_split_candidates"])
    if train_path is None:
        raise FileNotFoundError(
            "{}: missing train split. Tried {} under {}".format(
                task, CONFIG["train_split_candidates"], task_dir
            )
        )

    df = pd.read_parquet(train_path)
    seq_col = infer_seq_col(df, task)
    out = pd.DataFrame(
        {
            "seq": df[seq_col].map(normalize_sequence),
            "source_task": task,
            "source_split": train_path.name,
            "source_index": df.index.to_numpy(),
        }
    )
    out["seq_len"] = out["seq"].str.len()
    out = out[out["seq_len"] > 0].reset_index(drop=True)
    return out


def load_heldout_sequences(data_root: Path, tasks) -> set:
    heldout = set()
    for task in tasks:
        task_dir = data_root / task
        if not task_dir.exists():
            continue

        for name in CONFIG["heldout_split_candidates"]:
            path = task_dir / name
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            seq_col = infer_seq_col(df, task)
            heldout.update(df[seq_col].map(normalize_sequence).dropna().tolist())
            print("[HELDOUT] {} {:<22} n={:,}".format(task, name, len(df)))

    return heldout


def main() -> None:
    data_root = Path(CONFIG["data_root"])
    output = Path(CONFIG["output"])
    tasks = list(CONFIG["tasks"])
    keep_duplicates = bool(CONFIG["keep_duplicates"])
    drop_heldout_overlap = bool(CONFIG["drop_heldout_overlap"])
    minimal = bool(CONFIG["minimal"])
    require_all = bool(CONFIG["require_all"])

    print("[CONFIG]")
    print("  data_root:", data_root)
    print("  output:", output)
    print("  tasks:", ", ".join(tasks))
    print("  keep_duplicates:", keep_duplicates)
    print("  drop_heldout_overlap:", drop_heldout_overlap)
    print("  minimal:", minimal)
    print("  require_all:", require_all)
    print("")

    frames = []
    missing = []
    for task in tasks:
        try:
            task_df = load_task_train(data_root, task)
        except Exception as exc:
            if require_all:
                raise
            missing.append((task, str(exc)))
            print("[WARN] skip {}: {}".format(task, exc))
            continue

        frames.append(task_df)
        print(
            "[LOAD] {:<22} n={:>8,} len=[{}, {}] split={}".format(
                task,
                len(task_df),
                task_df["seq_len"].min(),
                task_df["seq_len"].max(),
                task_df["source_split"].iloc[0],
            )
        )

    if not frames:
        raise RuntimeError("No task data loaded. Please check CONFIG['data_root'] and CONFIG['tasks'].")

    corpus = pd.concat(frames, ignore_index=True)
    n_raw = len(corpus)

    if not keep_duplicates:
        corpus = corpus.drop_duplicates(subset=["seq"], keep="first").reset_index(drop=True)

    n_after_dedup = len(corpus)

    n_heldout_overlap = 0
    if drop_heldout_overlap:
        heldout = load_heldout_sequences(data_root, tasks)
        mask = corpus["seq"].isin(heldout)
        n_heldout_overlap = int(mask.sum())
        corpus = corpus.loc[~mask].reset_index(drop=True)

    if minimal:
        corpus = corpus[["seq"]]

    output.parent.mkdir(parents=True, exist_ok=True)
    corpus.to_parquet(output, index=False)

    print("\n[DONE] splice-family TAPT corpus saved")
    print("  output:", output)
    print("  loaded tasks: {} / {}".format(len(frames), len(tasks)))
    print("  raw sequences: {:,}".format(n_raw))
    print("  after dedup: {:,}".format(n_after_dedup))
    if drop_heldout_overlap:
        print("  dropped heldout overlaps: {:,}".format(n_heldout_overlap))
        print("  final sequences: {:,}".format(len(corpus)))
    if missing:
        print("\n[WARN] missing/skipped tasks:")
        for task, reason in missing:
            print("  - {}: {}".format(task, reason))


if __name__ == "__main__":
    main()
