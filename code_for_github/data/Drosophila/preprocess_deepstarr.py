"""
 DeepSTARR 
 FASTA () + TSV ()  Parquet 
 DataLoader 

Dev_log2_enrichment, Hk_log2_enrichment ()
"""

import os
import pandas as pd


def parse_fasta(fasta_path):
    """ FASTA """
    sequences = []
    current_seq = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))
    return sequences


def merge_fasta_tsv(fasta_path, tsv_path, output_path, target_cols):
    """
     FASTA  TSV  Parquet

    Args:
        fasta_path: FASTA 
        tsv_path: TSV 
        output_path:  Parquet 
        target_cols: 
    """
    print(f"  FASTA: {fasta_path}")
    sequences = parse_fasta(fasta_path)
    print(f"   >  {len(sequences)} ")

    print(f"  TSV: {tsv_path}")
    activity_df = pd.read_csv(tsv_path, sep="\t")
    print(f"   >  {len(activity_df)} , : {activity_df.columns.tolist()}")

    assert len(sequences) == len(activity_df), \
        f"  ({len(sequences)})  ({len(activity_df)}) "


    df = pd.DataFrame({"sequence": sequences})
    for col in target_cols:
        if col not in activity_df.columns:
            raise ValueError(f"  '{col}'  TSV : {activity_df.columns.tolist()}")
        df[col] = activity_df[col].values


    df.to_parquet(output_path, index=False)
    print(f" : {output_path} ({len(df)} , : {df.columns.tolist()})")
    print(f"   > : [{df['sequence'].str.len().min()}, {df['sequence'].str.len().max()}]")
    for col in target_cols:
        print(f"   > {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}, "
              f"min={df[col].min():.4f}, max={df[col].max():.4f}")
    return df


if __name__ == "__main__":
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    TARGET_COLS = ["Dev_log2_enrichment", "Hk_log2_enrichment"]

    # ==========================================

    # ==========================================
    splits = {
        "train": ("Sequences_Train.fa", "Sequences_activity_Train.txt"),
        "val":   ("Sequences_Val.fa",   "Sequences_activity_Val.txt"),
        "test":  ("Sequences_Test.fa",  "Sequences_activity_Test.txt"),
    }

    for split_name, (fasta_file, tsv_file) in splits.items():
        fasta_path = os.path.join(DATA_DIR, fasta_file)
        tsv_path = os.path.join(DATA_DIR, tsv_file)
        output_path = os.path.join(DATA_DIR, f"{split_name}.parquet")

        if not os.path.exists(fasta_path):
            print(f"  {split_name}: FASTA  ({fasta_path})")
            continue
        if not os.path.exists(tsv_path):
            print(f"  {split_name}: TSV  ({tsv_path})")
            continue

        print(f"\n{'='*50}")
        print(f"  {split_name.upper()} ")
        print(f"{'='*50}")
        merge_fasta_tsv(fasta_path, tsv_path, output_path, TARGET_COLS)

    print("\n ")
