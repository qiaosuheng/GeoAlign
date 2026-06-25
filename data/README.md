# Data placeholders

This directory contains only preprocessing scripts and placeholder folders. Dataset files are intentionally excluded from the GitHub-ready copy.

Expected classification files:

- `train_split.parquet`
- `val.parquet`
- `test.parquet`

Expected columns:

- `sequence`: DNA sequence string
- `label`: integer class label

Expected DeepSTARR-style regression files:

- `train.parquet`
- `val.parquet`
- `test.parquet`

Expected columns:

- `sequence`
- `Dev_log2_enrichment`
- `Hk_log2_enrichment`

Use `Drosophila/preprocess_deepstarr.py` to convert the original FASTA/activity files into parquet format.
