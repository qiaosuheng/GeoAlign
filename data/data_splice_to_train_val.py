import pandas as pd
from sklearn.model_selection import train_test_split
import os

#tasks = ["promoter_all","enhancers","H3K4me3","enhancers_types","splice_sites_acceptor","splice_sites_donors","splice_sites_all","H3K4me1","H3K27ac","promoter_tata","promoter_no_tata","H3K9me3","H3K36me3"]
for task in tasks:
    train_file = f"/root/autodl-tmp/general/data/{task}/train.parquet"
    if os.path.exists(train_file):
        df = pd.read_parquet(train_file)

        train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

        train_df.to_parquet(f"/root/autodl-tmp/general/data/{task}/train_split.parquet")
        val_df.to_parquet(f"/root/autodl-tmp/general/data/{task}/val.parquet")
        print(f" {task} Train: {len(train_df)}, Val: {len(val_df)}")
    else:
        print(f" : {train_file}")