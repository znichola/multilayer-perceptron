import argparse
import sys
from typing import Optional, Tuple

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import common as cm


def split_data(df: DataFrame, train_frac: float, seed: int) -> Tuple[DataFrame, DataFrame]:
    print(f"Using seed {seed} to split data for a train proportion of {train_frac}")

    df = df.sample(frac=1, random_state=seed)

    train_size = int(len(df) * train_frac)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    return train_df, val_df


def main():
    args = parser.parse_args()

    seed = cm.load_seed(overwrite_seed=args.seed)

    df = cm.load_data(args.file_path)
    if df is None:
        return
    
    df = cm.add_headers(cm.strip_ID(df))

    train_df, val_df = split_data(df, train_frac=args.train_frac, seed=seed)
    print(f"Training set: {len(train_df)} rows")
    print(pd.DataFrame({
        'count': train_df['diagnosis'].value_counts(),
        'proportion': train_df['diagnosis'].value_counts(normalize=True)
    }))

    print()

    print(f"Validation set: {len(val_df)} rows")
    print(pd.DataFrame({
        'count': val_df['diagnosis'].value_counts(),
        'proportion': val_df['diagnosis'].value_counts(normalize=True)
    }))

    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("validation.csv", index=False)

    print()

    print("Saved training set to 'train.csv' and validation set to 'validation.csv'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split data into training and validation sets")
    parser.add_argument("file_path", metavar="file_path.csv",
                        type=str, help="data file to describe")
    parser.add_argument("-f", "--train_frac", metavar="FRAC",
                        type=float, default=0.8,
                        help="Fraction of data to use for training (default 0.8)")
    parser.add_argument("--seed", metavar="SEED_VAL", type=int, default=42,
                        help="Value used for seed instead of reading from seed file")
    main()
