import argparse
from typing import Optional

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt

import common as cm

def main():
    args = parser.parse_args()

    df = cm.load_data(args.file_path)
    if df is None:
        return

    print("\nData overview")
    print(df)
    print("\nData describe")
    print(df.describe(include="all"))
    print("\nData shape")
    print(df.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe raw data")
    parser.add_argument("file_path", metavar="file_path.csv",
                        type=str, help="data file to describe")
    main()

