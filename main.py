import argparse
from typing import Optional

import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


def loadData(file_path: str = "data.csv") -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(file_path, header=None)
        return data
    except OSError:
        print(f"Error: Could not open file '{file_path}'")
        return None
    except pd.errors.ParserError:
        print("Error: File contents are not valid or improperly formatted")
        return None


def appendHeader(df: DataFrame) -> DataFrame:
    col_names = ["diagnosis" if i == 1 else f"f{i+1}"
                 for i in range(df.shape[1])]
    return df.set_axis(col_names, axis=1)


def main():
    args = parser.parse_args()

    df = loadData(args.file_path)
    if df is None:
        return
    df = appendHeader(df)

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

