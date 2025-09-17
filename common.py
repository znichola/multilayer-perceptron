from typing import Optional

import pandas as pd
import sys


def load_data(file_path: str = "data.csv") -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(file_path, header=None)
        return data
    except OSError:
        print(f"Error: Could not open file '{file_path}'")
        return None
    except pd.errors.ParserError:
        print("Error: File contents are not valid or improperly formatted")
        return None


def strip_ID(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(data.columns[0], axis=1)


def add_headers(data: pd.DataFrame) -> pd.DataFrame:
    col_names = ["diagnosis"] + [f"f{i}" for i in range(1, data.shape[1])]
    return data.set_axis(col_names, axis=1)


def load_seed(file_path="seed", default_seed: int=42, overwrite_seed: Optional[int]=None) -> int:
    def create_seed_file(seed: int):
        try:
            with open(file_path, "w") as f:
                f.write(f"{seed}\n")
        except OSError as e:
            print(f"Warning: Unable to write seed to {file_path}: {e}", file=sys.stderr)

    def is_number(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    seed: int

    if overwrite_seed is not None:
        seed = overwrite_seed
    else:
        try:
            with open(file_path, "r") as f:
                line = f.readline().strip()
                if not line or not is_number(line):
                    print(f"Warning: {file_path} contains invalid seed. Using default seed {default_seed}.",
                          file=sys.stderr)
                    seed = default_seed
                else:
                    seed = int(line)
        except OSError:
            print(f"Warning: {file_path} not found. Creating it with default seed {default_seed}.",
                  file=sys.stderr)
            seed = default_seed

    create_seed_file(seed)
    return seed


def load_data_or_throw(path: str) -> pd.DataFrame:
    df = load_data(path)
    if df is None:
        raise ValueError(f"Failed to load data from {path}")
    return add_headers(strip_ID(df))
