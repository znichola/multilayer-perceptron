from typing import Optional

import pandas as pd


def load_data(file_path: str = "data.csv") -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(file_path, header=None)
        col_names = ["diagnosis" if i == 1 else f"f{i+1}"
        for i in range(data.shape[1])]
        return data.set_axis(col_names, axis=1)
    except OSError:
        print(f"Error: Could not open file '{file_path}'")
        return None
    except pd.errors.ParserError:
        print("Error: File contents are not valid or improperly formatted")
        return None


def load_data_or_throw(path: str) -> pd.DataFrame:
    df = load_data(path)
    if df is None:
        raise ValueError(f"Failed to load data from {path}")
    return df