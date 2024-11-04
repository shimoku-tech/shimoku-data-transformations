from typing import List, Callable, Optional, Dict, Union
import json

import pandas as pd
import numpy as np


def create_col_dummy(
        df: pd.DataFrame, col_name: str, force_int: float = False, drop: bool = False) -> pd.DataFrame:
    dummy_df = pd.get_dummies(df[col_name], prefix=col_name, dummy_na=True)

    if force_int:
        # Rename columns to use integers instead of floats
        new_column_names = {col: col.split('_')[0] + '_' + col.split('_')[-1].split('.')[0]
                            for col in dummy_df.columns if not col.endswith('_nan')}
        dummy_df = dummy_df.rename(columns=new_column_names)

    # Rename the NaN column to something more readable
    dummy_df = dummy_df.rename(columns={f'{col_name}_nan': f'{col_name}_missing'})

    # Convert float columns to boolean
    for col in dummy_df.columns:
        dummy_df[col] = dummy_df[col].astype(bool)

    # Concatenate the dummy variables with the original dataframe
    df = pd.concat([df, dummy_df], axis=1)

    if drop:
        df.drop(columns=[col_name], axis=1, inplace=True)

    return df


def fill_column_pattern(df: pd.DataFrame, col_pattern: str) -> pd.DataFrame:
    cols = [c for c in df.columns if col_pattern in c]
    for i in range(cols):
        i = i + 1
        col_name = f'{col_pattern}_{i}'
        if not any([c for c in cols if col_name in c.lower()]):
            df[f'{col_name}'] = 0
    return df
