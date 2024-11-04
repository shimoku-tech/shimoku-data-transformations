from typing import List, Callable, Optional, Dict, Union
import json

import pandas as pd
import numpy as np


def fix_broken_dates_to_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def custom_parse(date_string):
        if date_string == "1970-01-01":
            return "1970-01-01 00:00:00"
        return date_string

    df[col] = pd.to_datetime(df[col].apply(custom_parse), format="%Y-%m-%d %H:%M:%S")
    return df


def set_column_projection(
    df: pd.DataFrame,
    target_col_name: str,
    projection_col_name: str = '_last',
) -> pd.DataFrame:
    """"""
    df[target_col_name] = (
        df[target_col_name]
        .fillna(df[f'{target_col_name}{projection_col_name}'])
    )
    return df


def shift_date_years(
    df: pd.DataFrame,
    date_name_col: str,
    years: int = 1,
    shift_date_col_name: str = 'shifted_date',
) -> pd.DataFrame:
    """"""
    # create the date to merge last and current year
    df[shift_date_col_name] = (
        df[date_name_col] + pd.DateOffset(years=years)
    )
    return df


def get_all_columns_that_match_patterns(
    cols: List[str],
    patterns: List[str],
) -> List[str]:
    """"""
    cols_with_pattern = list()
    for pattern in patterns:
        cols_with_pattern += [col for col in cols if pattern in col]
    return cols_with_pattern


def replace_inf_by_nan(df: pd.DataFrame) -> List[str]:
    """"""
    return df.replace([np.inf, -np.inf], np.nan)


def replace_non_zero_values(
    df: pd.DataFrame,
    new_value: float,
    zero_value: float = 0,
) -> pd.DataFrame:
    """
    https://stackoverflow.com/questions/24021909/pandas-replace-non-zero-values
    """
    df[df != zero_value] = new_value
    return df


def find_different_rows(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Given two dataframes return the different rows"""
    return df1[~df1.isin(df2)].dropna()


def count_active_columns(df: pd.DataFrame) -> List[str]:
    """Given a dataset of numerical columns.
    Count how many are different from 0"""
    df = replace_non_zero_values(df=df, new_value=1)
    return df.sum(axis=1)


def fetch_columns_with_nan(df: pd.DataFrame) -> List[str]:
    """ Find any column that has a NaN value
    Biblio: https://stackoverflow.com/questions/36226083/how-to-find-which-columns-contain-any-nan-value-in-pandas-dataframe
    """
    return df.columns[df.isna().any()].tolist()


def find_next_date(df: pd.DataFrame, date_name: str) -> pd.Timestamp:
    """"""
    max_date = df[date_name].max()
    next_date = max_date + pd.Timedelta(days=1)
    return next_date


def find_all_dates_inbetween(
    start_date: pd.Timestamp, end_date: pd.Timestamp,
) -> pd.DatetimeIndex:
    """"""
    return pd.date_range(
        start_date, end_date - pd.Timedelta(days=1), freq='d'
    )


def attach_new_row(
    df: pd.DataFrame, row: pd.DataFrame,
    sort_values_by: str,
) -> pd.DataFrame:
    """Used in times-series forecasting"""
    return pd.concat([df, row]).sort_values(by=sort_values_by)


def join_multilevel_col_name(df: pd.DataFrame) -> pd.DataFrame:
    """
    Refference:
    https://stackoverflow.com/questions/19078325/naming-returned-columns-in-pandas-aggregate-function

    We do not use exactly the code of the refference to avoid
    to add underscore to columns without multi-level such as:

    `date` -> `date_`
    """
    cols_levels: List[str] = list()
    for x in df.columns.ravel():
        s: str = str()
        for element in x:
            if element:
                if s:
                    s += f'_{element}'
                else:
                    s += element
        cols_levels.append(s)
    df.columns = cols_levels

    return df


def soft_assert_series_equal(s1: pd.Series, s2: pd.Series):
    """A softer version of pd.testing.assert_series_equal"""
    assert np.array_equal(s1.values, s2.values)
    assert all(s1.index == s2.index)


def get_col_names_by_pattern(
    patterns: Dict[str, List[str]],
    output_type: str,
) -> Callable:
    """Check whether all elements from a pattern are in a string

    Example
    ------------------
    input
        patterns = {'is_dictionary': ['a', 'b']
        df.cols = ['abc', 'hujemetupexi']

    output
        {'is_dictionary': 'abc'}

    Refferences
    --------------------
    https://thispointer.com/python-check-if-a-list-contains-all-the-elements-of-another-list/
    """
    def _get_col_names_by_pattern(
        df: pd.DataFrame
    ) -> Union[Dict[str, str], str, List[str]]:
        """"""
        cols: List[str] = df.columns

        d: Dict[str, str] = {
            pattern_name: col
            for pattern_name, pattern in patterns
            for col in cols
            if all(
                single_pattern in col
                for single_pattern in pattern
            )
        }

        if output_type == 'str':
            return list(d.values())[0]
        elif output_type == 'list':
            return list(d.values())
        elif output_type == 'dict':
            return d
        else:
            raise ValueError(
                f'output_type value must be either "str", '
                f'"list" or "dict". Provided: {output_type}'
            )
    return _get_col_names_by_pattern
