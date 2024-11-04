import pandas as pd
import numpy as np
from scipy import stats


def label_numeric_feature(
        df: pd.DataFrame, column_name: str, num_categories=3, labels=None, zero_label=None) -> pd.Series:
    """
    Label a numeric feature based on its distribution, with an option to assign a specific label to zero values.

    This function categorizes numeric data into bins and assigns labels to each bin. It can handle
    zero values separately if desired.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be labeled.
    column_name (str): The name of the column in the DataFrame to be labeled.
    num_categories (int, optional): Number of categories to create (default is 3).
    labels (list, optional): List of labels to use for the categories. If None, default labels will be used
                             (default is None, which will use ['Low', 'Medium', 'High'] for 3 categories).
    zero_label (str, optional): Specific label to assign to zero values. If provided, this enables
                                special handling of zero values (default is None).

    Returns:
    pd.Series: A new series with labels assigned to each value in the original column.

    Behavior:
    1. If zero_label is None:
       - The function creates bins based on the entire range of data (including zeros).
       - All values (including zeros) are labeled according to these bins.

    2. If zero_label is provided (not None):
       - Zero values are first separated from the rest of the data.
       - Bins are created based only on non-zero values.
       - Non-zero values are labeled according to these bins.
       - All zero values are assigned the label specified by zero_label.
       - The function then combines the zero and non-zero labels, maintaining the original data order.

    This approach allows for special treatment of zero values, which can be useful in scenarios where
    zero has a distinct meaning separate from other low values (e.g., 'No purchase' vs. 'Low value purchase').

    The binning process uses quantile-based bins by default, falling back to equal-width bins if
    there aren't enough unique values to create the desired number of categories.

    Example:
    cols_to_label: List[str] = [
        'Unique_Products_Count', 'Total_Quantity', 'Avg_Product_Price',
        'Product_Price_Range', 'Purchase_Diversity',
        'avg_purchases_per_month', 'avg_revenue_per_month', 'avg_products_per_purchase',
    ]
    for col in cols_to_label:
        df_summary[f'{col}_label'] = label_numeric_feature(
            df=df_summary.copy(),
            column_name=col,
            num_categories=3,
            labels=['Low', 'Medium', 'High'],
            zero_label='No Activity'
        )

    In this example, any zero values in the specified columns will be labeled as 'No Activity',
    while all other values will be categorized as 'Low', 'Medium', or 'High'.

    Note:
    - The function will raise a ValueError if there aren't enough unique values to create
      the specified number of categories.
    - NaN values are dropped before processing and will result in NaN labels in the output.
    - If zero_label is provided, ensure that it's different from the labels used for non-zero values
      to avoid confusion in the resulting categorization.
    """

    # Check if the column exists
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame")

    # Get the data
    data = df[column_name]

    # Remove NaN values
    data_clean = data.dropna()

    if zero_label is not None:
        # Separate zero values
        zero_mask = data_clean == 0
        zero_values = data_clean[zero_mask]
        non_zero_values = data_clean[~zero_mask]

        # Check if we have enough unique non-zero values
        if len(non_zero_values.unique()) < num_categories:
            raise ValueError(f"Not enough unique non-zero values in '{column_name}' to create {num_categories} categories")

        # Create quantile-based bins for non-zero values
        bins = stats.mstats.mquantiles(non_zero_values, prob=np.linspace(0, 1, num_categories + 1))
    else:
        # Check if we have enough unique values
        if len(data_clean.unique()) < num_categories:
            raise ValueError(f"Not enough unique values in '{column_name}' to create {num_categories} categories")

        # Create quantile-based bins
        bins = stats.mstats.mquantiles(data_clean, prob=np.linspace(0, 1, num_categories + 1))

    # Ensure unique bin edges
    bins = np.unique(bins)

    # If we don't have enough unique bins, fall back to equal-width bins
    if len(bins) < num_categories + 1:
        if zero_label is not None:
            bins = np.linspace(non_zero_values.min(), non_zero_values.max(), num_categories + 1)
        else:
            bins = np.linspace(data_clean.min(), data_clean.max(), num_categories + 1)

    bins = np.where(np.isnan(bins), np.inf, bins)

    # Create labels if not provided
    if labels is None:
        if num_categories == 3:
            labels = ['Low', 'Medium', 'High']
        else:
            labels = [f'category_{i + 1}' for i in range(num_categories)]

    # Assign labels
    if zero_label is not None:
        non_zero_labels = pd.cut(non_zero_values, bins=bins, labels=labels, include_lowest=True)
        zero_labels = pd.Series(zero_label, index=zero_values.index)
        labeled_data = pd.concat([zero_labels, non_zero_labels]).reindex(data.index)
    else:
        labeled_data = pd.cut(data, bins=bins, labels=labels, include_lowest=True)

    return labeled_data


def label_binary_feature(df: pd.DataFrame, column_name: str, threshold: float = 0.8) -> tuple:
    """
    Label a feature based on its most common value, with two categories:
    'value_{common_value}' and 'value_different_to_{common_value}'.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be labeled.
    column_name (str): The name of the column in the DataFrame to be labeled.
    threshold (float): The threshold for the proportion of the most common value (default is 0.8).

    Returns:
    tuple: A tuple containing:
        - pd.Series: A new series with labels assigned to each value in the original column.
        - str: The common value used for labeling.
    """
    value_counts = df[column_name].value_counts(normalize=True)
    most_common_value = value_counts.index[0]

    labels = df[column_name].apply(
        lambda x: f'value_{most_common_value}' if x == most_common_value
        else f'value_different_to_{most_common_value}'
    )

    return labels, str(most_common_value)


def choose_labeling_method(df: pd.DataFrame, column_name: str, threshold: float = 0.8) -> str:
    """
    Decide which labeling method to use based on the distribution of values in the column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be analyzed.
    column_name (str): The name of the column in the DataFrame to be analyzed.
    threshold (float): The threshold for the proportion of the most common value (default is 0.8).

    Returns:
    str: 'binary' if a single value occurs more than the threshold, 'numeric' otherwise.
    """
    value_counts = df[column_name].value_counts(normalize=True)
    if value_counts.iloc[0] > threshold:
        return 'binary'
    else:
        return 'numeric'


def smart_label_feature(df: pd.DataFrame, column_name: str, num_categories: int = 3,
                        labels: list = None, zero_label: str = None, threshold: float = 0.8) -> pd.DataFrame:
    """
    Choose the appropriate labeling method and apply it to the given column.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data to be labeled.
    column_name (str): The name of the column in the DataFrame to be labeled.
    num_categories (int): Number of categories for numeric labeling (default is 3).
    labels (list): List of labels for numeric labeling (default is None).
    zero_label (str): Specific label for zero values in numeric labeling (default is None).
    threshold (float): The threshold for choosing between binary and numeric labeling (default is 0.8).

    Returns:
    pd.DataFrame: The input DataFrame with a new column added for the labels.
    """
    labeling_method = choose_labeling_method(df, column_name, threshold)

    if labeling_method == 'binary':
        binary_labels, common_value = label_binary_feature(df, column_name, threshold)
        df[f'{column_name}_value_{common_value}_label'] = binary_labels
    else:
        numeric_labels = label_numeric_feature(df, column_name, num_categories, labels, zero_label)
        df[f'{column_name}_label'] = numeric_labels

    return df


def get_top_quantile(df: pd.DataFrame, quantile=.1) -> pd.DataFrame:
    df = df.sort_values(by='prediction', ascending=False).head(int(len(df)*quantile))
    df['prediction'] = round(100 * df['prediction'])
# TODO llevate esto de aqui
    df.rename(columns={'customer_id': 'ID', 'prediction': 'Prediction'}, inplace=True)
    return df


def calculate_group_perc(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    val = df[cols].sum(axis=1)
    for col in cols:
        df[f'{col}_perc'] = round(100 * df[col] / val, 2)
    return df
