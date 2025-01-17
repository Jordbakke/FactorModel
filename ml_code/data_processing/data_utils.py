import torch
import pandas as pd
import numpy as np
import tempfile
import webbrowser
import matplotlib.pyplot as plt
import tensorstore as ts
import os
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

def get_cols_to_transform(df, cols_to_transform: list):
    return df.select_dtypes(include='number').columns.tolist() if cols_to_transform is None else cols_to_transform

# Column-wise standardization function
def standardize_columns(df, cols_to_transform=None, inplace=False) -> pd.DataFrame:
    cols = get_cols_to_transform(df, cols_to_transform)
    scaler = StandardScaler()
    if not inplace:
        standardized_df = df.copy()
        standardized_df[cols] = scaler.fit_transform(standardized_df[cols])
        return standardized_df
    else:
        df[cols] = scaler.fit_transform(df[cols])
        return df

# Group-wise standardization function
def standardize_grouped_columns(df, group_columns, cols_to_transform=None, inplace=False):
    return df.groupby(group_columns, group_keys=False)[df.columns].apply(lambda group: standardize_columns(group, cols_to_transform, inplace=inplace)).reset_index(drop=True)

def rolling_standardize_within_window(series, window) -> pd.Series:
    """
    Standardizes values within each rolling window.

    Parameters:
    - series (pd.Series): The series to standardize.
    - window (int): The size of the rolling window.

    Returns:
    - pd.Series: The standardized values for each rolling window.
    """
    result = series.rolling(window=window, min_periods=window).apply(
        lambda x: (x - x.mean()).iloc[-1] / x.std(ddof=0), raw=False
    )

    return result

def standardize_rolling_window(df, window_size, cols_to_transform=None, inplace=False) -> pd.DataFrame:

    """
    Standardizes each column in a pandas DataFrame using a rolling window strategy.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with numeric columns to standardize.
    - window_size (int): The size of the rolling window.

    Returns:
    - pd.DataFrame: A DataFrame with standardized columns based on the rolling window.
    """
    cols_to_transform = get_cols_to_transform(df, cols_to_transform)
    # Apply rolling standardization to each column
    if not inplace:
        standardized_df = df.copy()
        standardized_df[cols_to_transform] = standardized_df[cols_to_transform].apply(lambda col: rolling_standardize_within_window(col, window_size))
        return standardized_df
    else:
        df[cols_to_transform] = df[cols_to_transform].apply(lambda col: rolling_standardize_within_window(col, window_size))
        return df

def standardize_time_windows(df, period_col, period_freq, num_periods, cols_to_transform=None, include_current_period=False, min_periods=1) -> pd.DataFrame:
    """
    Standardizes values within a time window, such as past 3 years, where the number of entities per year is more than 1.
    """
    if period_freq not in ["D", "M", "Q", "Y"]:
        raise ValueError("Invalid period frequency. Supported frequencies are 'D', 'M', 'Q', and 'Y'.")
    
    if not pd.api.types.is_period_dtype(df[period_col]):
        df[period_col] = pd.to_datetime(df[period_col]).dt.to_period(period_freq)
    
    cols_to_transform = get_cols_to_transform(df, cols_to_transform)
    results = []
    offset = 0 if include_current_period else -1  # Adjust offset for current period inclusion

    unique_periods = df[period_col].unique() 
    for period in unique_periods:
        end_period = period + offset
        start_period = max(min(unique_periods), end_period - num_periods + 1)


        # Define time window
        time_window = list(get_periods_between(
            start_period, end_period, freq=period_freq,
            include_start_period=True, include_end_period=True
        ))

        subset = df[df[period_col].isin(time_window)]
        current_data = df[df[period_col] == period].copy()

        if current_data.empty or len(subset[period_col].unique()) < min_periods:
            continue

        # Calculate rolling mean and std
        subset_means = subset[cols_to_transform].mean()
        subset_sds = subset[cols_to_transform].std(ddof=0).replace(0, np.nan)
        
        current_data[cols_to_transform] = (current_data[cols_to_transform] - subset_means) / subset_sds

        results.append(current_data)
    
    if not results:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(results, ignore_index=True)

def log_ratio_transform(df, cols_to_transform: list, shift_value=1, inplace=False, new_column_names: list = None,
                        drop_original_cols=False, invalid_replacement_value=0) -> pd.DataFrame:
    
    """
        Log-transforms the ratios of the specified columns and updates column names.

        Parameters:
        - df (pd.DataFrame): The input DataFrame with numeric columns.
        - cols_to_transform (list): The columns to transform.
        - shift_value (int): The number of periods to shift when calculating the ratio.
        - inplace (bool): Whether to modify the DataFrame in place.
        - new_column_suffix (str): Suffix to append to the transformed column names.
        - drop_first_row (bool): Whether to drop the first row after transformation.

        Returns:
        - pd.DataFrame: The transformed DataFrame.
    """
    cols_to_transform = get_cols_to_transform(df, cols_to_transform)
    
    # Prepare new column names
    if new_column_names is None:
        new_column_names = [f"{col}_log_ratio" for col in cols_to_transform]
    elif len(new_column_names) != len(cols_to_transform):
        raise ValueError("Number of new column names must match the number of columns to transform.")
    
    df[cols_to_transform] = df[cols_to_transform].apply(pd.to_numeric, errors='coerce').astype(float)

    if not inplace:
        transformed_df = df.copy()

        log_ratios = np.where(
        (transformed_df[cols_to_transform] == 0) | 
        (transformed_df[cols_to_transform].shift(shift_value) == 0),
        invalid_replacement_value,
        np.log(transformed_df[cols_to_transform] / transformed_df[cols_to_transform].shift(shift_value)))
        transformed_df[new_column_names] = log_ratios

        if drop_original_cols:
            transformed_df = transformed_df.drop(columns=cols_to_transform)
        return transformed_df
    else:
        log_ratios = np.where(
        (df[cols_to_transform] == 0) | 
        (df[cols_to_transform].shift(shift_value) == 0),
        invalid_replacement_value,
        np.log(df[cols_to_transform] / df[cols_to_transform].shift(shift_value)))
        df[new_column_names] = log_ratios

        if drop_original_cols:
            df = df.drop(columns=cols_to_transform)

        return df

def one_hot_encode_categorical_columns(df, categorical_columns: list) -> pd.DataFrame:
    # Only include categorical columns that exist in the DataFrame
    categorical_columns = [column for column in categorical_columns if column in df.columns]
    
    # Perform one-hot encoding on the specified categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dummy_na=True)
    
    # Convert one-hot encoded columns to float (1.0 and 0.0) if needed
    df_encoded[df_encoded.columns.difference(df.columns)] = df_encoded[df_encoded.columns.difference(df.columns)].astype(float)
    
    return df_encoded

def pad_or_slice_and_create_key_padding_mask(tensor: torch.Tensor, target_sequence_length: int, pad_value=0, pre_pad=False):
    """
    Adjust the tensor to match the target sequence length by padding or slicing.
    Also creates a key padding mask to indicate which positions are padded.

    Args:
        tensor (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).
        target_sequence_length (int): Desired sequence length.
        pad_value (float, optional): Value used for padding. Default is 0.
        pre_pad (bool, optional): If True, pad at the beginning. Otherwise, pad at the end. Default is False.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Adjusted tensor and key padding mask.
    """
    if len(tensor.shape) != 3:
        raise ValueError("Input tensor must be 3D")

    sequence_length = tensor.shape[1]
    batch_size = tensor.shape[0]

    padding_size = max(0, target_sequence_length - sequence_length)

    if sequence_length < target_sequence_length:
        # Pad with pad_value along the sequence dimension if sequence_length < target_sequence_length
        if pre_pad:
            # Pad at the start (left)
            adjusted_tensor = torch.nn.functional.pad(
                tensor, (0, 0, padding_size, 0), value=pad_value
            )
        else:
            # Pad at the end (right)
            adjusted_tensor = torch.nn.functional.pad(
                tensor, (0, 0, 0, padding_size), value=pad_value
            )
    elif sequence_length > target_sequence_length:
        # Slice the tensor along the sequence dimension if sequence_length > target_sequence_length
        adjusted_tensor = tensor[:, -target_sequence_length:, :]
    else:
        # No adjustment needed if sequence_length == target_sequence_length
        adjusted_tensor = tensor

    # Create the attention mask
    key_padding_mask = torch.ones((batch_size, target_sequence_length), dtype=torch.bool)
    if pre_pad:
        # For pre-padding, real tokens are at the end of the sequence
        key_padding_mask[:, padding_size:] = 0
    else:
        # For post-padding, real tokens are at the start of the sequence
        key_padding_mask[:, :sequence_length] = 0

    return adjusted_tensor, key_padding_mask

def add_year_month_column(df: pd.DataFrame, date_column: str, year_month_column_name:str = None, drop_date_column=False) -> pd.DataFrame:
    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    year_month_column_name = year_month_column_name or "year_month"
    # Map dates to year-month periods
    df[year_month_column_name] = df[date_column].dt.to_period("M")

    if drop_date_column:
        df = df.drop(columns=[date_column])
    return df

def add_year_column(df: pd.DataFrame, date_column:str, year_column_name:str = None, drop_date_column=False) -> pd.DataFrame:
    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    year_column_name = year_column_name or "year"
    # Extract the year from the date column
    df[year_column_name] = df[date_column].dt.to_period("Y")

    if drop_date_column:
        df = df.drop(columns=[date_column])
    return df

def add_year_quarter_column(df: pd.DataFrame, date_column: str, year_quarter_column_name:str = None, drop_date_column=False) -> pd.DataFrame:
    # Ensure the date column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Map dates to quarters
    year_quarter_column_name = year_quarter_column_name or "year_quarter"
    df[year_quarter_column_name] = df[date_column].dt.to_period("Q")

    if drop_date_column:
        df = df.drop(columns=[date_column])
    return df

def get_periods_between(period1, period2, freq: str, include_start_period=True, include_end_period=True) -> list:

    # Convert period1 and period2 to Period objects with the specified frequency
    start = pd.Period(period1, freq=freq)
    end = pd.Period(period2, freq=freq)
    start = start if include_start_period else start + 1
    end = end if include_end_period else end - 1
    # Check if the frequency of the periods matches the desired frequency
    if pd.Period(period1).freqstr.split("-")[0] != freq or pd.Period(period2).freqstr.split("-")[0] != freq:
        raise ValueError(f"Periods do not match the specified frequency '{freq}'")
    
    # Generate the range of periods between start and end
    periods = pd.period_range(start=start, end=end, freq=freq)
    
    return periods

def get_missing_periods(periods, freq: str) -> list:
    
    unique_periods = pd.unique(periods)
    sorted_periods = unique_periods[np.argsort(unique_periods)]
    min_period = sorted_periods[0]
    max_period = sorted_periods[-1]

    all_periods = get_periods_between(min_period, max_period, freq=freq)
    missing_periods = all_periods[~all_periods.isin(sorted_periods)]

    return list(missing_periods)

def expand_time_series(df:pd.DataFrame, period_col: str, new_period_col_name:str,
                       from_period_freq:str="Q", to_period_freq:str="M") -> pd.DataFrame:

    """
    Function that expands a time series from a lower frequency to a higher frequency (including periods between first and last period).
    Values are filled with NaNs.
    """
    # Convert the period column to a period dtype
    if not pd.api.types.is_period_dtype(df[period_col]):
        df[period_col] = pd.to_datetime(df[period_col]).dt.to_period(from_period_freq)
    
    start_time = df[period_col].min().start_time
    end_time = df[period_col].max().end_time
    start_period = start_time.to_period(to_period_freq)
    end_period = end_time.to_period(to_period_freq)
    periods = get_periods_between(start_period, end_period, to_period_freq)

    period_df = pd.DataFrame({new_period_col_name: periods})
    period_df[period_col] = period_df[new_period_col_name].apply(lambda p: p.start_time.to_period(from_period_freq))
    
    df = df.merge(period_df, on=period_col, how="outer")
    
    return df

def remove_columns_with_nulls(df: pd.DataFrame, max_null_percentage: float) -> pd.DataFrame:
    """
    Keeps only columns where the percentage of null values is less than or equal to the threshold value.
    """
    # Calculate the threshold count based on the threshold value
    threshold_count = len(df) * max_null_percentage
    
    # Filter columns by checking if the count of nulls is less than or equal to the threshold count
    filtered_df = df.loc[:, df.isnull().sum() <= threshold_count]
    
    return filtered_df

def add_missing_timestamps( df: pd.DataFrame, period_column: str, freq: str, release_period_column= None, release_period_column_offset = 0, entity_columns: list = None, include_padding_mask=False,
                           padding_mask_col: str = None, pad_value=0, numeric_cols=None, include_missing_features_mask=False,
                           missing_features_mask_col: str = None) -> pd.DataFrame:
    
    """
    Adds missing timestamps to a DataFrame and optionally includes padding and missing feature masks for each timestamp. 
    """
    if period_column not in df.columns:
        raise ValueError(f"Period column '{period_column}' is not a column in the DataFrame.")
    
    if not pd.api.types.is_period_dtype(df[period_column]):
        try:
            df[period_column] = pd.to_datetime(df[period_column]).dt.to_period(freq)
        except Exception as e:
            raise TypeError(f"Period column '{period_column}' could not be converted to pd.Period.") from e

    missing_periods = get_missing_periods(df[period_column], freq)
    if missing_periods:
    
        if release_period_column is None:
            missing_periods_df = pd.DataFrame({period_column: missing_periods})
        else:
            missing_periods_df = pd.DataFrame({period_column: missing_periods, release_period_column: missing_periods + release_period_column_offset})

        df = pd.merge(
            missing_periods_df, df, on=[period_column] + ([release_period_column] if release_period_column else []),
            how="outer", suffixes=('_missing', '')
        ).reset_index(drop=True)

    if entity_columns:
        df[entity_columns] = df[entity_columns].ffill()

    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if include_padding_mask:
        if padding_mask_col is None:
            raise ValueError("`padding_mask_col` must be provided when `include_padding_mask` is True.")
        is_missing = df[period_column].isin(missing_periods)
        padding_mask = pd.Series(np.where(is_missing, 1, 0), name=padding_mask_col, index=df.index)
        df = df.drop(columns=[padding_mask_col], errors='ignore') #drop pre-existing padding mask column
        df = pd.concat([df, padding_mask], axis=1)

        # Fill numeric columns with pad_value for padded rows
        if pad_value is not None:
            if not isinstance(numeric_cols, list):
                raise ValueError("`numeric_cols` must be a list of column names.")
            df.loc[df[period_column].isin(missing_periods), numeric_cols] = pad_value

    # Handle missing feature masks
    if include_missing_features_mask:
        if missing_features_mask_col is None:
            raise ValueError("`missing_features_mask_col` must be provided when `include_missing_features_mask` is True.")
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if not numeric_cols:
            raise ValueError("`numeric_cols` must contain at least one column for missing features mask.")
        
        missing_features_mask = df[missing_features_mask_col].combine_first(
        pd.Series([[1] * len(numeric_cols)] * len(df), index=df.index))
        missing_features_mask = pd.Series(missing_features_mask.tolist(), name=missing_features_mask_col, index=df.index)
        df = df.drop(columns=[missing_features_mask_col], errors='ignore') #drop pre-existing missing features mask column
        df = pd.concat([df, missing_features_mask], axis=1)
    
    return df

def convert_df_to_sequences(df, value_columns: list, date_column, new_column_name, padding_mask_column: str = None,
                            missing_features_mask_column: str = None, sequence_length=None,include_current_row=False,
                   drop_empty_sequences=False, direction="previous", drop_value_columns=False) -> pd.DataFrame:

    # Ensure the date column is in datetime format
    if not (pd.api.types.is_datetime64_any_dtype(df[date_column]) or pd.api.types.is_period_dtype(df[date_column])):
        raise TypeError("date column is not of datetime or Period type")

    df.sort_values(by=date_column, inplace=True)
    values = np.vstack(df[value_columns])
    
    collected_sequences = []
    padding_mask = None
    missing_features_mask= None

    if padding_mask_column is not None:
        padding_mask_series = df[padding_mask_column].to_numpy().to_list()
        padding_masks = []
    
    if missing_features_mask_column is not None:
        missing_features_mask_series = np.vstack(df[missing_features_mask_column])
        missing_features_masks = []

    row_indices = np.arange(len(df))
    
    if direction == "previous":
        offset = 1 if include_current_row else 0
        end_indices = row_indices + offset
        start_indices = np.maximum(0, end_indices - sequence_length)
    elif direction == "future":
        offset = 0 if include_current_row else 1
        start_indices = row_indices + offset
        end_indices = np.minimum(len(df), start_indices + sequence_length)

    else:
        raise ValueError("Invalid direction. Must be 'previous' or 'future'.")

    for start, end in zip(start_indices, end_indices):
        seq_values = values[start:end, :]
        seq_values = seq_values if seq_values.any() else np.nan
        collected_sequences.append(seq_values)

        if padding_mask_column is not None:
            padding_mask = padding_mask_series[start:end]
            padding_masks.append(padding_mask)

        if missing_features_mask_column is not None:
            missing_features_mask = missing_features_mask_series[start:end]
            missing_features_masks.append(missing_features_mask)

    df[new_column_name] = collected_sequences

    if padding_mask_column is not None:
        df["padding_mask"] = padding_masks

    if missing_features_mask_column is not None:
        df["missing_features_mask"] = missing_features_masks


    # Optionally drop empty sequences
    if drop_empty_sequences:
        df = df.dropna(subset=[new_column_name])

    if drop_value_columns:
        df = df.drop(columns=value_columns)

    return df

def release_date_aware_processing_and_sequence_creation(df: pd.DataFrame, calendar_period_col: str, release_period_col: str,
                                                      new_column_name: str, sequence_length, cols_to_standardize=None,
                                                      log_transform_cols = None, log_transform_shift_value=1) -> pd.DataFrame:

    """
    Finds all available information prior or equal to the release date, aligns the date along the same timestamps and returns for that release date.
    Should be used when data for the same calendar timestamp have different release dates. Should only be used for relatively small dataframes. 
    """

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    cols = numeric_cols + [calendar_period_col]
    
    global_means = None
    global_stds = None
    missing_periods = None

    release_date_aware_dfs = []
    for i, release_period in enumerate(sorted(df[release_period_col].unique(), reverse=True)):
        available_data_df = df[df[release_period_col] <= release_period] #only get available information
        available_data_df = available_data_df.groupby(calendar_period_col, group_keys=False)[cols].apply(lambda group: group.ffill().bfill()) #align data on fiscal time
        available_data_df = available_data_df.drop_duplicates()
        available_data_df = available_data_df.sort_values(by=calendar_period_col)
        
        available_data_df["missing_features_mask"] = available_data_df[numeric_cols].isna().astype(float).to_numpy().tolist()
        available_data_df["padding_mask"] = 0
        available_data_df = available_data_df.ffill() #fill in missing features with previous known values

        if i == 0: # last release period
            if cols_to_standardize:
                global_means = available_data_df[numeric_cols].mean()
                global_stds = available_data_df[numeric_cols].std(ddof=0)

            missing_periods_global = get_missing_periods(available_data_df[calendar_period_col], freq="M")
            if missing_periods_global:
                missing_periods_df_global = pd.DataFrame({calendar_period_col: missing_periods_global, release_period_col: missing_periods, "padding_mask": [1] * len(missing_periods),
                                               "missing_feature_mask": [[1] * len(numeric_cols)] * len(missing_periods)})
        if cols_to_standardize:
            available_data_df[numeric_cols] = (available_data_df[numeric_cols] - global_means) / global_stds

        if missing_periods_global:
            missing_periods = missing_periods_df_global[missing_periods_df_global[calendar_period_col] <= available_data_df.iloc[-1][calendar_period_col]]

        if missing_periods:
            available_data_df = available_data_df.merge(missing_periods, on=calendar_period_col, how="outer")
            available_data_df["padding_mask"] = available_data_df["padding_mask_x"].combine_first(available_data_df["padding_mask_y"])
            available_data_df["padding_mask"] = available_data_df["padding_mask"].astype(int)
            available_data_df["missing_features_mask"] = available_data_df["missing_features_mask_x"].combine_first(available_data_df["missing_feature_mask_y"])
            available_data_df = available_data_df.drop(columns=["padding_mask_x", "padding_mask_y", "missing_features_mask_x", "missing_features_mask_y"])
            available_data_df.fillna(0, inplace=True) #padded rows should have 0 values
        
        if log_transform_cols:
            available_data_df = log_ratio_transform(available_data_df, log_transform_cols, shift_value=log_transform_shift_value, inplace=True)

        available_data_df["value_list"] = available_data_df[numeric_cols].to_numpy().tolist()

        value_list_sequence = available_data_df["value_list"].iloc[-sequence_length:].to_numpy().tolist()
        padding_mask = available_data_df["padding_mask"].iloc[-sequence_length:].to_numpy().tolist()
        missing_features_mask = available_data_df["missing_features_mask"].iloc[-sequence_length:].to_numpy().tolist()

        new_df = pd.DataFrame({new_column_name: [value_list_sequence], release_period_col: [release_period], "padding_mask": [padding_mask],
                               "missing_features_mask": [missing_features_mask]})
        
        release_date_aware_dfs.append(new_df)

    return pd.concat(release_date_aware_dfs, axis=0)

def iterable_is_2d(iterable):
    """
    Check if an iterable is 2D (i.e., a list of lists, a 2D NumPy array, or a 2D tensor).

    Args:
    - iterable: The iterable to check.

    Returns:
    - bool: True if the iterable is 2D, False otherwise.
    """
    if isinstance(iterable, list) and iterable and isinstance(iterable[0], list):
        return True
    elif isinstance(iterable, np.ndarray) and iterable.ndim == 2:
        return True
    elif isinstance(iterable, torch.Tensor) and iterable.dim() == 2:
        return True
    else:
        return False

def convert_iterable_to_tensor(iterable, target_sequence_length=None, sequence_pad_value=0,
                               pre_pad=False, tensor_dtype=torch.float32):
    
    """
    Function that converts iterable into a tensors and pads or slices it to the specified sequence length.
    Stacks the tensors into a single tensor.
    """
    if not iterable:
        raise ValueError("Input iterable is empty")
    
    tensors = []
    key_padding_masks = []

    for tensor in iterable:
        if not iterable_is_2d(tensor):
            raise ValueError("All tensors must be 2D")
        if target_sequence_length is None:
            tensors.append(torch.tensor(tensor, dtype=tensor_dtype))

        else:
            padded_tensor, key_padding_mask = pad_or_slice_and_create_key_padding_mask(torch.tensor(tensor, dtype=tensor_dtype).unsqueeze(0), target_sequence_length,
                                                                                        pad_value=sequence_pad_value, pre_pad=pre_pad)
            tensors.append(padded_tensor)
            key_padding_masks.append(key_padding_mask)

    stacked_tensor = torch.cat(tensors, 0)
    key_padding_mask = torch.cat(key_padding_masks, 0) if key_padding_masks else None

    return stacked_tensor, key_padding_mask

def load_tensor(tensorstore_path):

    """
    Loads a tensor from a specified TensorStore path.

    Parameters:
    tensorstore_path (str): The path to the TensorStore directory.

    Returns:
    torch.Tensor: The loaded tensor.
    """
    # Define the TensorStore specification
    spec = {'driver': 'zarr','kvstore': {'driver': 'file', 'path': tensorstore_path}}
    
    # Open the TensorStore
    tensorstore = ts.open(spec, open=True).result()
    
    # Read the data and convert it to a PyTorch tensor
    np_array = tensorstore.read().result()
    tensor = torch.from_numpy(np_array)
    
    return tensor

def store_tensor(tensor, tensorstore_path, dtype:str ='<f4'):
 
    if not isinstance(tensor, np.ndarray):
    # Convert the PyTorch tensor to a NumPy array
        tensor = tensor.detach().cpu().numpy()
    
    # Ensure the directory exists
    os.makedirs(tensorstore_path, exist_ok=True)
    
    # Define the storage specification for the tensor with the correct dtype
    spec = {
    'driver': 'zarr','kvstore': {'driver': 'file','path': tensorstore_path},
    'metadata': {'dtype': dtype, 'shape': tensor.shape}
    }
   
    # Delete the existing TensorStore directory if it exists to clear cached metadata
    if os.path.exists(tensorstore_path):
        shutil.rmtree(tensorstore_path)
    
    # Re-create and write the TensorStore data
    tensorstore = ts.open(spec, create=True, open=True).result()
    tensorstore[tensorstore.domain] = tensor
    print(f"Tensor successfully overwritten at {tensorstore_path}")

def create_tensor_mapping(mapping_df: pd.DataFrame, tensor_df: pd.DataFrame, tensor_df_reference_name: str, tensor_column: str,
                               left_on: list, right_on: list, how="left", apply_add_missing_timestamps: bool = False,
                               period_col: str = None, freq: str=None, entity_columns: list = None, ffill_limit: int=0,
                               mapping_df_storage_path: str = None) -> pd.DataFrame:
    
    """
    Useful when different target_data uses same input. Just referring to the index of the input dataframe rather than copying the input sequence many times is more memory-efficient.
    """

    if mapping_df is None:
        raise ValueError("Mapping DataFrame is None")
    
    # Validate that required keys are present
    if tensor_df is None or tensor_df_reference_name is None or tensor_column is None:
        raise ValueError(" One of 'tensor_df', 'reference_name', and 'tensor_column' is None.")

    missing_left_on_cols = set(left_on).difference(set(mapping_df.columns))
    if missing_left_on_cols:
        raise ValueError(f"Columns {missing_left_on_cols} are not present in mapping_df")

    missing_right_on_cols = set(right_on).difference(set(tensor_df.columns))

    if missing_right_on_cols:
        raise ValueError(f"Columns {missing_right_on_cols} are not present in tensor_df")
    
    tensor_df = tensor_df.dropna(subset=[tensor_column, *right_on], how="any")
    tensor_df["row_number"] = range(len(tensor_df))
    tensor_df = tensor_df.rename(columns={"row_number": tensor_df_reference_name})
    tensor_df = tensor_df[right_on + [tensor_df_reference_name]]
    
    if apply_add_missing_timestamps and period_col and freq: #useful when data is not available for all periods so that several different periods should refer to the same tensor. E.g data released every third month.
        tensor_df = add_missing_timestamps(tensor_df, period_col, freq, entity_columns=entity_columns)

    if ffill_limit != 0 and period_col:
        if not entity_columns:
            entity_columns = []
        tensor_df = tensor_df.sort_values(by=period_col)
        tensor_df = tensor_df.groupby(entity_columns, group_keys=False).apply(lambda group: group.ffill(limit=ffill_limit))

    mapping_df = mapping_df.merge(tensor_df, left_on=left_on, right_on=right_on, how=how)
    columns_to_drop = set(right_on).difference(set(left_on))
    mapping_df = mapping_df.drop(columns=columns_to_drop)
    mapping_df = mapping_df.dropna(how="any")
    mapping_df[tensor_df_reference_name] = mapping_df[tensor_df_reference_name].astype(int)

    if mapping_df_storage_path:
        mapping_df.to_parquet(mapping_df_storage_path, index=False)

    return mapping_df

def plot_entity_time_series(df: pd.DataFrame, entity_columns: list, time_column: str, value_column: str):
    """
    Function that groups by entity columns and plots the time series for each combination of entity columns
    in the same plot, with different colors for each combination.
    """
    # Ensure the time_column is in a format compatible with Matplotlib
    if df[time_column].dtype.name == 'period[M]':  # Example for monthly periods
        df[time_column] = df[time_column].dt.to_timestamp()

    # Group by entity columns
    grouped = df.groupby(entity_columns)

    # Create the plot with a larger figure size
    plt.figure(figsize=(20, 12))  # Increased figure size for better visibility

    # Plot each group with a unique color
    for entity_values, group in grouped:
        group = group.sort_values(by=time_column)  # Sort by time column
        entity_label = ", ".join([f"{col}: {val}" for col, val in zip(entity_columns, entity_values)])
        plt.scatter(group[time_column], group[value_column], label=entity_label)  # Changed to scatter plot

    # Customize the plot
    plt.title(f"Time Series for {', '.join(entity_columns)}", fontsize=16)
    plt.xlabel(time_column, fontsize=14)
    plt.ylabel(value_column, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_moving_average(df: pd.DataFrame, date_column: str, value_column: str, window: int):
    """
    Plots the moving average of a value column over time.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        date_column (str): The name of the column containing date values.
        value_column (str): The name of the column containing the values to average.
        window (int): The number of periods to calculate the moving average.

    Returns:
        None: Displays the plot.
    """
    # Convert the date column to datetime if it is a Period type
    if pd.api.types.is_period_dtype(df[date_column]):
        df[date_column] = df[date_column].dt.to_timestamp()

    # Ensure the DataFrame is sorted by date
    df = df.sort_values(by=date_column)

    # Calculate the moving average
    df['Moving_Average'] = df[value_column].rolling(window=window).mean()

    # Plot the original values and the moving average
    plt.figure(figsize=(12, 6))
    plt.plot(df[date_column], df[value_column], label='Original Values', marker="o", alpha=0.5)
    plt.plot(df[date_column], df['Moving_Average'], label=f'{window}-Period Moving Average', marker="o", linewidth=2)

    # Customize the plot
    plt.title(f'Moving Average ({window} periods) Over Time', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_tsne_2d(vectors_dict):

    """
    This function takes a dictionary where keys are labels and values are high-dimensional vectors,
    applies t-SNE to reduce the dimensionality to 2D, and plots the 2D representation.

    :param vectors_dict: Dictionary where keys are labels (strings) and values are vectors (arrays).
    """
    # Extract labels and vectors from the dictionary
    labels = list(vectors_dict.keys())
    vectors = np.array(list(vectors_dict.values()))

    # Dynamically set perplexity to be less than the number of vectors
    n_samples = len(vectors)
    perplexity = min(2, n_samples - 1)  # Ensure perplexity is less than n_samples

    # Use t-SNE to reduce the dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    vectors_2d = tsne.fit_transform(vectors)

    # Plot the 2D representation of the vectors
    plt.figure(figsize=(8, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], marker='o')

    # Annotate the points with their labels
    for i, label in enumerate(labels):
        plt.annotate(label, (vectors_2d[i, 0], vectors_2d[i, 1]))

    plt.title('2D visualization of high-dimensional vectors using t-SNE')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()

def merge_dir_files_into_one_df(dir_path: str, read_func=pd.read_csv, to_func="to_parquet", input_file_extension="csv", output_file_extension="parquet") -> pd.DataFrame:

    """
    Merges all files in a directory into a single DataFrame and writes it to a file.
    
    Args:
        dir_path (str): Path to the directory containing files.
        read_func (callable): Function to read individual files (default: pd.read_csv).
        to_func (str): Method name for writing the DataFrame (default: 'to_parquet').
        file_extension (str): Desired file extension for the output file (default: 'parquet').
    
    Returns:
        pd.DataFrame: Merged DataFrame.
    """

    # Get all files in the directory with the correct extension
    files = [f for f in os.listdir(dir_path) if f.endswith(input_file_extension)]
    if not files:
        raise ValueError(f"No files with extension '{input_file_extension}' found in directory: {dir_path}")
    
    # Read all files into a single DataFrame
    df = pd.concat([read_func(os.path.join(dir_path, file)) for file in files], ignore_index=True)

    # Write the DataFrame to the desired file format
    output_file = os.path.join(dir_path, f"merged_data.{output_file_extension}")
    getattr(df, to_func)(output_file)
    return df

def open_in_excel(df: pd.DataFrame):
    # Create a temporary file with .xlsx extension
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        temp_file_path = tmp.name
        
    # Save DataFrame to the temporary file
    df.to_excel(temp_file_path, index=False)
    
    # Open the file in Excel
    webbrowser.open(f'file://{temp_file_path}')

def open_in_excel(df: pd.DataFrame, file_name: str = None):
    """
    Save a DataFrame to an Excel file and open it in Excel.

    Parameters:
        df (pd.DataFrame): The DataFrame to save and open.
        file_name (str, optional): Desired file name for the Excel file (should include .xlsx). Default is None.
    """
    if file_name:
        # Ensure the file name has the correct extension
        if not file_name.endswith(".xlsx"):
            file_name += ".xlsx"
        
        # Save the file in the temporary directory with the given name
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, file_name)
    else:
        # Create a temporary file with a .xlsx extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
            temp_file_path = tmp.name

    # Save DataFrame to the specified file
    df.to_excel(temp_file_path, index=False)
    
    # Open the file in Excel
    webbrowser.open(f'file://{temp_file_path}')

if __name__ == "__main__":

    pass