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

def get_cols_to_standardize(df, cols_to_standardize: list):
    return df.select_dtypes(include='number').columns.tolist() if cols_to_standardize is None else cols_to_standardize

# Column-wise standardization function
def standardize_columns(df, cols_to_standardize=None, inplace=False):
    cols = get_cols_to_standardize(df, cols_to_standardize)
    scaler = StandardScaler()
    if not inplace:
        standardized_df = df.copy()
        standardized_df[cols] = scaler.fit_transform(standardized_df[cols])
        return standardized_df
    else:
        df[cols] = scaler.fit_transform(df[cols])
        return df

# Group-wise standardization function
def standardize_grouped_columns(df, group_columns, cols_to_standardize=None, inplace=False):
    return df.groupby(group_columns, group_keys=False)[df.columns].apply(lambda group: standardize_columns(group, cols_to_standardize, inplace=inplace)).reset_index(drop=True)

def rolling_standardize_within_window(series, window):
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

def standardize_rolling_window(df, window_size, cols_to_standardize=None, inplace=False):

    """
    Standardizes each column in a pandas DataFrame using a rolling window strategy.

    Parameters:
    - df (pd.DataFrame): The input DataFrame with numeric columns to standardize.
    - window_size (int): The size of the rolling window.

    Returns:
    - pd.DataFrame: A DataFrame with standardized columns based on the rolling window.
    """

    cols_to_standardize = get_cols_to_standardize(df, cols_to_standardize)
    # Apply rolling standardization to each column
    if not inplace:
        standardized_df = df.copy()
        standardized_df[cols_to_standardize] = standardized_df[cols_to_standardize].apply(lambda col: rolling_standardize_within_window(col, window_size))
        return standardized_df
    else:
        df[cols_to_standardize] = df[cols_to_standardize].apply(lambda col: rolling_standardize_within_window(col, window_size))
        return df

def standardize_time_windows(df, period_col, period_freq, num_periods, cols_to_standardize=None, include_current_period=False, min_periods=1):
    """
    Standardizes values within a time window, such as past 3 years, where the number of entities per year varies.
    """
    if period_freq not in ["D", "M", "Q", "Y"]:
        raise ValueError("Invalid period frequency. Supported frequencies are 'D', 'M', 'Q', and 'Y'.")
    
    if not pd.api.types.is_period_dtype(df[period_col]):
        df[period_col] = pd.to_datetime(df[period_col]).dt.to_period(period_freq)
        nat_mask = df[period_col].isna()

        # Display rows with NaT
        rows_with_nat = df[nat_mask]
        print(rows_with_nat)
    
    cols_to_standardize = get_cols_to_standardize(df, cols_to_standardize)
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
       
        if len(time_window) < min_periods or current_data.empty or subset.empty: 
            current_data[cols_to_standardize] = np.nan
            results.append(current_data)
            continue

        # Calculate rolling mean and std
        subset_means = subset[cols_to_standardize].mean()
        subset_sds = subset[cols_to_standardize].std(ddof=0).replace(0, np.nan)
        
        current_data[cols_to_standardize] = (current_data[cols_to_standardize] - subset_means) / subset_sds

        results.append(current_data)
    
    if not results:
        return pd.DataFrame(columns=df.columns)

    return pd.concat(results, ignore_index=True)

def log_ratio_transform(df, cols_to_transform: list, shift_value=1, inplace=False, new_column_names: list = None):
    
    """
        Log-transforms the ratios of the specified columns and updates column names.

        Parameters:
        - df (pd.DataFrame): The input DataFrame with numeric columns.
        - cols_to_transform (list): The columns to transform.
        - shift_value (int): The number of periods to shift when calculating the ratio.
        - inplace (bool): Whether to modify the DataFrame in place.
        - new_column_suffix (str): Suffix to append to the transformed column names.

        Returns:
        - pd.DataFrame: The transformed DataFrame.
    """
    cols_to_transform = get_cols_to_standardize(df, cols_to_transform)
    
    # Prepare new column names
    if new_column_names is None:
        new_column_names = [f"{col}_log_ratio" for col in cols_to_transform]
    elif len(new_column_names) != len(cols_to_transform):
        raise ValueError("Number of new column names must match the number of columns to transform.")
    
    if not inplace:
        transformed_df = df.copy()
        transformed_df[new_column_names] = np.log(transformed_df[cols_to_transform] / transformed_df[cols_to_transform].shift(shift_value))
        return transformed_df
    else:
        df[new_column_names] = np.log(df[cols_to_transform] / df[cols_to_transform].shift(shift_value))
        return df

def one_hot_encode_categorical_columns(df, categorical_columns: list):
    # Only include categorical columns that exist in the DataFrame
    categorical_columns = [column for column in categorical_columns if column in df.columns]
    
    # Perform one-hot encoding on the specified categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dummy_na=True)
    
    # Convert one-hot encoded columns to float (1.0 and 0.0) if needed
    df_encoded[df_encoded.columns.difference(df.columns)] = df_encoded[df_encoded.columns.difference(df.columns)].astype(float)
    
    return df_encoded

def add_missing_feature_mask(tensor: torch.Tensor):
    # Create a mask where non-NaN values are 0 and NaNs are 1
    if len(tensor.shape) != 3:
        raise ValueError("Tensor must be 3D")
    
    feature_mask = torch.isnan(tensor)
    feature_mask = feature_mask.float()

    return torch.cat([tensor, feature_mask], dim=2)

def pad_or_slice_and_create_key_padding_mask(tensor: torch.tensor, target_sequence_length: int, pad_value=0):
    
    if len(tensor.shape) != 3:
        raise ValueError("Input tensor must be 3D")
    
    sequence_length = tensor.shape[1]
    batch_size = tensor.shape[0]

    if sequence_length < target_sequence_length:
        # Pad with zeros along the sequence dimension if sequence_length < target_sequence_length
        padding_size = target_sequence_length - sequence_length
        adjusted_tensor = torch.nn.functional.pad(tensor, (pad_value, pad_value, pad_value, padding_size))
    elif sequence_length > target_sequence_length:
        # Slice the tensor along the sequence dimension if sequence_length > target_sequence_length
        adjusted_tensor = tensor[:, -target_sequence_length:, :]
    else:
        # No adjustment needed if sequence_length == target_sequence_length
        adjusted_tensor = tensor

    # Create the attention mask with correct batch size
    key_padding_mask = torch.ones((batch_size, target_sequence_length), dtype=torch.bool)
    key_padding_mask[:, :min(sequence_length, target_sequence_length)] = 0  # Real tokens marked as 0, padding as 1

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

def get_year_quarters_between(start_period: str, end_period: str) -> list:
    return get_periods_between(start_period, end_period, freq="Q")

def get_year_months_between(start_period: str, end_period: str) -> list:
    return get_periods_between(start_period, end_period, freq='M')

def get_dates_between(start_date, end_date) -> list:
    return get_periods_between(start_date, end_date, freq='D')

def add_missing_periods(df: pd.DataFrame, period_column: str, freq: str,
                        entity_columns: list=None) -> pd.DataFrame:

    """
    Fills missing periods in a DataFrame based on a period column and frequency, ensuring
    all specified entities have data for each period in the range.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a time period column and optional entity columns.
    period_column : str
        Column containing period data, to be checked/converted to period dtype.
    freq : str
        Frequency for the periods (e.g., 'M' for monthly).
    entity_columns : list of str, optional
        List of columns representing entities.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing periods (and entities) filled in as NaN rows.
    """
        
    if period_column not in df.columns:
        raise ValueError(f"Period column '{period_column}' is not a column in the DataFrame.")
    
    # Ensure period_column is of period dtype or convert it
    if not pd.api.types.is_period_dtype(df[period_column]):
        try:
            df[period_column] = pd.to_datetime(df[period_column]).dt.to_period(freq)
        except Exception as e:
            raise TypeError(f"Period column '{period_column}' could not be converted to pd.Period.") from e

    # Calculate the range of missing periods
    min_period = df[period_column].min()
    max_period = df[period_column].max()
    missing_periods = get_periods_between(min_period, max_period, include_start_period=True, include_end_period=True, freq=freq)

    missing_periods_df = pd.DataFrame({period_column: missing_periods})
    
    if entity_columns:
        entity_combinations = df[entity_columns].drop_duplicates()
        missing_periods_df = missing_periods_df.merge(entity_combinations, how='cross')
    
    # Merge to include missing periods and entities
    result = pd.merge(missing_periods_df, df, on=[period_column] + (entity_columns or []), how="outer", suffixes=('_missing', '')).reset_index(drop=True)

    return result

def expand_time_series(df:pd.DataFrame, period_col: str, new_period_col_name:str,
                       from_period_freq:str="Q", to_period_freq:str="M") -> pd.DataFrame:

    """
    Function that expands a time series from a lower frequency to a higher frequency (including periods between first and last period).
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

def collect_values(df, value_column, date_column, new_column_name, sequence_length=None, include_current_row=False,
                   drop_empty_sequences=False, direction="previous"):
    # Ensure the date column is in datetime format
    if not (pd.api.types.is_datetime64_any_dtype(df[date_column]) or pd.api.types.is_period_dtype(df[date_column])):
        raise TypeError("date column is not of datetime or Period type")

    # Sort the DataFrame by date within each group
    df = df.sort_values(by=date_column)

    # Initialize a list to store collected sequences
    collected_sequences = []
    for i in range(len(df)):
        if direction == "previous":
            # Collect previous values up to the sequence length
            collected_values = df[value_column].iloc[max(0, i - sequence_length): i].tolist()
            if include_current_row:
                value_to_insert = df[value_column].iloc[i]
                try:
                    value_to_insert = float(value_to_insert)
                except:
                    pass
                collected_values.insert(0, value_to_insert)  # Convert to regular float
                if len(collected_values) > sequence_length:
                    collected_values = collected_values[:-1]
        elif direction == "future":
            # Collect future values up to the sequence length
            collected_values = df[value_column].iloc[i + 1: i + 1 + sequence_length].tolist()
            if include_current_row:
                value_to_insert = df[value_column].iloc[i]
                try:
                    value_to_insert = float(value_to_insert)
                except:
                    pass
                collected_values.insert(0, value_to_insert)  # Convert to regular float
                if len(collected_values) > sequence_length:
                    collected_values = collected_values[:-1]
        
        # Trim the sequence to the specified length
        if sequence_length is not None:
            collected_values = collected_values[-sequence_length:] if direction == "previous" else collected_values[:sequence_length]

        # Check for empty list
        if (drop_empty_sequences and len(collected_values) == 0):
            collected_sequences.append(None)  # Mark for later dropping
        else:
            collected_sequences.append(collected_values)

    # Add the collected sequences as a new column
    df[new_column_name] = collected_sequences

    # Drop rows with None in the new column if drop_sequences_with_na is True
    if drop_empty_sequences:
        df = df.dropna(subset=[new_column_name])

    return df

# Wrappers for previous and future value collection
def collect_previous_values(df, value_column, date_column, new_column_name, sequence_length=None, include_current_row=False, drop_empty_sequences=False):
    return collect_values(df, value_column, date_column, new_column_name, sequence_length, include_current_row, drop_empty_sequences, direction="previous")

def collect_future_values(df, value_column, date_column, new_column_name, sequence_length=None, include_current_row=False, drop_empty_sequences=False):
    return collect_values(df, value_column, date_column, new_column_name, sequence_length, include_current_row, drop_empty_sequences, direction="future")

def convert_iterable_to_tensor(iterable, target_sequence_length=None, sequence_pad_value=0,
                               dtype=torch.float32):

    """
    Function that converts each row's sequence column into a tensor and pads or slices it to the specified sequence length.
    """
    if target_sequence_length is None:
        tensors = [torch.tensor(sequence, dtype=dtype) for sequence in iterable]
        return torch.stack(tensors, dim=0), None
    else:
        tensors_and_masks = [pad_or_slice_and_create_key_padding_mask(torch.tensor(sequence, dtype=dtype).unsqueeze(0), target_sequence_length, pad_value=sequence_pad_value) for sequence in iterable]
        tensors = torch.stack([tensor_mask[0] for tensor_mask in tensors_and_masks], dim=0)
        key_padding_masks = torch.stack([tensor_mask[1] for tensor_mask in tensors_and_masks], dim=0)
        return tensors, key_padding_masks

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

def store_tensor(tensor, tensorstore_path):
 
    if not isinstance(tensor, np.ndarray):
    # Convert the PyTorch tensor to a NumPy array
        tensor = tensor.detach().cpu().numpy()
    
    # Ensure the directory exists
    os.makedirs(tensorstore_path, exist_ok=True)
    
    # Define the storage specification for the tensor with the correct dtype
    spec = {
    'driver': 'zarr','kvstore': {'driver': 'file','path': tensorstore_path},
    'metadata': {'dtype': '<f4', 'shape': tensor.shape}
    }
   
    # Delete the existing TensorStore directory if it exists to clear cached metadata
    if os.path.exists(tensorstore_path):
        shutil.rmtree(tensorstore_path)
    
    # Re-create and write the TensorStore data
    tensorstore = ts.open(spec, create=True, open=True).result()
    tensorstore[tensorstore.domain] = tensor
    print(f"Tensor successfully overwritten at {tensorstore_path}")

def create_tensorstore_mapping(tensor_df_units: list, id_cols: list,
                               date_column: str = None, mapping_df_path: str = None,
                               mapping_df_storage_path: str = None, ffill=False, ffill_limit: int = None) -> pd.DataFrame:
    
    if mapping_df_path:
        mapping_df = pd.read_parquet(mapping_df_path)
    else:
        mapping_df = None

    tensor_df_reference_names = []

    for tensor_df_unit in tensor_df_units:
        # Extract DataFrame and column info
        tensor_df = tensor_df_unit.get("tensor_df")
        tensor_df_reference_name = tensor_df_unit.get("reference_name")
        tensor_column = tensor_df_unit.get("tensor_column")
        
        # Validate that required keys are present
        if tensor_df is None or tensor_df_reference_name is None or tensor_column is None:
            raise ValueError("Each tensor_df_unit must contain 'tensor_df', 'reference_name', and 'tensor_column' keys.")

        tensor_df_reference_names.append(tensor_df_reference_name)

        if set(id_cols).difference(set(tensor_df.columns)):
            raise ValueError("id_cols must be a subset of tensor_df columns")

        tensor_df = tensor_df.dropna(subset=[tensor_column, *id_cols])
        tensor_df["row_number"] = range(len(tensor_df))
        tensor_df = tensor_df.rename(columns={"row_number": tensor_df_reference_name})
        tensor_df = tensor_df[id_cols + [tensor_df_reference_name]]

        if mapping_df is None:
            mapping_df = tensor_df[id_cols]
        
        mapping_df = mapping_df.merge(tensor_df, on=id_cols, how="outer")

    if ffill and ffill_limit and date_column:
        mapping_df = add_missing_periods(mapping_df, period_column=date_column, freq="M", insert_entity=True, entity_columns=id_cols)
        group_cols = [id_col for id_col in id_cols if id_col != date_column]
        mapping_df = mapping_df.groupby(group_cols, group_keys=False)[mapping_df.columns].apply(lambda group: group.sort_values(by=date_column, ascending=True).ffill(limit=ffill_limit)).reset_index(drop=True)

    if mapping_df_storage_path:
        mapping_df.to_parquet(mapping_df_storage_path, index=False)
    
    return mapping_df


def from_tensor_df_to_mapped_tensorstore(tensor_df: pd.DataFrame, tensor_df_reference_name: str,
                                         tensor_column: str, id_cols: list, date_column: str = None,
                                         mapping_df: pd.DataFrame = None, mapping_df_storage_path: str = None,
                                         target_sequence_length=None, pad_value=0, tensorstore_path: str = None):
    
    mapping_df = create_tensorstore_mapping(tensor_df, tensor_df_reference_name, tensor_column,
                                            id_cols, date_column, mapping_df, mapping_df_storage_path)
    
    tensor_to_store, key_padding_mask = convert_iterable_to_tensor(tensor_df[tensor_column].tolist(),
                                                 target_sequence_length=target_sequence_length, pad_value=pad_value)
    
    store_tensor(tensor_to_store, tensorstore_path)

    if key_padding_mask is not None:
        store_tensor(key_padding_mask, tensorstore_path + "_mask")

    mapping_df.to_parquet(mapping_df_storage_path, index=False)

    return mapping_df

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

def open_in_excel(df: pd.DataFrame):
    # Create a temporary file with .xlsx extension
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        temp_file_path = tmp.name
        
    # Save DataFrame to the temporary file
    df.to_excel(temp_file_path, index=False)
    
    # Open the file in Excel
    webbrowser.open(f'file://{temp_file_path}')

if __name__ == "__main__":

    # df = pd.DataFrame({
    # "id": ["A", "B"],
    # "date": ["2020-01-01", "2021-03-01"],
    # "sequence": [
    #     [[10, 10, 10], [20, 20, 20]],
    #     [[80, 80, 80]],

    # ]
    # })
    # df["date"] = pd.to_datetime(df["date"])
    # print(df)
    # df1 =  pd.DataFrame({
    # "id": ["A", "C", "B"],
    # "date": ["2020-02-01", "2021-07-01", "2021-03-01"],
    # "sequence": [
    #     [[100, 100, 100], [200, 200, 200]],
    #     [[800, 800, 800]], [[900, 900, 900]] 

    # ]
    # })
    # df1["date"] = pd.to_datetime(df1["date"])
    # print(df1)

    # units = [{"tensor_df": df, "reference_name": "df1", "tensor_column": "sequence"},
    #          {"tensor_df": df1, "reference_name": "df2", "tensor_column": "sequence"}]
    # mapping = create_tensorstore_mapping(units, id_cols=["id", "date"], date_column="date", ffill=True, limit = 2)
    # print(mapping)

    # df = pd.DataFrame({"date": ["2020-01-01", "2020-03-01", "2020-08-01", "2020-03-01"], "id": ["A", "A", "B", "B"], "id1": [1, 1, 2, 3], "value": [10, 20, 30, 40]})
    # df["date"] = pd.to_datetime(df["date"])
    # df["date"] = df["date"].dt.to_period("M")
    # df = add_missing_periods(df, period_column="date", freq="M", entity_columns=["id", "id1"])
    # print(df.sort_values(by=["id", "id1", "date"]))
    
    corporate_debt = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data\US_corporate_debt_growth.csv")
    corporate_debt = corporate_debt.rename(columns={"BOGZ1FG104104005Q": "corporate_debt_growth", "DATE": "date"})
    corporate_debt = add_year_quarter_column(corporate_debt, date_column="date", year_quarter_column_name="year_quarter", drop_date_column=True)
    corporate_debt["release_month_debt"] = corporate_debt["year_quarter"].apply(lambda x: pd.Period(x.end_time, freq="M") + 2)
    corporate_debt["corporate_debt_growth"] = pd.to_numeric(corporate_debt["corporate_debt_growth"], errors="coerce")
    corporate_debt = corporate_debt.sort_values(by="year_quarter").ffill(limit=1)
    corporate_debt = corporate_debt[["release_month_debt", "corporate_debt_growth", "year_quarter"]]

    core_inflation = pd.read_csv(r"C:\repos\Deep-learning-trj\data\raw_data_storage\macro_data\US_core_inflation.csv")
    core_inflation = core_inflation.rename(columns={"CORESTICKM159SFRBATL": "core_inflation", "DATE": "date"})
    core_inflation = add_year_month_column(core_inflation, date_column="date", year_month_column_name="year_month", drop_date_column=True)
    core_inflation["core_inflation"] = pd.to_numeric(core_inflation["core_inflation"], errors="coerce")
    core_inflation = core_inflation.sort_values(by="year_month").ffill(limit=1)
    core_inflation["release_month_core_inflation"] = core_inflation["year_month"] + 1
    core_inflation = core_inflation[["year_month","release_month_core_inflation", "core_inflation"]]

    df = core_inflation.merge(corporate_debt, left_on="release_month_core_inflation", right_on="release_month_debt", how="outer").ffill(limit=3)
    open_in_excel(corporate_debt)
    open_in_excel(core_inflation)
    open_in_excel(df)