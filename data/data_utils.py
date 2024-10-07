import torch
import pandas as pd

def global_standardization(df):
    # Select only numeric columns
    numeric_df = df.select_dtypes(include='number')
    
    # Calculate the mean and standard deviation across all numeric values
    overall_mean = numeric_df.values.mean()
    overall_std = numeric_df.values.std()
    
    # Standardize the numeric DataFrame
    standardized_numeric_df = (numeric_df - overall_mean) / overall_std
    
    # Combine the standardized numeric columns with the non-numeric columns
    non_numeric_df = df.select_dtypes(exclude='number')
    result_df = pd.concat([standardized_numeric_df, non_numeric_df], axis=1)
    
    # Ensure the original column order is preserved
    result_df = result_df[df.columns]
    
    return result_df

def standardize_across_entities_and_time(df):

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns
    
    # Apply the standardization for each numeric column, grouped by the specified column
    df[numeric_cols] = df[numeric_cols].transform(lambda x: (x - x.mean()) / x.std())

    return df

def standardize_across_col(df, group_col):

    # Select numeric columns only
    numeric_cols = df.select_dtypes(include='number').columns
    
    # Apply the standardization for each numeric column, grouped by the specified column
    df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(lambda x: (x - x.mean()) / x.std())

    return df

def one_hot_encode_categorical_columns(df, categorical_columns=["iso_country", "primary_sic_code",
                                                            "industry_code", "sector_code",
                                                            "listing_country"]):
    # Only include categorical columns that exist in the DataFrame
    categorical_columns = [column for column in categorical_columns if column in df.columns]
    non_categorical_columns = df.columns.difference(categorical_columns)
    
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=False, dummy_na=True)
    
    columns_to_convert = df_encoded.columns.difference(non_categorical_columns)
    df_encoded[columns_to_convert] = df_encoded[columns_to_convert].astype(float)  # Convert boolean to float (1.0 and 0.0)
    
    return df_encoded

def add_missing_feature_mask(values: torch.Tensor):
    # Create a mask where non-NaN values are 1 and NaNs are 0
    if len(values.shape) != 3:
        raise ValueError("Tensor must be 3D")
    
    feature_mask = ~torch.isnan(values)
    feature_mask = feature_mask.float()

    return torch.cat([values, feature_mask], dim=2)

def pad_or_slice_and_create_key_padding_mask(tensor, target_sequence_length):
    
    if len(tensor.shape) != 3:
        raise ValueError("Input tensor must be 3D")
    
    sequence_length = tensor.shape[1]
    batch_size = tensor.shape[0]

    if sequence_length < target_sequence_length:
        # Pad with zeros along the sequence dimension if sequence_length < target_sequence_length
        padding_size = target_sequence_length - sequence_length
        adjusted_tensor = torch.nn.functional.pad(tensor, (0, 0, 0, padding_size))
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

