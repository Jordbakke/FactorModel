import torch
import json
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, prices_csv, fundamentals_csv, company_descriptions_embeddings_df, transactions_csv, fundamentals_max_sequence_length=100, prices_max_sequence_length=100):
        prices_df = pd.read_csv(prices_csv).sort_values("price_date")
        columns_to_drop = ["index_fsym_id", "index_price_date"]
        prices_df = prices_df.drop(columns=columns_to_drop)
        prices_df = prices_df.sort_values(["price_date"])
        self.prices_df = self.standardize_across_col(prices_df, 'fsym_id')

        fundamentals_df = pd.read_csv(fundamentals_csv).sort_values('date')
        columns_to_drop = ['currency', 'ff_upd_type', 'ff_fp_ind_code', 'ff_report_freq_code', 
                   'ff_fiscal_date', 'ff_fy_length_days', 'ff_fyr', 'ff_fpnc']
        fundamentals_df = fundamentals_df.drop(columns=columns_to_drop)
        fundamentals_df = fundamentals_df.sort_values(["date"])
        self.fundamental_df = self.standardize_across_entities_and_time(fundamentals_df)

        company_descriptions_embeddings_df = pd.read_csv(company_descriptions_embeddings_df, sep='|')
        company_descriptions_embeddings_df["embedding"] = company_descriptions_embeddings_df["embedding"].apply(lambda x: torch.tensor(json.loads(x)).unsqueeze(0))
        self.company_descriptions_embeddings_df = company_descriptions_embeddings_df

        transactions_df = pd.read_csv(transactions_csv)
        transactions_df['previous_transactions'] = transactions_df['previous_transactions'].apply(
        lambda x: sorted(json.loads(x), key=lambda d: d['report_date'])
        )
        transactions_df["portfolio_last_reporting_date"] = transactions_df["portfolio_last_reporting_date"].apply(lambda x: json.loads(x))
        self.transactions_df = transactions_df

        self.fundamentals_max_sequence_length = fundamentals_max_sequence_length
        self.prices_max_sequence_length = prices_max_sequence_length

    @staticmethod
    def standardize_across_entities_and_time(df):

        # Select numeric columns only
        numeric_cols = df.select_dtypes(include='number').columns
        
        # Apply the standardization for each numeric column, grouped by the specified column
        df[numeric_cols] = df[numeric_cols].transform(lambda x: (x - x.mean()) / x.std())

        return df

    @staticmethod
    def standardize_across_col(df, group_col):
 
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include='number').columns
        
        # Apply the standardization for each numeric column, grouped by the specified column
        df[numeric_cols] = df.groupby(group_col)[numeric_cols].transform(lambda x: (x - x.mean()) / x.std())

        return df

    def padd_slice_sequence_and_add_missing_feature_mask(self, values: np.array, max_sequence_length):
        """
        Pads the sequence with zeros for missing time periods, then creates and concatenates a feature mask.
        
        Args:
        - values: np.array of shape (sequence_length, num_features), where missing values are represented by np.nan.
        - max_sequence_length: int, the maximum length of the sequence to pad to.
        
        Returns:
        - concatenated_data: np.array of shape (max_sequence_length, 2 * num_features), where the first num_features columns
        are the original values (with padding if necessary) and the next num_features columns are the feature mask.
        """
        sequence_length = len(values)
        
        # Create the feature mask (1 for non-NaN values, 0 for NaN or padded values)
        feature_mask = (~np.isnan(values)).astype(float)
        
        # If sequence is longer than max_sequence_length, slice it
        if sequence_length > max_sequence_length:
            values = values[-max_sequence_length:]
            feature_mask = feature_mask[-max_sequence_length:]
        else:
            # If sequence is shorter, pad the beginning with zeros
            padding_size = max_sequence_length - sequence_length
            values = np.concatenate([np.zeros((padding_size, values.shape[1])), values], axis=0)
            feature_mask = np.concatenate([np.zeros((padding_size, values.shape[1])), feature_mask], axis=0)
        
        # Concatenate the feature mask with the values
        concatenated_data = np.concatenate([values, feature_mask], axis=1)
        
        return concatenated_data

    def retrieve_fundamentals(self, company_list: list):
        fundamentals_list = []

        for company in company_list:
            fsym_id = company["fsym_id"]
            date = company["report_date"]

            fundamentals = self.fundamental_df[
                (self.fundamental_df["fsym_id"] == fsym_id) & 
                (self.fundamental_df["date"] <= date)
            ]

            # Append the numeric values after dropping unnecessary columns
            fundamentals_values = fundamentals.drop(columns=["fsym_id", "date"]).select_dtypes(include='number').values
            fundamentals_values =self.padd_slice_sequence_and_add_missing_feature_mask(fundamentals_values, self.fundamentals_max_sequence_length)
            fundamentals_list.append(fundamentals_values)
        # Convert the list of numpy arrays into a single numpy array
        fundamentals_array = np.stack(fundamentals_list)  # Stack along a new dimension (default axis 0)

        return torch.tensor(fundamentals_array)
    
    def retrieve_prices(self, company_list: list):

        prices_list = []
        max_sequence_length = 0
        for company in company_list:
            fsym_id = company["fsym_id"]
            date = company["report_date"]

            prices = self.prices_df[
                (self.prices_df["fsym_id"] == fsym_id) & 
                (self.prices_df["price_date"] <= date)
            ]
            
            max_sequence_length = max(max_sequence_length, len(prices))
            # Append the numeric values after dropping unnecessary columns
            values = prices.drop(columns=["fsym_id", "price_date"]).select_dtypes(include='number').values
            values = self.padd_slice_sequence_and_add_missing_feature_mask(values, self.prices_max_sequence_length)
            prices_list.append(values)

        # Convert the list of numpy arrays into a single numpy array
        prices_array = np.stack(prices_list)  # Stack along a new dimension (0)

        return torch.tensor(prices_array)

    def retrieve_company_description_embeddings(self, company_list: list): 
        """
        Takes in list of companies (previous transaction companies or last reported portfolio companies) and returns the company description embeddings. 
        """
        company_description_embeddings = []
        for company in company_list:
            ticker_quarter_year = company["ticker_quarter_year"]
            company_description_embedding = self.company_descriptions_embeddings_df[
                self.company_descriptions_embeddings_df["ticker_quarter_year"] == ticker_quarter_year
                ]["embedding"].values[0]
            company_description_embeddings.append(company_description_embedding)
        
        return torch.stack(company_description_embeddings, dim=0)

    def __len__(self):
        return len(self.transactions_df)
    
    def __getitem__(self, idx):
        transaction = self.transactions_df.iloc[idx]

        #previous transaction companies
        previous_transaction_companies_data = {}
        previous_transactions = transaction['previous_transactions']
        
        previous_transactions_company_description_embeddings_tensor = self.retrieve_company_description_embeddings(previous_transactions)
        previous_transaction_companies_data['company_description_embeddings'] = previous_transactions_company_description_embeddings_tensor

        previous_transactions_fundamentals_tensor = self.retrieve_fundamentals(previous_transactions)
        previous_transaction_companies_data['fundamentals'] = previous_transactions_fundamentals_tensor

        previous_transactions_prices_tensor = self.retrieve_prices(previous_transactions)
        previous_transaction_companies_data['prices'] = previous_transactions_prices_tensor

        #last reported portfolio companies
        portfolio_companies_data = {}
        portfolio_companies = transaction['portfolio_last_reporting_date']

        portfolio_companies_description_embeddings = self.retrieve_company_description_embeddings(portfolio_companies)
        portfolio_companies_data['company_description_embeddings'] = portfolio_companies_description_embeddings

        portfolio_companies_fundamentals = self.retrieve_fundamentals(portfolio_companies)
        portfolio_companies_data['fundamentals'] = portfolio_companies_fundamentals

        portfolio_companies_prices = self.retrieve_prices(portfolio_companies)
        portfolio_companies_data['prices'] = portfolio_companies_prices


        return previous_transaction_companies_data, portfolio_companies_data


if __name__ == "__main__":
    custom_dataset = CustomDataset(prices_csv=r"C:\repos\Deep-learning-trj\data\monthly_prices_example\example_prices.csv",
                                fundamentals_csv=r"C:\repos\Deep-learning-trj\data\fundamentals_example\example_fundamentals.csv",
                                company_descriptions_embeddings_df=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv",
                                transactions_csv=r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")


    previous_transaction_companies_data, portfolio_companies_data = custom_dataset[2]

    for key, value in previous_transaction_companies_data.items():
        print(key, value.shape)

    for key, value in portfolio_companies_data.items():
        print(key, value.shape)
