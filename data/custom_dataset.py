import torch
import json
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

CUTOFF_DATE = '2024-06-30'
class CustomDataset(Dataset):

    def __init__(self, prices_csv, fundamentals_csv, company_descriptions_embeddings_csv, fixed_company_features_csv, transactions_csv, target_transactions_period = 90, fundamentals_max_sequence_length=100, prices_max_sequence_length=100):
        prices_df = pd.read_csv(prices_csv).sort_values("price_date")
        prices_df = prices_df.sort_values(["price_date"])
        self.prices_df = self.standardize_across_col(prices_df, 'fsym_id')

        fundamentals_df = pd.read_csv(fundamentals_csv).sort_values('date')
        columns_to_drop = ['currency', 'ff_upd_type', 'ff_fp_ind_code', 'ff_report_freq_code', 
                   'ff_fiscal_date', 'ff_fy_length_days', 'ff_fyr', 'ff_fpnc']
        fundamentals_df = fundamentals_df.drop(columns=columns_to_drop)
        fundamentals_df = fundamentals_df.sort_values(["date"])
        self.fundamental_df = self.standardize_across_entities_and_time(fundamentals_df)

        company_descriptions_embeddings_df = pd.read_csv(company_descriptions_embeddings_csv, sep='|')
        company_descriptions_embeddings_df["embedding"] = company_descriptions_embeddings_df["embedding"].apply(lambda x: torch.tensor(json.loads(x)).unsqueeze(0))
        self.company_descriptions_embeddings_df = company_descriptions_embeddings_df

        fixed_company_features_df = pd.read_csv(fixed_company_features_csv)
        columns_to_drop = ["ticker_region", "factset_company_entity_id", "entity_proper_name",
                           "primary_sic_code","industry_code","sector_code"]

        fixed_company_features_df = fixed_company_features_df.drop(columns=columns_to_drop)
        fixed_company_features_df["year_founded"] = fixed_company_features_df["year_founded"].transform(lambda x: (x - x.mean()) / x.std()) 
        self.fixed_company_features_df = self.one_hot_encode_categorical_columns(fixed_company_features_df)

        transactions_df = pd.read_csv(transactions_csv)
        transactions_df['previous_transactions'] = transactions_df['previous_transactions'].apply(
        lambda x: sorted(json.loads(x), key=lambda d: d['report_date'])
        )
        transactions_df["portfolio_last_reporting_date"] = transactions_df["portfolio_last_reporting_date"].apply(lambda x: json.loads(x))
        transactions_df["company_info"] = transactions_df["company_info"].apply(lambda x: json.loads(x))
        transactions_df["target_transactions_" + str(target_transactions_period)] = transactions_df["target_transactions_" + str(target_transactions_period)].apply(
        lambda x: sorted(json.loads(x), key=lambda d: d['report_date'])
        )

        self.transactions_df = transactions_df[transactions_df["report_date"] <= CUTOFF_DATE]
        self.fundamentals_max_sequence_length = fundamentals_max_sequence_length
        self.prices_max_sequence_length = prices_max_sequence_length
        self.target_transactions_period = target_transactions_period

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
    
    @staticmethod
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

    def padd_slice_sequence(self, values: np.array, max_sequence_length: int) -> np.array:
        sequence_length = len(values)
        
        # If sequence is longer than max_sequence_length, slice it
        if sequence_length > max_sequence_length:
            padded_or_sliced_values = values[-max_sequence_length:]
        else:
            # If sequence is shorter, pad the beginning with np.nan
            padding_size = max_sequence_length - sequence_length
            padded_or_sliced_values = np.concatenate([np.full((padding_size, values.shape[1]), np.nan), values], axis=0)
        
        return padded_or_sliced_values
    
    def add_missing_feature_mask(self, values: np.array):
        feature_mask = (~np.isnan(values)).astype(float)
        return np.concatenate([values, feature_mask], axis=1)

    def retrieve_fundamentals(self, company_list: list):
        fundamentals_list = []

        for company in company_list:
            fsym_id = company["fsym_regional_id"]
            date = company["report_date"]

            fundamentals = self.fundamental_df[
                (self.fundamental_df["fsym_id"] == fsym_id) & 
                (self.fundamental_df["date"] <= date)
            ]

            # Append the numeric values after dropping unnecessary columns
            fundamentals_values = fundamentals.drop(columns=["fsym_id", "date"]).select_dtypes(include='number').values
            #pad/slice
            #mask
            #fillna
            fundamentals_values =self.padd_slice_sequence(fundamentals_values, self.fundamentals_max_sequence_length)
            fundamentals_values = self.add_missing_feature_mask(fundamentals_values)
            fundamentals_values = np.nan_to_num(fundamentals_values, nan=0.0)
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
            prices = prices.drop(columns=["fsym_id", "price_date"])
            
            values = prices.select_dtypes(include='number').values
            values = self.padd_slice_sequence(values, self.prices_max_sequence_length)
            values = self.add_missing_feature_mask(values)
            values = np.nan_to_num(values, nan=0.0)
            prices_list.append(values)

        # Convert the list of numpy arrays into a single numpy array
        prices_array = np.stack(prices_list)  # Stack along a new dimension (0)
        prices_tensor = torch.tensor(prices_array)
        
        return prices_tensor
    
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

    def retrieve_company_features(self, company_list: list):
        company_features = []
        for company in company_list:
            fsym_id = company["fsym_id"]
            values =self.fixed_company_features_df[self.fixed_company_features_df["fsym_id"] == fsym_id].drop(columns=["fsym_id"]).values
            company_features.append(values)
        
        # First stack the numpy arrays, then convert to a torch tensor
        stacked_features = np.stack(company_features)
        stacked_features = torch.tensor(stacked_features)
        return stacked_features
 
    def __len__(self):
        return len(self.transactions_df)
    
    def __getitem__(self, idx):
        transaction = self.transactions_df.iloc[idx]
        print(f"Num portfolio companies: {len(transaction["portfolio_last_reporting_date"])}")
        print(f"Num previous transactions: {len(transaction['previous_transactions'])}")
        print(f"Num target transactions: {len(transaction['target_transactions_' + str(self.target_transactions_period)])}")
              
        #previous transaction companies
        previous_transaction_companies_data = {}
        previous_transactions = transaction['previous_transactions']
        previous_transaction_companies_data['prices'] = self.retrieve_prices(previous_transactions).float()
        previous_transaction_companies_data['fundamentals'] = self.retrieve_fundamentals(previous_transactions).float()
        previous_transaction_companies_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(previous_transactions).float()
        previous_transaction_companies_data["fixed_company_features"] = self.retrieve_company_features(previous_transactions).float()
        #last reported portfolio companies

        portfolio_companies_data = {}
        portfolio_companies = transaction['portfolio_last_reporting_date']
        portfolio_companies_data['prices'] = self.retrieve_prices(portfolio_companies).float()
        portfolio_companies_data['fundamentals'] = self.retrieve_fundamentals(portfolio_companies).float()
        portfolio_companies_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(portfolio_companies).float()
        portfolio_companies_data["fixed_company_features"] = self.retrieve_company_features(portfolio_companies).float()

        #future transaction companies (target variables) including current transaction company
        target_transactions = transaction['target_transactions_' + str(self.target_transactions_period)]
        target_transactions_data = {}
        target_transactions_data['prices'] = self.retrieve_prices(target_transactions).float()
        target_transactions_data['fundamentals'] = self.retrieve_fundamentals(target_transactions).float()
        target_transactions_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(target_transactions).float()
        target_transactions_data["fixed_company_features"] = self.retrieve_company_features(target_transactions).float()

        return previous_transaction_companies_data, portfolio_companies_data, target_transactions_data

if __name__ == "__main__":
    custom_dataset = CustomDataset(prices_csv=r"C:\repos\Deep-learning-trj\data\monthly_prices\example_prices.csv",
                                fundamentals_csv=r"C:\repos\Deep-learning-trj\data\fundamentals\example_fundamentals.csv",
                                company_descriptions_embeddings_csv=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv",
                                fixed_company_features_csv=r"C:\repos\Deep-learning-trj\data\fixed_company_features\example_company_features.csv",
                                transactions_csv=r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")

    dataloader = DataLoader(custom_dataset, batch_size=None, shuffle=True)

    for i, batch in enumerate(dataloader):
        if i > 1:
            break
        print(f"Batch: {i}")
        prev_trans, portf, targets = batch
        print("target:")
        for key, tensor in targets.items():
            print(key)
            print(tensor.shape)
            print()
        print("prev_trans:")
        for key, tensor in prev_trans.items():
            print(key)
            print(tensor.shape)
            print()
        print("portf:")
        for key, tensor in portf.items():
            print(key)
            print(tensor.shape)
            print()

