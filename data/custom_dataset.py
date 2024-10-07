import torch
import json
import pandas as pd
import numpy as np
import data_utils
from torch import nn
from torch.utils.data import Dataset, DataLoader

CUTOFF_DATE = '2024-06-30'
class CustomDataset(Dataset):

    def __init__(self, prices_csv, fundamentals_csv, company_descriptions_embeddings_csv, fixed_company_features_csv, transactions_csv, target_transactions_period = 90, fundamentals_max_sequence_length=100, prices_max_sequence_length=100):
        super(CustomDataset, self).__init__()
        prices_df = pd.read_csv(prices_csv).sort_values("price_date")
        prices_df = prices_df.sort_values(["price_date"])
        self.prices_df = data_utils.standardize_across_col(prices_df, 'fsym_id')

        fundamentals_df = pd.read_csv(fundamentals_csv).sort_values('date')
        columns_to_drop = ['currency', 'ff_upd_type', 'ff_fp_ind_code', 'ff_report_freq_code', 
                   'ff_fiscal_date', 'ff_fy_length_days', 'ff_fyr', 'ff_fpnc']
        fundamentals_df = fundamentals_df.drop(columns=columns_to_drop)
        fundamentals_df = fundamentals_df.sort_values(["date"])
        self.fundamentals_df = data_utils.global_standardization(fundamentals_df)

        company_descriptions_embeddings_df = pd.read_csv(company_descriptions_embeddings_csv, sep='|')
        company_descriptions_embeddings_df["embedding"] = company_descriptions_embeddings_df["embedding"].apply(lambda x: torch.tensor(json.loads(x)).unsqueeze(0))
        self.company_descriptions_embeddings_df = company_descriptions_embeddings_df

        fixed_company_features_df = pd.read_csv(fixed_company_features_csv)
        columns_to_drop = ["ticker_region", "factset_company_entity_id", "entity_proper_name",
                           "primary_sic_code","industry_code","sector_code"]

        fixed_company_features_df = fixed_company_features_df.drop(columns=columns_to_drop)
        fixed_company_features_df["year_founded"] = fixed_company_features_df["year_founded"].transform(lambda x: (x - x.mean()) / x.std()) 
        self.fixed_company_features_df = data_utils.one_hot_encode_categorical_columns(fixed_company_features_df)

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
    
    def retrieve_fundamentals(self, company_list: list):
        fundamentals_list = []
        key_padding_mask_list = []
        max_sequence_length = 0

        for company in company_list:
            fsym_id = company["fsym_regional_id"]
            date = company["report_date"]

            fundamentals = self.fundamentals_df[
                (self.fundamentals_df["fsym_id"] == fsym_id) & 
                (self.fundamentals_df["date"] <= date)
            ]

            fundamentals_values = fundamentals.drop(columns=["fsym_id", "date"]).select_dtypes(include='number').values
            fundamentals_tensor = torch.tensor(fundamentals_values).float().unsqueeze(0)

            if fundamentals_tensor.shape[0] > max_sequence_length:
                max_sequence_length = fundamentals_tensor.shape[0]

            fundamentals_tensor = data_utils.add_missing_feature_mask(fundamentals_tensor)

            fundamentals_tensor, key_padding_mask = data_utils.pad_or_slice_and_create_key_padding_mask(fundamentals_tensor, self.fundamentals_max_sequence_length)
            fundamentals_tensor = torch.nan_to_num(fundamentals_tensor, nan=0.0)
            fundamentals_list.append(fundamentals_tensor)
            key_padding_mask_list.append(key_padding_mask)

        fundamentals = torch.stack(fundamentals_list).squeeze(1)
        key_padding_mask = torch.stack(key_padding_mask_list).squeeze(1)

        max_sequence_length = min(max_sequence_length, self.fundamentals_max_sequence_length)
        fundamentals = fundamentals[:, :max_sequence_length, :]
        key_padding_mask = key_padding_mask[:, :max_sequence_length]
        
        return fundamentals, key_padding_mask
    
    def retrieve_prices(self, company_list: list):

        prices_list = []
        key_padding_mask_list = []
        max_sequence_length = 0

        for company in company_list:
            fsym_id = company["fsym_id"]
            date = company["report_date"]

            prices = self.prices_df[
                (self.prices_df["fsym_id"] == fsym_id) & 
                (self.prices_df["price_date"] <= date)
            ]
            
            # Append the numeric values after dropping unnecessary columns
            prices = prices.drop(columns=["fsym_id", "price_date"])
            prices_values = prices.select_dtypes(include='number').values
            prices_tensor = torch.tensor(prices_values).float().unsqueeze(0)

            if prices.shape[0] > max_sequence_length:
                max_sequence_length = prices.shape[0]

            #prices_tensor = data_utils.add_missing_feature_mask(prices_tensor)
            prices_tensor, key_padding_mask = data_utils.pad_or_slice_and_create_key_padding_mask(prices_tensor, self.prices_max_sequence_length)
            prices_tensor = torch.nan_to_num(prices_tensor, nan=0.0)
            prices_list.append(prices_tensor)
            key_padding_mask_list.append(key_padding_mask)
        
        prices_tensor = torch.stack(prices_list).squeeze(1)
        key_padding_mask = torch.stack(key_padding_mask_list).squeeze(1)

        max_sequence_length = min(max_sequence_length, self.prices_max_sequence_length)
        prices_tensor = prices_tensor[:, :max_sequence_length, :]
        key_padding_mask = key_padding_mask[:, :max_sequence_length]
        
        return prices_tensor, key_padding_mask
    
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
        company_features_list = []
        for company in company_list:
            fsym_id = company["fsym_id"]
            values =self.fixed_company_features_df[self.fixed_company_features_df["fsym_id"] == fsym_id].drop(columns=["fsym_id"]).values
            company_features_tensor = torch.tensor(values).float().unsqueeze(0)
            company_features_list.append(company_features_tensor)
        
        # First stack the numpy arrays, then convert to a torch tensor
        company_features = torch.stack(company_features_list, dim=0).squeeze(1)
        return company_features
 
    def __len__(self):
        return len(self.transactions_df)
    
    def __getitem__(self, idx):

        """
        Returns a tuple of three dicts
        """
        transaction = self.transactions_df.iloc[idx]
        print(f"Num portfolio companies: {len(transaction["portfolio_last_reporting_date"])}")
        print(f"Num previous transactions: {len(transaction['previous_transactions'])}")
        print(f"Num target transactions: {len(transaction['target_transactions_' + str(self.target_transactions_period)])}")
        print("----------------------")
              
        #previous transaction companies
        previous_transaction_companies_data = {}
        previous_transactions = transaction['previous_transactions']
        previous_transaction_companies_data['prices'] = self.retrieve_prices(previous_transactions)
        previous_transaction_companies_data['fundamentals'] = self.retrieve_fundamentals(previous_transactions)
        previous_transaction_companies_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(previous_transactions).float()
        previous_transaction_companies_data["fixed_company_features"] = self.retrieve_company_features(previous_transactions).float()
        #last reported portfolio companies

        portfolio_companies_data = {}
        portfolio_companies = transaction['portfolio_last_reporting_date']
        portfolio_companies_data['prices'] = self.retrieve_prices(portfolio_companies)
        portfolio_companies_data['fundamentals'] = self.retrieve_fundamentals(portfolio_companies)
        portfolio_companies_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(portfolio_companies).float()
        portfolio_companies_data["fixed_company_features"] = self.retrieve_company_features(portfolio_companies).float()

        #future transaction companies (target variables) including current transaction company
        target_transactions = transaction['target_transactions_' + str(self.target_transactions_period)]
        target_transactions_data = {}
        target_transactions_data['prices'] = self.retrieve_prices(target_transactions)
        target_transactions_data['fundamentals'] = self.retrieve_fundamentals(target_transactions)
        target_transactions_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(target_transactions).float()
        target_transactions_data["fixed_company_features"] = self.retrieve_company_features(target_transactions).float()

        return previous_transaction_companies_data, portfolio_companies_data, target_transactions_data

    def collate_fn(self, batch):
        return batch

if __name__ == "__main__":
    custom_dataset = CustomDataset(prices_csv=r"C:\repos\Deep-learning-trj\data\monthly_prices\example_prices.csv",
                                fundamentals_csv=r"C:\repos\Deep-learning-trj\data\fundamentals\example_fundamentals.csv",
                                company_descriptions_embeddings_csv=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv",
                                fixed_company_features_csv=r"C:\repos\Deep-learning-trj\data\fixed_company_features\example_company_features.csv",
                                transactions_csv=r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")

    dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True, collate_fn=custom_dataset.collate_fn)
    
    print(len(dataloader)) # number of batches. A list of list
    for batch in dataloader: # batch is a list of tuples
        for ent in batch: #ent is a tuple of dicts
            print(type(batch))
            for d in ent:
                print(type(d))
        
