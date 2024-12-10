import torch
import json
import pandas as pd
import numpy as np
import data_utils
import math
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import time

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
        start = time.time()
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
        print(f"Time taken to retrieve fundamentals: {time.time() - start}")
        return fundamentals, key_padding_mask

    def retrieve_prices(self, company_list: list):
        start = time.time()
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
        print(f"Time taken to retrieve prices: {time.time() - start}")
        return prices_tensor, key_padding_mask

    def retrieve_company_description_embeddings(self, company_list: list): 
        start = time.time()
        
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
        print(f"Time taken to retrieve company description embeddings: {time.time() - start}")
        return torch.stack(company_description_embeddings, dim=0)

    def retrieve_company_features(self, company_list: list):
        start = time.time()
        company_features_list = []
        for company in company_list:
            fsym_id = company["fsym_id"]
            values =self.fixed_company_features_df[self.fixed_company_features_df["fsym_id"] == fsym_id].drop(columns=["fsym_id"]).values
            company_features_tensor = torch.tensor(values).float().unsqueeze(0)
            company_features_list.append(company_features_tensor)

        # First stack the numpy arrays, then convert to a torch tensor
        company_features = torch.stack(company_features_list, dim=0).squeeze(1)
        company_features = torch.nan_to_num(company_features, nan=-1.0)
        print(f"Time taken to retrieve company features: {time.time() - start}")
        return company_features

    def retrieve_portfolio_weights(self, company_list: list):
        start = time.time()
        portfolio_weights_list = []
        for company in company_list:
            portfolio_weights = company["portfolio_weight"]
            portfolio_weights_list.append(portfolio_weights)
        print(f"Time taken to retrieve portfolio weights: {time.time() - start}")
        return torch.tensor(portfolio_weights_list).unsqueeze(1).unsqueeze(0).float()
    
    def retrieve_buy_sale_data(self, company_list: list):
        start = time.time()
        buy_sale_data_list = []
        for company in company_list:
            buy_sale_data = company["buy_or_sale"]
            buy_sale_data_list.append(buy_sale_data)
        print(f"Time taken to retrieve buy/sale data: {time.time() - start}")
        return torch.tensor(buy_sale_data_list).unsqueeze(1).unsqueeze(0)

    def __len__(self):
        return len(self.transactions_df)

    def __getitem__(self, idx):
        """
        Returns a tuple of three dicts
        """
        start = time.time()
        transaction = self.transactions_df.iloc[idx]
        # print(f"Num portfolio companies: {len(transaction["portfolio_last_reporting_date"])}")
        # print(f"Num previous transactions: {len(transaction['previous_transactions'])}")
        # print(f"Num target transactions: {len(transaction['target_transactions_' + str(self.target_transactions_period)])}")
        # print("----------------------")

        #previous transaction companies
        previous_transaction_companies_data = {}
        previous_transactions = transaction['previous_transactions']
        previous_transaction_companies_data['prices'] = self.retrieve_prices(previous_transactions)
        previous_transaction_companies_data['fundamentals'] = self.retrieve_fundamentals(previous_transactions)
        previous_transaction_companies_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(previous_transactions).float()
        previous_transaction_companies_data["fixed_company_features"] = self.retrieve_company_features(previous_transactions).float()
        previous_transaction_companies_data["buy_or_sale"] = self.retrieve_buy_sale_data(previous_transactions).float()
        #last reported portfolio companies

        portfolio_companies_data = {}
        portfolio_companies = transaction['portfolio_last_reporting_date']
        portfolio_companies_data['prices'] = self.retrieve_prices(portfolio_companies)
        portfolio_companies_data['fundamentals'] = self.retrieve_fundamentals(portfolio_companies)
        portfolio_companies_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(portfolio_companies).float()
        portfolio_companies_data["fixed_company_features"] = self.retrieve_company_features(portfolio_companies).float()
        portfolio_companies_data["portfolio_weights"] = self.retrieve_portfolio_weights(portfolio_companies).float()

        #future transaction companies (target variables) including current transaction company
        target_transactions = transaction['target_transactions_' + str(self.target_transactions_period)]
        target_transactions_data = {}
        target_transactions_data['prices'] = self.retrieve_prices(target_transactions)
        target_transactions_data['fundamentals'] = self.retrieve_fundamentals(target_transactions)
        target_transactions_data['company_description_embeddings'] = self.retrieve_company_description_embeddings(target_transactions).float()
        target_transactions_data["fixed_company_features"] = self.retrieve_company_features(target_transactions).float()
        print(f"Time taken to retrieve all data: {time.time() - start}")
        return previous_transaction_companies_data, portfolio_companies_data, target_transactions_data

    def collate_fn(self, batch):
        return batch

def create_dataloaders(custom_dataset, collate_fn, batch_size=32,
                        val_split=0.1, test_split=0.1,
                        shuffle=True, random_seed=42):
    # Get the total number of samples
    dataset_size = len(custom_dataset)
    # Generate indices for splitting the dataset
    indices = list(range(dataset_size))
    
    # Shuffle the indices if specified
    if shuffle:
        torch.manual_seed(random_seed)
        torch.random.manual_seed(random_seed)
        torch.manual_seed(random_seed)
        torch.randperm(len(indices))
        
    # Calculate the split sizes
    test_size = math.ceil(test_split * dataset_size) 
    val_size = math.ceil(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    # Split the indices into train, validation, and test sets
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=random_seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=random_seed)

    # Create data samplers for each set
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create data loaders for each set
    train_dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=train_sampler)
    val_dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=val_sampler)
    test_dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn, sampler=test_sampler)

    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    custom_dataset = CustomDataset(prices_csv=r"C:\repos\Deep-learning-trj\data\monthly_prices\example_prices.csv",
                                fundamentals_csv=r"C:\repos\Deep-learning-trj\data\fundamentals\example_fundamentals.csv",
                                company_descriptions_embeddings_csv=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv",
                                fixed_company_features_csv=r"C:\repos\Deep-learning-trj\data\fixed_company_features\example_company_features.csv",
                                transactions_csv=r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(custom_dataset, collate_fn=custom_dataset.collate_fn, batch_size=2,
                                                                   shuffle=True)
