import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

def filter_fundamentals(fundamentals_csv, null_pct_limit=0.1, null_replacement_value=0):

    fundamentals = pd.read_csv(fundamentals_csv)
    for column in fundamentals.columns:
        null_pct = fundamentals[column].isnull().sum() / len(fundamentals)
        if null_pct > null_pct_limit:
            fundamentals = fundamentals.drop(column, axis=1)
    
    fundamentals = fundamentals.fillna(null_replacement_value)

    fundamentals.to_csv(fundamentals_csv, index=False)

    return fundamentals

def replace_null_values(csv_file, value=0):
    df = pd.read_csv(csv_file)
    df = df.fillna(value)
    df.to_csv(csv_file, index=False)
    return df

def retrieve_fundamentals(ticker, date):
    pass

def retrieve_prices(ticker, date):
    pass

def retrieve_company_description(ticker, date, company_description_id):
    pass

def drop_first_transaction(transactions_csv, overwrite=False):
    "Drop the first transaction since we don't have any history on that investor"
    transactions = pd.read_csv(transactions_csv)
    transactions = transactions.dropna(subset=['previous_transactions', 'portfolio_last_reporting_date'], how='all')
    if overwrite:
        transactions.to_csv(transactions_csv, index=False)
    return transactions

class CustomDataset(Dataset):

    def __init__(self, prices_csv, fundamentals_csv, company_description_csv, transactions_csv):
        self.prices = pd.read_csv(prices_csv)
        self.fundamentals = pd.read_csv(fundamentals_csv)
        self.company_description = pd.read_csv(company_description_csv)
        self.transactions = pd.read_csv(transactions_csv)
        self.transactions = drop_first_transaction(self.transactions)

    def __len__(self):
        return len(self.transactions)
    
    def __getitem__(self, idx):
        transaction = self.transactions.iloc[idx]
        
        for previous_transaction in transaction['previous_transactions']:
            report_date = previous_transaction['report_date']
            ticker_quarter_year = previous_transaction["ticker_quarter_year"]
        
        company_description_id = transaction['company_description_id']
        price = retrieve_prices(ticker, date)
        fundamentals = retrieve_fundamentals(ticker, date)
        company_description = retrieve_company_description(ticker, date)
        return price, fundamentals, company_description


a = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
t = torch.tensor(a)
print(t.size())