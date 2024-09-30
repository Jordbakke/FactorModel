import os
import sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.custom_dataset import CustomDataset

def train(company_embedding_model, transaction_prediction_model, loss_fn, custom_dataset, num_epochs, lr, weight_decay, device, save_path, batch_size=None):

    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(list(company_embedding_model.parameters()) + list(transaction_prediction_model.parameters()), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        
        for previous_transaction_companies_data, portfolio_companies_data, target_companies_data in tqdm(dataloader):
            previous_transactions_price_batch = previous_transaction_companies_data['price']
            previous_transactions_fundamentals_batch = previous_transaction_companies_data['fundamentals']
            previous_transactions_company_description_batch = previous_transaction_companies_data['company_description']
            previous_transactions_fixed_company_features_batch = previous_transaction_companies_data['fixed_company_features']
            previous_transactions_embeddings = company_embedding_model(previous_transactions_price_batch, previous_transactions_fundamentals_batch,
                                                                      previous_transactions_company_description_batch,
                                                                      previous_transactions_fixed_company_features_batch)
            
            portfolio_companies_price_batch = portfolio_companies_data['price']
            portfolio_companies_fundamentals_batch = portfolio_companies_data['fundamentals']
            portfolio_companies_company_description_batch = portfolio_companies_data['company_description']
            portfolio_companies_fixed_company_features_batch = portfolio_companies_data['fixed_company_features']
            portfolio_companies_embeddings = company_embedding_model(portfolio_companies_price_batch, portfolio_companies_fundamentals_batch,
                                                                    portfolio_companies_company_description_batch,
                                                                    portfolio_companies_fixed_company_features_batch)
            
            target_transactions_price_batch = target_companies_data['price']
            target_transactions_fundamentals_batch = target_companies_data['fundamentals']
            target_transactions_company_description_batch = target_companies_data['company_description']
            target_transactions_fixed_company_features_batch = target_companies_data['fixed_company_features']
            target_transactions_embeddings = company_embedding_model(target_transactions_price_batch,
                                                                              target_transactions_fundamentals_batch,
                                                                        target_transactions_company_description_batch,
                                                                        target_transactions_fixed_company_features_batch)
            
            predicted_transaction = transaction_prediction_model(previous_transactions_embeddings, portfolio_companies_embeddings)
            
            future_transaction_companies_embeddings = future_transaction_companies_embeddings.squeeze(1)

            loss = loss_fn(predicted_transaction, future_transaction_companies_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            



      