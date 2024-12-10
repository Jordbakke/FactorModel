import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import time
from data import data_utils
from utils import PositionalEncoding
from torch import nn

class EndToEndModel(nn.Module):
    def __init__(self, company_embedding_model,
                 prev_trans_portf_model,
                 next_transaction_model, max_companies_sequence_len=1000):
        super(EndToEndModel, self).__init__()
        self.company_embedding_model = company_embedding_model
        self.linear_projection = nn.Linear(company_embedding_model.output_dim + 1, #+1 for portfolio weights and buy/sale 
                                           prev_trans_portf_model.output_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim=company_embedding_model.output_dim + 1,
                                                      max_seq_len=max_companies_sequence_len)
        self.prev_trans_portf_model = prev_trans_portf_model
        self.next_transaction_model = next_transaction_model

    def forward(self, batch: list):
            start = time.time()
            """
            batch is a list of samples. Each sample is a tuple of three dictionaries
            containing the data for previous transactions, portfolio companies and target transactions
            """
            previous_transactions_batch = []
            previous_transaction_mask = []

            portfolio_companies_batch = []
            portfolio_companies_mask = []

            target_transactions_batch = []
            target_transactions_mask = []

            for previous_transaction_companies_data, portfolio_companies_data, target_transactions_data in batch:
                
                # PREVIOUS TRANSACTIONS
                previous_transactions_price_batch, key_padding_mask_price_batch = previous_transaction_companies_data['prices']
                assert torch.sum(torch.isnan(previous_transactions_price_batch)).item() == 0, "NaN values in previous transactions prices"

                previous_transactions_fundamentals_batch, key_padding_mask_fundamentals_batch = previous_transaction_companies_data['fundamentals']
                assert torch.sum(torch.isnan(previous_transactions_fundamentals_batch)).item() == 0, "NaN values in previous transactions fundamentals"

                previous_transactions_company_description_batch = previous_transaction_companies_data['company_description_embeddings']
                previous_transactions_fixed_company_features_batch = previous_transaction_companies_data['fixed_company_features']
                assert torch.sum(torch.isnan(previous_transactions_fixed_company_features_batch)).item() == 0, "NaN values in previous transactions fixed company features"

                previous_transactions_embeddings = self.company_embedding_model(previous_transactions_price_batch, previous_transactions_fundamentals_batch,
                                                                        previous_transactions_company_description_batch, previous_transactions_fixed_company_features_batch,
                                                                        price_layer_normalization=False, fundamentals_layer_normalization=False,
                                                                        key_padding_mask_price_batch=key_padding_mask_price_batch,
                                                                        key_padding_mask_fundamentals_batch=key_padding_mask_fundamentals_batch)
                
                # Shape is now (1, num_prev_transactions_companies, embedding_dim)
                previous_transactions_embeddings = previous_transactions_embeddings.permute(1, 0, 2)
                previous_transactions_embeddings =  torch.cat([previous_transactions_embeddings,previous_transaction_companies_data['buy_or_sale']], dim=-1)
                previous_transactions_embeddings = self.linear_projection(previous_transactions_embeddings)
                previous_transactions_embeddings = self.positional_encoding(previous_transactions_embeddings) # Add positional encoding to previous transactions (not needed for portfolio companies or target transactions)
                print(f"Previous transactions embeddings shape: {previous_transactions_embeddings.shape}")
                previous_transactions_batch.append(previous_transactions_embeddings)
                # PORTFOLIO COMPANIES
                portfolio_companies_price_batch, key_padding_mask_price_batch = portfolio_companies_data['prices']
                assert torch.sum(torch.isnan(portfolio_companies_price_batch)).item() == 0, "NaN values in portfolio companies prices"

                portfolio_companies_fundamentals_batch, key_padding_mask_fundamentals_batch  = portfolio_companies_data['fundamentals']
                assert torch.sum(torch.isnan(portfolio_companies_fundamentals_batch)).item() == 0, "NaN values in portfolio companies fundamentals"
                portfolio_companies_company_description_batch = portfolio_companies_data['company_description_embeddings']
                portfolio_companies_fixed_company_features_batch = portfolio_companies_data['fixed_company_features']
                assert torch.sum(torch.isnan(portfolio_companies_fixed_company_features_batch)).item() == 0, "NaN values in portfolio companies fixed company features"

                portfolio_companies_embeddings = self.company_embedding_model(portfolio_companies_price_batch, portfolio_companies_fundamentals_batch,
                                                                        portfolio_companies_company_description_batch, portfolio_companies_fixed_company_features_batch,
                                                                        price_layer_normalization=False, fundamentals_layer_normalization=False,
                                                                        key_padding_mask_price_batch=key_padding_mask_price_batch,
                                                                        key_padding_mask_fundamentals_batch=key_padding_mask_fundamentals_batch)
                
                portfolio_companies_embeddings = portfolio_companies_embeddings.permute(1, 0, 2) # Shape is now (1, num_portfolio_companies, embedding_dim)
                #add portfolio weights to the embeddings
                portfolio_companies_embeddings =  torch.cat([portfolio_companies_embeddings,portfolio_companies_data['portfolio_weights']], dim=-1)
                portfolio_companies_embeddings = self.linear_projection(portfolio_companies_embeddings)
                print(f"Portfolio companies embeddings shape: {portfolio_companies_embeddings.shape}")
                portfolio_companies_batch.append(portfolio_companies_embeddings)

                # TARGET TRANSACTIONS
                target_transactions_price_batch, key_padding_mask_price_batch = target_transactions_data['prices']
                assert torch.sum(torch.isnan(target_transactions_price_batch)).item() == 0, "NaN values in target transactions prices"

                target_transactions_fundamentals_batch, key_padding_mask_fundamentals_batch  = target_transactions_data['fundamentals']
                assert torch.sum(torch.isnan(target_transactions_fundamentals_batch)).item() == 0, "NaN values in target transactions fundamentals"

                target_transactions_company_description_batch = target_transactions_data['company_description_embeddings']
                target_transactions_fixed_company_features_batch = target_transactions_data['fixed_company_features']
                assert torch.sum(torch.isnan(target_transactions_fixed_company_features_batch)).item() == 0, "NaN values in target transactions fixed company features"
                
                target_transactions_embeddings = self.company_embedding_model(target_transactions_price_batch, target_transactions_fundamentals_batch,
                                                                            target_transactions_company_description_batch, target_transactions_fixed_company_features_batch,
                                                                            price_layer_normalization=False, fundamentals_layer_normalization=False,
                                                                            key_padding_mask_price_batch=key_padding_mask_price_batch,
                                                                            key_padding_mask_fundamentals_batch=key_padding_mask_fundamentals_batch)
                target_transactions_embeddings = target_transactions_embeddings.permute(1, 0, 2)
                # Shape is now (1, num_target_companies, embedding_dim)
                print(f"Target transactions embeddings shape: {target_transactions_embeddings.shape}")
                target_transactions_batch.append(target_transactions_embeddings)
                print("----------------------------")
            #Pad the sequences to the same length
            max_prev_transaction_seq_len = max([x.shape[1] for x in previous_transactions_batch])
            for i, prev_transactions_tensor in enumerate(previous_transactions_batch):
                padded_tensor, key_padding_mask = data_utils.pad_or_slice_and_create_key_padding_mask(prev_transactions_tensor,
                                                                                                         max_prev_transaction_seq_len)
                previous_transactions_batch[i] = padded_tensor
                previous_transaction_mask.append(key_padding_mask)

            max_portfolio_companies_seq_len = max([x.shape[1] for x in portfolio_companies_batch])
            for i, portfolio_companies_tensor in enumerate(portfolio_companies_batch):
                padded_tensor, key_padding_mask = data_utils.pad_or_slice_and_create_key_padding_mask(portfolio_companies_tensor,
                                                                                max_portfolio_companies_seq_len)
                portfolio_companies_batch[i] = padded_tensor
                portfolio_companies_mask.append(key_padding_mask)

            max_target_transactions_seq_len = max([x.shape[1] for x in target_transactions_batch])
            for i, target_transactions_tensor in enumerate(target_transactions_batch):
                padded_tensor, key_padding_mask = data_utils.pad_or_slice_and_create_key_padding_mask(target_transactions_tensor,
                                                                                max_target_transactions_seq_len)
                target_transactions_batch[i] = padded_tensor
                target_transactions_mask.append(key_padding_mask)

            previous_transactions_batch = torch.stack(previous_transactions_batch).squeeze(1)
            previous_transaction_mask = torch.stack(previous_transaction_mask).squeeze(1)
            print(f"Previous transactions batch shape: {previous_transactions_batch.shape}")
            print(f"Previous transactions mask shape: {previous_transaction_mask.shape}")
            portfolio_companies_batch = torch.stack(portfolio_companies_batch).squeeze(1)
            portfolio_companies_mask = torch.stack(portfolio_companies_mask).squeeze(1)
            print(f"Portfolio companies batch shape: {portfolio_companies_batch.shape}")
            print(f"Portfolio companies mask shape: {portfolio_companies_mask.shape}")
            target_transactions_batch = torch.stack(target_transactions_batch).squeeze(1)
            target_transactions_mask = torch.stack(target_transactions_mask).squeeze(1)
            print(f"Target transactions batch shape: {target_transactions_batch.shape}")
            print(f"Target transactions mask shape: {target_transactions_mask.shape}")

            assert previous_transactions_batch.shape[0] == portfolio_companies_batch.shape[0] == \
            target_transactions_batch.shape[0] == len(batch), "Batch size mismatch"
            
            # Interaction between previous transactions and portfolio companies
  
            prev_trans_portf_output = self.prev_trans_portf_model(previous_transactions_batch, portfolio_companies_batch,
                                                                  src_key_padding_mask=previous_transaction_mask, tgt_key_padding_mask=portfolio_companies_mask,
                                                                  memory_key_padding_mask=previous_transaction_mask)
            print(f"Prev_trans_portf output shape: {prev_trans_portf_output.shape}")
            # Use the output of the previous transactions and portfolio companies interaction to predict the next transaction
            predicted_transaction = self.next_transaction_model(prev_trans_portf_output)
            print(f"Predicted transaction shape: {predicted_transaction.shape}")
            # Create company embeddings the target companies
            print(f"Time taken forward: {time.time() - start}")
            return predicted_transaction, target_transactions_batch, target_transactions_mask
