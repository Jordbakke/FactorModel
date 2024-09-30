import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from company_embedding_model import CompanyEmbeddingModel
from prev_trans_portf_model import PreviousTransactionsPortfolioModel
from next_transactions_model import NextTransactionsModel
from price_model import PriceModel
from fundamentals_model import FundamentalsModel
from company_description_model import CompanyDescriptionModel
from utils import HeadCombinationLayer
from utils import PositionalEncoding
from torch import nn

class EndToEndModel(nn.Module):
    def __init__(self, company_embedding_model, company_embedding_model_output_dim, prev_trans_portf_model, next_transaction_model, max_companies_sequence_len=1000):
        super(EndToEndModel, self).__init__()
        self.company_embedding_model = company_embedding_model
        self.positional_encoding = PositionalEncoding(embedding_dim=company_embedding_model_output_dim, max_seq_len=max_companies_sequence_len)
        self.prev_trans_portf_model = prev_trans_portf_model
        self.next_transaction_model = next_transaction_model

    def forward(self, previous_transaction_companies_data, portfolio_companies_data, target_transactions_data):

            # Create company embeddings the input companies (prev transactions and portfolio companies)
            previous_transactions_price_batch = previous_transaction_companies_data['prices']
            previous_transactions_fundamentals_batch = previous_transaction_companies_data['fundamentals']
            previous_transactions_company_description_batch = previous_transaction_companies_data['company_description_embeddings']
            previous_transactions_fixed_company_features_batch = previous_transaction_companies_data['fixed_company_features']
            previous_transactions_embeddings = self.company_embedding_model(previous_transactions_price_batch, previous_transactions_fundamentals_batch,
                                                                      previous_transactions_company_description_batch,
                                                                      previous_transactions_fixed_company_features_batch)
            previous_transactions_embeddings = self.positional_encoding(previous_transactions_embeddings) # Add positional encoding to previous transactions (not needed for portfolio companies or target transactions)
            print(f"Previous transactions embeddings shape: {previous_transactions_embeddings.shape}")

            portfolio_companies_price_batch = portfolio_companies_data['prices']
            portfolio_companies_fundamentals_batch = portfolio_companies_data['fundamentals']
            portfolio_companies_company_description_batch = portfolio_companies_data['company_description_embeddings']
            portfolio_companies_fixed_company_features_batch = portfolio_companies_data['fixed_company_features']
            portfolio_companies_embeddings = self.company_embedding_model(portfolio_companies_price_batch, portfolio_companies_fundamentals_batch,
                                                                    portfolio_companies_company_description_batch,
                                                                    portfolio_companies_fixed_company_features_batch)
            print(f"Portfolio companies embeddings shape: {portfolio_companies_embeddings.shape}")

            # Interaction between previous transactions and portfolio companies
            prev_trans_portf_output = self.prev_trans_portf_model(previous_transactions_embeddings, portfolio_companies_embeddings, is_causal=False)
            print(f"Prev_trans_portf output shape: {prev_trans_portf_output.shape}")
            # Use the output of the previous transactions and portfolio companies interaction to predict the next transaction
            predicted_transaction = self.next_transaction_model(target_transactions_data)
            print(f"Predicted transaction shape: {predicted_transaction.shape}")
            # Create company embeddings the target companies

            target_transactions_price_batch = target_transactions_data['prices']
            target_transactions_fundamentals_batch = target_transactions_data['fundamentals']
            target_transactions_company_description_batch = target_transactions_data['company_description_embeddings']
            target_transactions_fixed_company_features_batch = target_transactions_data['fixed_company_features']
            target_transactions_embeddings = self.company_embedding_model(target_transactions_price_batch,
                                                                              target_transactions_fundamentals_batch,
                                                                        target_transactions_company_description_batch,
                                                                        target_transactions_fixed_company_features_batch)
            print(f"Target transactions embeddings shape: {target_transactions_embeddings.shape}")
            return predicted_transaction, target_transactions_embeddings
    
if __name__ == "__main__":
    price_model = PriceModel(embedding_dim=6, num_heads=3, ffnn_hidden_dim=124,
                            num_ffnn_hidden_layers=4, activation_function=nn.GELU,
                            ffnn_dropout_prob=0.1, attention_dropout_prob=0.1,
                            batch_first=True, num_encoder_blocks=3,
                            max_seq_len=1000, prepend_embedding_vector=True)
    
    fundamentals_model = FundamentalsModel(embedding_dim=124,
                                        num_heads=4, ffnn_hidden_dim=124,
                                        num_ffnn_hidden_layers=2,
                                        activation_function=nn.GELU, ffnn_dropout_prob=0.1,
                                        attention_dropout_prob=0.1, batch_first=True,
                                        num_encoder_blocks=3, force_inner_dimensions=False,
                                        max_seq_len=1000, prepend_embedding_vector=True)

    company_desciption_model = CompanyDescriptionModel(hidden_dim=1536, num_hidden_layers=2,
                                                    output_dim=1536, dropout_prob=0.1,
                                                    activation_function=nn.GELU, embedding_dim=1536)

    head_combination_model = HeadCombinationLayer(input_dims=[6, 124, 1536], num_hidden_layers=4,
                                                        final_dim = 1536, num_heads = 6)
                                                        
    company_embedding_model = CompanyEmbeddingModel(price_model, fundamentals_model,
                                                company_desciption_model, head_combination_model)

    prev_trans_portf_model = PreviousTransactionsPortfolioModel(embedding_dim=1543, num_heads=2, ffnn_hidden_dim=2048,
                                                 num_ffnn_hidden_layers=3, ffnn_dropout_prob=0.1,
                                                 attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True,
                                                 num_encoder_blocks=3, num_decoder_blocks=3, force_inner_dimensions=True)

    next_transaction_model = NextTransactionsModel(embedding_dim=1544, num_heads=2, ffnn_hidden_dim=2048,
                                                   num_ffnn_hidden_layers=3, ffnn_dropout_prob=0.1,
                                                    attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True,
                                                    num_encoder_blocks=3, force_inner_dimensions=False)
    
    model = EndToEndModel(company_embedding_model=company_embedding_model, company_embedding_model_output_dim=1543,
                          prev_trans_portf_model=prev_trans_portf_model, next_transaction_model=next_transaction_model)

    custom_dataset = CustomDataset(prices_csv=r"C:\repos\Deep-learning-trj\data\monthly_prices\example_prices.csv",
                                fundamentals_csv=r"C:\repos\Deep-learning-trj\data\fundamentals\example_fundamentals.csv",
                                company_descriptions_embeddings_csv=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv",
                                fixed_company_features_csv=r"C:\repos\Deep-learning-trj\data\fixed_company_features\example_company_features.csv",
                                transactions_csv=r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")

    dataloader = DataLoader(custom_dataset, batch_size=None, shuffle=True)

    for previous_transaction_companies_data, portfolio_companies_data, target_transactions_data in dataloader:
        prediction, target_embeddings= model(previous_transaction_companies_data, portfolio_companies_data, target_transactions_data)
        print(f"Prediction shape: {prediction.shape}")
        print(f"Target embeddings shape: {target_embeddings.shape}")
        break