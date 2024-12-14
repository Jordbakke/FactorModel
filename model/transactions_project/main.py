import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import test_train
import torch
import custom_loss_functions
from data.custom_dataset import CustomDataset, create_dataloaders
from torch.utils.data import DataLoader
from company_embedding_model import CompanyEmbeddingModel
from prev_trans_portf_model import PreviousTransactionsPortfolioModel
from next_transactions_model import NextTransactionsModel
from price_model import PriceModel
from fundamentals_model import FundamentalsModel
from company_description_model import CompanyDescriptionModel
from utils import HeadCombinationLayer, PositionalEncoding
from model.transactions_project.end_to_end_model import EndToEndModel
from torchinfo import summary
from torch import nn

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    price_model = PriceModel(embedding_dim=3, num_heads=3, ffnn_hidden_dim=124,
                            num_ffnn_hidden_layers=4, activation_function=nn.GELU,
                            ffnn_dropout_prob=0.1, attention_dropout_prob=0.1,
                            batch_first=True, num_encoder_blocks=3,
                            max_seq_len=1000, prepend_cls_vector=True)
    
    fundamentals_model = FundamentalsModel(embedding_dim=124,
                                        num_heads=4, ffnn_hidden_dim=124,
                                        num_ffnn_hidden_layers=2,
                                        activation_function=nn.GELU, ffnn_dropout_prob=0.1,
                                        attention_dropout_prob=0.1, batch_first=True,
                                        num_encoder_blocks=3,
                                        max_seq_len=1000, prepend_cls_vector=True)

    company_desciption_model = CompanyDescriptionModel(embedding_dim=1536,hidden_dim=1536, num_hidden_layers=4,
                                                    output_dim=1536, dropout_prob=0.1,
                                                    activation_function=nn.GELU)

    head_combination_model = HeadCombinationLayer(input_dims=[3, 124, 1536], num_ffnn_hidden_layers=2,
                                                        final_dim = 1536, num_heads = 8)
                                                        
    company_embedding_model = CompanyEmbeddingModel(price_model, fundamentals_model,
                                                company_desciption_model, head_combination_model, company_fixed_features_dim=7)

    prev_trans_portf_model = PreviousTransactionsPortfolioModel(embedding_dim=1544,
                                                                num_heads=2, ffnn_hidden_dim=1544,
                                                                num_encoder_layers=2,
                                                 dropout_prob=0.1, num_decoder_layers=2,
                                                 activation_function="gelu", batch_first=True,
                                                 )

    next_transaction_model = NextTransactionsModel(embedding_dim=prev_trans_portf_model.output_dim, num_heads=2, ffnn_hidden_dim=1544,
                                                    output_dim=company_embedding_model.output_dim,
                                                   num_ffnn_hidden_layers=3, ffnn_dropout_prob=0.1,
                                                    attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True,
                                                    num_encoder_blocks=3)
    
    model = EndToEndModel(company_embedding_model=company_embedding_model,
                          prev_trans_portf_model=prev_trans_portf_model,
                          next_transaction_model=next_transaction_model)
    
    custom_dataset = CustomDataset(prices_csv=r"C:\repos\Deep-learning-trj\data\monthly_prices\example_prices.csv",
                                fundamentals_csv=r"C:\repos\Deep-learning-trj\data\fundamentals\example_fundamentals.csv",
                                company_descriptions_embeddings_csv=r"C:\repos\Deep-learning-trj\data\company_descriptions\company_description_embeddings.csv",
                                fixed_company_features_csv=r"C:\repos\Deep-learning-trj\data\fixed_company_features\example_company_features.csv",
                                transactions_csv=r"C:\repos\Deep-learning-trj\data\transactions\example_transactions.csv")

    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(custom_dataset=custom_dataset,
                                                                           collate_fn=custom_dataset.collate_fn,
                                                                           val_split=0.1, test_split=0.1, shuffle=True,
                                                                           random_seed=42
                                                                           )
    
    test_train.train_and_evaluate(model, train_dataloader, val_dataloader, existing_model_path=None, device="cpu",
          loss_fn=custom_loss_functions.min_euclidean_distance, num_epochs=2, lr=1e-4,
          weight_decay=1e-4, save_path=None)

   