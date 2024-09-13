import torch
import os
import sys
from price_model import PriceModel
from company_description_model import CompanyDescriptionModel
from fundamentals_model import FundamentalsModel
from torch import nn
from torch.utils.data import DataLoader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.custom_dataset import CustomDataset

embedding_dim = 2
num_heads = 4
ffnn_hidden_dim = 2
num_ffnn_hidden_layers = 3
activation_function = nn.GELU
ffnn_dropout_prob = 0.1
attention_dropout_prob = 0.1
batch_first = True
num_transformer_blocks = 3
max_seq_len = 1000
prepend_embedding_vector = True

price_batch = torch.ones(2, 10, 2)
price_model = PriceModel(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob, batch_first=batch_first,
                num_transformer_blocks=num_transformer_blocks, max_seq_len=max_seq_len, prepend_embedding_vector=prepend_embedding_vector)

#Create Fundamentals Model
embedding_dim = 124
num_heads = 4
ffnn_hidden_dim=124
num_ffnn_hidden_layers = 8
activation_function=nn.GELU
ffnn_dropout_prob=0.1,
attention_dropout_prob=0.1
batch_first=True
num_transformer_blocks=3
force_inner_dimensions=True
max_seq_len=1000
prepend_embedding_vector=False
fundamentals_model = FundamentalsModel(embedding_dim=embedding_dim,
                                        num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function, ffnn_dropout_prob=ffnn_dropout_prob,
                                       attention_dropout_prob=attention_dropout_prob, batch_first=batch_first, num_transformer_blocks=num_transformer_blocks, force_inner_dimensions=force_inner_dimensions, max_seq_len=max_seq_len, prepend_embedding_vector=prepend_embedding_vector)