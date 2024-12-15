import torch
import sys
sys.path.append(r"C:\repos\Deep-learning-trj")
from price_model import PriceModel
from company_description_model import CompanyDescriptionModel
from fundamentals_model import FundamentalsModel
from model.utils import HeadCombinationLayer
from torch import nn
from torchinfo import summary

class CompanyEmbeddingModel(nn.Module):
    def __init__(self, price_model, fundamentals_model, company_description_model,
                 head_combination_model, company_fixed_features_dim):
        super(CompanyEmbeddingModel, self).__init__()
        self.price_model = price_model
        self.fundamentals_model = fundamentals_model
        self.company_description_model = company_description_model
        self.head_combination_model = head_combination_model
        self.output_dim = head_combination_model.final_dim + company_fixed_features_dim
        
    def forward(self, price_batch, fundamentals_batch, company_description_batch, fixed_company_features_batch,
                price_layer_normalization=False, fundamentals_layer_normalization=False,
                key_padding_mask_price_batch=None,
                key_padding_mask_fundamentals_batch=None):
        
        price_cls_vector = self.price_model(price_batch, layer_normalization=price_layer_normalization,
                                           key_padding_mask=key_padding_mask_price_batch
                                           ) #0:1 to avoid unsqueeze(1) at the end
        
        fundamentals_embedding = self.fundamentals_model(fundamentals_batch, layer_normalization=fundamentals_layer_normalization,
                                                         key_padding_mask=key_padding_mask_fundamentals_batch
                                                         )[:, 0:1, :]
        
        company_description_embedding = self.company_description_model(company_description_batch)[:, 0:1, :]
        company_embedding = self.head_combination_model(price_cls_vector,
                                                        fundamentals_embedding,
                                                        company_description_embedding) #(batch_size, 1, embedding_dim)
        company_embedding = torch.cat([company_embedding, fixed_company_features_batch], dim=-1)
        #add company features horizontally
        return company_embedding
    
if __name__ == "__main__":
    embedding_dim = 6
    num_heads = 3
    ffnn_hidden_dim = 124
    num_ffnn_hidden_layers = 2
    activation_function = nn.GELU
    ffnn_dropout_prob = 0.1
    attention_dropout_prob = 0.1
    batch_first = True
    num_encoder_blocks = 3
    max_seq_len = 1000
    prepend_cls_vector = True

    price_batch = torch.ones(2, 10, 2)
    price_model = PriceModel(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                            num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                            ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob,
                            batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                            max_seq_len=max_seq_len, prepend_cls_vector=prepend_cls_vector)

    #Create Fundamentals Model
    embedding_dim = 124
    num_heads = 4
    ffnn_hidden_dim=124
    num_ffnn_hidden_layers = 2
    activation_function=nn.GELU
    ffnn_dropout_prob=0.1,
    attention_dropout_prob=0.1
    batch_first=True
    num_encoder_blocks=3
    max_seq_len=1000
    prepend_cls_vector=True
    fundamentals_model = FundamentalsModel(embedding_dim=embedding_dim,
                                        num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                        activation_function=activation_function, ffnn_dropout_prob=ffnn_dropout_prob,
                                        attention_dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                        num_encoder_blocks=num_encoder_blocks,
                                        max_seq_len=max_seq_len, prepend_cls_vector=prepend_cls_vector)

    hidden_dim = 1536
    num_hidden_layers = 2 
    output_dim = 1536
    dropout_prob = 0.1
    activation_function = nn.GELU
    embedding_dim = 1536
    company_desciption_model = CompanyDescriptionModel(hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers,
                                                    output_dim=output_dim, dropout_prob=dropout_prob,
                                                    activation_function=activation_function, embedding_dim=embedding_dim)

    head_combination_model = HeadCombinationLayer(input_dims=[6, 124, 1536], num_ffnn_hidden_layers=4,
                                                        final_dim = 1536, num_heads = 6)
                                                        
    company_embedding_model = CompanyEmbeddingModel(price_model, fundamentals_model,
                                                company_desciption_model, head_combination_model, 7)
    prices = torch.randn(4, 3, 6)
    fundamentals = torch.randn(4, 17, 124)
    company_description = torch.randn(4, 1, 1536)
    fixed_company_features = torch.randn(4, 1, 7)
    result = company_embedding_model(prices, fundamentals, company_description, fixed_company_features)
    #print(result.shape)
    summary(company_embedding_model, input_size=[(2, 5, 6), (2, 2, 124), (2, 1, 1536), (2, 1, 7)])
