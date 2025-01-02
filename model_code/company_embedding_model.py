import torch
import sys
sys.path.append(r"C:\repos\Deep-learning-trj")
from price_model import PriceModel
from company_description_model import CompanyDescriptionModel
from fundamentals_model import FundamentalsModel
from company_features_model import CompanyFeaturesModel
from macro_model import MacroModel
from model.utils import HeadCombinationLayer
from torch import nn
from torchinfo import summary

class CompanyEmbeddingModel(nn.Module):
    def __init__(self, price_model, fundamentals_model, company_description_model, company_features_model,
                 macro_model, head_combination_model):
        super(CompanyEmbeddingModel, self).__init__()
        self.price_model = price_model
        self.fundamentals_model = fundamentals_model
        self.macro_model = macro_model
        self.company_description_model = company_description_model
        self.company_features_model = company_features_model
        self.head_combination_model = head_combination_model
         
    def forward(self, price_tensor_batch, price_key_padding_mask_batch, fundamentals_tensor_batch, fundamentals_key_padding_mask_batch,
                fundamentals_missing_features_mask_batch, fundamentals_missing_features_key_padding_mask_batch,
                company_features_tensor_batch, company_description_input_ids_batch, company_description_key_padding_mask_batch,
                macro_tensor_batch, macro_key_padding_mask_batch):
        
        print("Macro tensor shape: ", macro_tensor_batch.shape)
        print("Price tensor shape: ", price_tensor_batch.shape)
        print("Fundamentals tensor shape: ", fundamentals_tensor_batch.shape)
        print("Company features tensor shape: ", company_features_tensor_batch.shape)
        print("Company description tensor shape: ", company_description_input_ids_batch.shape)
        
        macro_cls_tensor = self.macro_model(macro_tensor_batch, layer_normalization=True, key_padding_mask=macro_key_padding_mask_batch)
        price_cls_tensor1 = self.price_model(price_tensor_batch, layer_normalization=True, key_padding_mask=price_key_padding_mask_batch)
        fundamentals_cls_tensor1 = self.fundamentals_model(fundamentals_tensor_batch, missing_features_mask=fundamentals_missing_features_mask_batch,
                                                           tgt_key_padding_mask=fundamentals_missing_features_key_padding_mask_batch, src_key_padding_mask=fundamentals_key_padding_mask_batch,
                                                            memory_key_padding_mask=fundamentals_key_padding_mask_batch)
        company_description_cls_tensor1 = self.company_description_model(company_description_input_ids_batch, company_description_key_padding_mask_batch)
        company_features_tensor1 = self.company_features_model(company_features_tensor_batch)
        company_embedding1 = self.head_combination_model(price_cls_tensor1, fundamentals_cls_tensor1, company_description_cls_tensor1, company_features_tensor1, macro_cls_tensor)

        return company_embedding1
    
if __name__ == "__main__":
    embedding_dim = 2
    num_heads = 2
    ffnn_hidden_dim = 24
    num_ffnn_hidden_layers = 2
    activation_function = nn.GELU
    ffnn_dropout_prob = 0.1
    attention_dropout_prob = 0.1
    batch_first = True
    num_encoder_blocks = 6
    price_model = PriceModel(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                            num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                            ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob,
                            batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                            )

    #Create Fundamentals Model
    embedding_dim = 300
    num_heads = 6
    ffnn_hidden_dim=600
    num_ffnn_hidden_layers = 2
    activation_function=nn.GELU
    dropout=0.1
    batch_first=True
    num_encoder_blocks=6
    num_decoder_layers = 6

    # embedding_dim, num_heads=8, num_decoder_layers=6,
    #                                       ffnn_hidden_dim=2048, dropout=0.1, activation_function=nn.GELU, batch_first=True,
    #                                       num_ffnn_hidden_layers=2, num_encoder_blocks=6
    fundamentals_model = FundamentalsModel(embedding_dim=embedding_dim,
                                        num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                        activation_function=activation_function, dropout=dropout,
                                        batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                                        num_decoder_layers=num_decoder_layers)

    company_desciption_model = CompanyDescriptionModel(transformer_model="bert-large-uncased", embedding_dim=1024)
    
    embedding_dim = 24
    ffnn_hidden_dim = 24
    output_dim=24
    num_ffnn_hidden_layers = 2
    activation_function = nn.GELU
    ffnn_dropout_prob = 0.1
    company_features_model = CompanyFeaturesModel(embedding_dim=embedding_dim, hidden_dim=ffnn_hidden_dim, output_dim=output_dim,
                                                  num_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                                                    dropout_prob=ffnn_dropout_prob, dropout_layer_frequency=2)
    
    embedding_dim = 24
    num_heads = 8
    ffnn_hidden_dim = 48
    num_ffnn_hidden_layers = 2
    activation_function = nn.GELU
    ffnn_dropout_prob = 0.1
    attention_dropout_prob = 0.1
    batch_first = True
    num_encoder_blocks = 6
    macro_model = MacroModel(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                             num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                            ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob,
                            batch_first=batch_first, num_encoder_blocks=num_encoder_blocks)

    head_combination_model = HeadCombinationLayer(input_dims=[2, 300, 1024, 24, 24], projection_num_ffnn_hidden_layers=6,
                                                projection_hidden_dim=1024, projection_output_dim=1024, num_heads=8, 
                                                encoder_cls_hidden_dim=1024, encoder_cls_ffnn_num_hidden_layers=2, num_encoder_blocks=6,
                                                activation_function=nn.GELU, ffnn_dropout_prob=0.1, attention_dropout_prob=0.1,
                                                dropout_layer_frequency=2, batch_first=True
                                                )
                                              
    company_embedding_model = CompanyEmbeddingModel(price_model, fundamentals_model,
                                                company_desciption_model, company_features_model, macro_model, head_combination_model)
    prices_tensor = torch.randn(3, 60, 2)
    prices_tensor_key_padding_mask = torch.zeros(3, 60)
    fundamentals = torch.randn(3, 20, 300)
    fundamentals_key_padding_mask = torch.zeros(3, 20)
    fundamentals_missing_features_mask = torch.randn(3, 20, 300)
    fundamentals_missing_features_key_padding_mask = torch.zeros(3, 20)
    company_description_input_ids = torch.ones(3, 400, dtype=torch.long)
    company_description_attention_mask = torch.zeros(3, 400, dtype=torch.long)
    fixed_company_features = torch.randn(3, 1, 24)
    macro = torch.randn(3, 60, 24)
    macro_key_padding_mask = torch.zeros(3, 60)
    result = company_embedding_model(price_tensor_batch=prices_tensor, price_key_padding_mask_batch=prices_tensor_key_padding_mask,
                                     fundamentals_tensor_batch=fundamentals, fundamentals_key_padding_mask_batch=fundamentals_key_padding_mask,
                                    fundamentals_missing_features_mask_batch=fundamentals_missing_features_mask,
                                    fundamentals_missing_features_key_padding_mask_batch=fundamentals_missing_features_key_padding_mask,
                                    company_features_tensor_batch=fixed_company_features,
                                    company_description_input_ids_batch=company_description_input_ids,
                                    company_description_key_padding_mask_batch=company_description_attention_mask,
                                    macro_tensor_batch=macro, macro_key_padding_mask_batch=macro_key_padding_mask)
    
    total_params = sum(p.numel() for p in company_embedding_model.parameters())
    print(total_params)
    
