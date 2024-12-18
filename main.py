import torch
import pandas as pd
from model.test_train import train_and_evaluate, evaluate_and_save
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from company_embedding_model import CompanyEmbeddingModel
from price_model import PriceModel
from company_description_model import CompanyDescriptionModel
from fundamentals_model import FundamentalsModel
from company_features_model import CompanyFeaturesModel
from macro_model import MacroModel
from model.utils import HeadCombinationLayer
from torch import nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from custom_data import CustomDataset
from model.custom_loss_functions import embedding_correlation_dot_product_loss

def create_model():
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
    
    return company_embedding_model

def create_custom_dataset():


    mapping_df = pd.read_parquet(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\mapping_df.parquet")
    price_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\prices_tensor"
    price_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\prices_key_padding_mask"
    fundamentals_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_tensor"
    fundamentals_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_key_padding_mask"
    fundamentals_missing_features_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_mask"
    fundamentals_missing_features_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_key_padding_mask"
    company_features_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_features_tensor"
    company_description_input_ids_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_description_input_ids_tensor"
    company_description_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_description_key_padding_mask"
    macro_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_tensor"
    macro_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_key_padding_mask"
    target_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\target_tensor"
    custom_dataset = CustomDataset(mapping_df=mapping_df, price_tensor_path=price_tensor_path, price_key_padding_mask_path=price_key_padding_mask_path,
                                   fundamentals_tensor_path=fundamentals_tensor_path, fundamentals_key_padding_mask_path=fundamentals_key_padding_mask_path,
                 fundamentals_missing_features_mask_path=fundamentals_missing_features_mask_path, fundamentals_missing_features_key_padding_mask_path=fundamentals_missing_features_key_padding_mask_path,
                 company_features_tensor_path=company_features_tensor_path, macro_tensor_path=macro_tensor_path, macro_key_padding_mask_path=macro_key_padding_mask_path,
                 company_description_input_ids_tensor_path=company_description_input_ids_tensor_path, company_description_key_padding_mask_path=company_description_key_padding_mask_path, target_tensor_path=target_tensor_path)

    return custom_dataset

if __name__ == "__main__":
    
    company_embedding_model = create_model()
    custom_dataset = create_custom_dataset()
    optimizer = torch.optim.AdamW(company_embedding_model.parameters(), lr=1e-4, weight_decay=1e-2)
    num_epochs = 1
    num_training_steps = num_epochs * len(custom_dataset) 
    num_warmup_steps = 100000     # Warmup steps
    T_max = num_training_steps - num_warmup_steps  # Cosine decay phase

    warmup_scheduler = LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=num_warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    # Combine using SequentialLR
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[num_warmup_steps])
