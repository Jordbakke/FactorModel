import torch
import pandas as pd
import sys
import os
sys.path.append(r"C:\repos\Deep-learning-trj")
from data.code import data_utils
from model.test_train import train_and_evaluate, evaluate_and_save, shuffle_split_load_dataset
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
from model.test_train import train_and_evaluate, evaluate_and_save, shuffle_split_load_dataset

def create_model():

    embedding_dim = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_tensor").shape[-1]
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
    embedding_dim = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_tensor").shape[-1]
    num_heads = 8
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
    
    embedding_dim = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_features_tensor").shape[-1]
    ffnn_hidden_dim = 24
    output_dim=24
    num_ffnn_hidden_layers = 2
    activation_function = nn.GELU
    ffnn_dropout_prob = 0.1
    company_features_model = CompanyFeaturesModel(embedding_dim=embedding_dim, hidden_dim=ffnn_hidden_dim, output_dim=output_dim,
                                                  num_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                                                    dropout_prob=ffnn_dropout_prob, dropout_layer_frequency=2)
    
    embedding_dim = data_utils.load_tensor(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\macro_tensor").shape[-1]
    num_heads = 7
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
    price_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_tensor"
    price_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\price_key_padding_mask"
    fundamentals_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_tensor"
    fundamentals_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\fundamentals_key_padding_mask"
    fundamentals_missing_features_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_mask"
    fundamentals_missing_features_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\missing_features_fundamentals_key_padding_mask"
    company_features_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_features_tensor"
    company_description_input_ids_tensor_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_descriptions_input_ids_tensor"
    company_description_key_padding_mask_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_descriptions_key_padding_mask"
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
    
    # company_embedding_model = create_model()
    # custom_dataset = create_custom_dataset()
    # train_dataloader, val_dataloader, test_dataloader = shuffle_split_load_dataset(custom_dataset, batch_size=32, collate_fn=custom_dataset.custom_collate_fn)

    # optimizer = torch.optim.AdamW(company_embedding_model.parameters(), lr=1e-4, weight_decay=1e-2)
    # num_epochs = 1
    # num_training_steps = num_epochs * len(train_dataloader)
    # num_warmup_steps = 100000 # Warmup steps
    # T_max = num_training_steps - num_warmup_steps  # Cosine decay phase

    # warmup_scheduler = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=num_warmup_steps)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=1e-6)

    # # Combine using SequentialLR
    # scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
    #                          milestones=[num_warmup_steps])

    # # train_dataloader, val_dataloader, loss_fn, model=None, optimizer = None, scheduler=None,
    # #                    existing_model_path=None, device="cpu", num_epochs=1, eval_frequency=1000, is_siamese_network=False,
    # #                    max_val_batches=10000, save_path=None, **loss_fn_kwargs
    # model, train_losses, val_losses, model_val_loss = train_and_evaluate(train_dataloader=train_dataloader, val_dataloader=val_dataloader,
    #                                                                      loss_fn=embedding_correlation_dot_product_loss, model=company_embedding_model, optimizer=optimizer,
    #                                                                      scheduler=scheduler, device="cpu", num_epochs=1, is_siamese_network=True, eval_frequency=100000, max_val_batches=10000,
    #                                                                      save_path = r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data\company_embedding_model.pt",
    #                                                                      correlation_weight=0.7, dot_product_weight=0.3)


    for tensor_path in os.listdir(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data"):
        if not tensor_path.endswith(".parquet"):
            tensor =  data_utils.load_tensor(os.path.join(r"C:\repos\Deep-learning-trj\stock_correlation_project\test_train_data", tensor_path))
            num_elements = tensor.numel()  # Total number of elements
            element_size = tensor.element_size()  # Size of one element in bytes
            size_in_bytes = num_elements * element_size
            size_in_mb = size_in_bytes / (1024 * 1024* 1024) * 10000

            print("Size of tensor {} is {} GB".format(tensor_path, size_in_mb))

