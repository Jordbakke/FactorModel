import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json


import torch
import custom_loss_functions
import matplotlib.pyplot as plt
import time
#import optuna
from company_embedding_model import CompanyEmbeddingModel
from prev_trans_portf_model import PreviousTransactionsPortfolioModel
from next_transactions_model import NextTransactionsModel
from price_model import PriceModel
from fundamentals_model import FundamentalsModel
from company_description_model import CompanyDescriptionModel
from utils import HeadCombinationLayer, PositionalEncoding
from model.transactions_project.end_to_end_model import EndToEndModel
from data.custom_dataset import CustomDataset
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from data.custom_dataset import CustomDataset, create_dataloaders

def plot_model_loss(model_loss: dict):
    # Extract keys and values from the dictionary
    x_values = list(model_loss.keys())
    y_values = list(model_loss.values())

    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o')

    plt.xlabel('Iteration Count')
    plt.ylabel('Avg Training Loss')
    plt.title('Training loss vs iteration count')

    plt.grid(True)
    plt.show()

def load_model(model, model_path=r"C:\repos\Deep-learning-trj\model\saved_model.pth"):
    checkpoint = torch.load(model_path)
    epoch_train_losses = checkpoint.get('epoch_train_losses', {})
    epoch_val_losses = checkpoint.get('epoch_val_losses', {})
    state_dict = checkpoint['model_state_dict']
    model_val_loss = checkpoint['model_val_loss']  # Best epoch loss
    model.load_state_dict(state_dict)

    return model, epoch_train_losses, epoch_val_losses, model_val_loss

def retrieve_model_loss(model_path=r"C:\repos\Deep-learning-trj\model\saved_model.pth"):
    checkpoint = torch.load(model_path)
    return checkpoint['epoch_train_losses'], checkpoint['epoch_val_losses']

def train_and_evaluate(model, train_dataloader: DataLoader, val_dataloader: DataLoader, existing_model_path=None, device="cpu",
          loss_fn=custom_loss_functions.min_euclidean_distance, num_epochs=2, lr=1e-4,
          weight_decay=1e-4, save_path=r"C:\repos\Deep-learning-trj\model\saved_model.pth"):
    
    epoch_train_losses = {}
    epoch_val_losses = {}
    model_val_loss = float('inf')  # Best model loss so far

    # Load existing model and losses if provided
    if existing_model_path is not None and os.path.exists(existing_model_path):
        print("Loading existing model and loss history")
        model, epoch_train_losses, epoch_val_losses, model_val_loss = load_model(model, existing_model_path)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model = model.to(device)
    epoch_count = len(epoch_train_losses)
    assert epoch_count == len(epoch_val_losses), "Epoch losses and val losses should have the same length"

    for i in range(num_epochs):
        print(f"Epoch {i + 1}/{num_epochs}")
        model.train()  # Set model to training mode
        epoch_train_loss = 0.0  # Track the loss for the epoch
        epoch_val_loss = 0.0

        for batch in tqdm(train_dataloader, desc="Training Batches"):
            # Move tensors to device
            start = time.time()
            predicted_transactions_batch, target_transactions_batch, target_transactions_mask_batch = model(batch)

            optimizer.zero_grad() #reset gradients

            # Calculate the loss and backpropagate
            loss = loss_fn(predicted_transactions_batch, target_transactions_batch, target_transactions_mask_batch)
            loss.backward()
            optimizer.step()

            # Update loss metrics
            loss_value = loss.item()  # Get the scalar loss value
            epoch_train_loss += loss_value
            print(f"Time taken: {time.time() - start}")
        epoch_train_loss /= len(train_dataloader)  # Average loss for the epoch
        epoch_train_losses[epoch_count + 1] = epoch_train_loss
        print(f"Epoch {i + 1} train Loss: {epoch_train_loss:.4f}")

        model.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation for validation
            for batch in tqdm(val_dataloader, desc="Validation Batches"):
                predicted_transactions_batch, target_transactions_batch, target_transactions_mask_batch = model(batch)
                loss = loss_fn(predicted_transactions_batch, target_transactions_batch, target_transactions_mask_batch)
                val_loss = loss.item()
                epoch_val_loss += val_loss
        
        epoch_val_loss /= len(val_dataloader)
        print(f"Epoch {i + 1} val Loss: {epoch_val_loss:.4f}")
        epoch_val_losses[epoch_count + 1] = epoch_val_loss
        
        epoch_count += 1

        # Save the best model if the current epoch's loss is the best
        if epoch_val_loss < model_val_loss and save_path is not None:
            print(f"Saving model at epoch {epoch_count}, new best val loss: {epoch_val_loss:.4f}")
            model_val_loss = epoch_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch_train_losses': epoch_train_losses,
                'epoch_val_losses': epoch_val_losses,
                'model_val_loss': model_val_loss
            }, save_path)

    print("Training complete.")
    return model_val_loss

def objective(trial):

    # ffnn_hidden dim bust be divisible by num_heads
    valid_heads = [2, 3, 4, 6, 8, 10, 12]

    num_heads = trial.suggest_categorical("num_heads", valid_heads)
    num_ffnn_hidden_layers = trial.suggest_int("num_ffnn_hidden_layers", 1, 6)
    ffnn_dropout_prob = trial.suggest_float("ffnn_dropout_prob", 0.0, 0.3)
    attention_dropout_prob = trial.suggest_float("attention_dropout_prob", 0.0, 0.3)
    num_encoder_blocks = trial.suggest_int("num_encoder_blocks", 1, 6)

    valid_price_model_ffnn_hidden_dim = [i for i in range(3, 512) if i % num_heads == 0]
    price_model_ffnn_hidden_dim = trial.suggest_categorical("price_model_ffnn_hidden_dim", valid_price_model_ffnn_hidden_dim)

    valid_fundamentals_model_ffnn_hidden_dim = [i for i in range(100, 512) if i % num_heads == 0]
    fundamentals_model_ffnn_hidden_dim = trial.suggest_categorical("fundamentals_model__ffnn_hidden_dim", valid_fundamentals_model_ffnn_hidden_dim)

    valid_previous_transactions_portfolio_model_ffnn_hidden_dim = [i for i in range(1000, 2048) if i % num_heads == 0]
    previous_transactions_portfolio_model_ffnn_hidden_dim = trial.suggest_categorical("previous_transactions_portfolio_model__ffnn_hidden_dim",
                                                                                      valid_previous_transactions_portfolio_model_ffnn_hidden_dim)
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 6)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 6)

    valid_next_transaction_model_ffnn_hidden_dim = [i for i in range(1000, 2048) if i % num_heads == 0]
    next_transaction_model_ffnn_hidden_dim = trial.suggest_categorical("next_transaction_model__ffnn_hidden_dim",
                                                                       valid_next_transaction_model_ffnn_hidden_dim)
    lr = trial.suggest_float("lr", 1e-5, 1e-3)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3)
    
    price_model = PriceModel(embedding_dim=3, num_heads=num_heads, ffnn_hidden_dim=price_model_ffnn_hidden_dim,
                            num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=nn.GELU,
                            ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob,
                            batch_first=True, num_encoder_blocks=num_encoder_blocks,
                            max_seq_len=1000, prepend_cls_vector=True)
    
    fundamentals_model = FundamentalsModel(embedding_dim=124,
                                        num_heads=num_heads, ffnn_hidden_dim=fundamentals_model_ffnn_hidden_dim,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                        activation_function=nn.GELU, ffnn_dropout_prob=ffnn_dropout_prob,
                                        attention_dropout_prob=attention_dropout_prob, batch_first=True,
                                        num_encoder_blocks=num_encoder_blocks,
                                        max_seq_len=1000, prepend_cls_vector=True)

    company_desciption_model = CompanyDescriptionModel(embedding_dim=1536, hidden_dim=1536, num_hidden_layers=num_ffnn_hidden_layers,
                                                    output_dim=1536, dropout_prob=ffnn_dropout_prob,
                                                    activation_function=nn.GELU)

    head_combination_model = HeadCombinationLayer(input_dims=[3, 124, 1536], num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                                        final_dim = 1536, num_heads = num_heads)
                                                        
    company_embedding_model = CompanyEmbeddingModel(price_model, fundamentals_model,
                                                company_desciption_model, head_combination_model, company_fixed_features_dim=7)

    prev_trans_portf_model = PreviousTransactionsPortfolioModel(embedding_dim=1544,
                                                                num_heads=num_heads, ffnn_hidden_dim=previous_transactions_portfolio_model_ffnn_hidden_dim,
                                                                num_encoder_layers=num_encoder_layers,
                                                 dropout_prob=ffnn_dropout_prob, num_decoder_layers=num_decoder_layers,
                                                 activation_function="gelu", batch_first=True,
                                                 )

    next_transaction_model = NextTransactionsModel(embedding_dim=prev_trans_portf_model.output_dim, num_heads=num_heads,
                                                   ffnn_hidden_dim=next_transaction_model_ffnn_hidden_dim,
                                                    output_dim=company_embedding_model.output_dim,
                                                   num_ffnn_hidden_layers=num_ffnn_hidden_layers, ffnn_dropout_prob=ffnn_dropout_prob,
                                                    attention_dropout_prob=attention_dropout_prob, activation_function=nn.GELU, batch_first=True,
                                                    num_encoder_blocks=num_encoder_blocks)
    
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
    
    model_val_loss = train_and_evaluate(model, train_dataloader, val_dataloader, existing_model_path=None, device="cpu",
          loss_fn=custom_loss_functions.min_euclidean_distance, num_epochs=2, lr=lr,
          weight_decay=weight_decay, save_path=None)

    return model_val_loss

def run_optuna_study(save_path=r"C:\repos\Deep-learning-trj\model\best_params.json"):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    best_params = study.best_params
    with open(save_path, 'w') as f:
        json.dump(best_params, f)
    return best_params

def test_model(model, test_dataloader: DataLoader, device="cpu", loss_fn=custom_loss_functions.min_euclidean_distance):
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    total_loss = 0.0
    num_batches = len(test_dataloader)
    
    with torch.no_grad():  # Disable gradient computation for testing
        for batch in tqdm(test_dataloader, desc="Testing Batches"):
            # Move batch to the appropriate device
            
            # Perform forward pass
            predicted_transactions_batch, target_transactions_batch, target_transactions_mask_batch = model(batch)
            # Compute loss
            loss = loss_fn(predicted_transactions_batch, target_transactions_batch, target_transactions_mask_batch)
            total_loss += loss.item()


    # Average the total loss across all batches
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"Test Loss: {avg_loss:.4f}")
    
    return avg_loss


if __name__ == "__main__":
    pass
      