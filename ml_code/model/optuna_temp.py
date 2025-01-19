import optuna
import json
import torch
import tqdm
from test_train import train_and_evaluate
from model import custom_loss_functions
from torch.utils.data import DataLoader

def objective(trial, model, train_dataloader, val_dataloader, int_dict, categorical_dict):

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

def test_model(model, test_dataloader: DataLoader, loss_fn, device="cpu"):
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
