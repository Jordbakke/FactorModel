import os
import sys
import optuna
sys.path.append(r"C:\repos\Deep-learning-trj")
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import custom_loss_functions
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

def shuffle_split_load_dataset( dataset: Dataset, train_size: float = 0.8, test_size: float = 0.1, val_size: float = 0.1, batch_size: int = 32, num_workers: int = 0, collate_fn=None, shuffle: bool = True, random_seed: int = 42):
   
    # Validate split sizes
    total_size = train_size + test_size + val_size
    if not np.isclose(total_size, 1.0):
        raise ValueError("train_size, test_size, and val_size must sum to 1.0")

    # Calculate dataset indices
    dataset_size = len(dataset)
    train_count = int(train_size * dataset_size)
    test_count = int(test_size * dataset_size)
    val_count = dataset_size - train_count - test_count  # Remaining samples for validation

    # Generate shuffled indices
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    # Split indices
    train_indices = indices[:train_count]
    val_indices = indices[train_count:train_count + val_count]
    test_indices = indices[train_count + val_count:]

    # Create Subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )
    test_dataloader = DataLoader(
        test_subset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader, test_dataloader

def evaluate_and_save(model, optimizer, val_dataloader, train_losses, val_losses, loss_fn, device="cpu", 
                      model_val_loss=float('inf'), is_siamese_network=False, scheduler=None, max_val_batches=10000,
                      save_path=None, **loss_fn_kwargs):
    if max_val_batches < 0:
        raise ValueError("max_val_batches must be a positive integer")
    
    accumulated_val_loss = 0.0
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validation Batches")):
            if i >= max_val_batches:
                break

            if is_siamese_network:
                inputs1 = batch["input"]["input1"]
                inputs2 = batch["input"]["input2"]
                for key, value in inputs1.items():
                    inputs1[key] = value.to(device)
                for key, value in inputs2.items():
                    inputs2[key] = value.to(device)
                
                target = batch["target"].to(device)

                # Use autocast for mixed precision
                with autocast():
                    output1 = model(**inputs1)
                    output2 = model(**inputs2)
                    loss = loss_fn(output1, output2, target, **loss_fn_kwargs)
            else:
                inputs = batch["input"]
                for key, value in inputs.items():
                    inputs[key] = value.to(device)

                target = batch["target"].to(device)

                # Use autocast for mixed precision
                with autocast(device=device):
                    output = model(**inputs)
                    loss = loss_fn(output, target, **loss_fn_kwargs)

            accumulated_val_loss += loss.item()

    val_loss = accumulated_val_loss / (i + 1)
    val_losses.append(val_loss)

    # Save the best model
    if val_loss < model_val_loss:
        print(f"Validation loss: {val_loss:.4f} is better than previous best: {model_val_loss:.4f}")
        model_val_loss = val_loss
        if save_path:
            print(f"Saving best model with validation loss: {val_loss:.4f} vs previous best: {model_val_loss:.4f}")
            torch.save({
                'model': model,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'model_val_loss': model_val_loss
            }, save_path)

    return val_losses, model_val_loss

def train_and_evaluate(train_dataloader, val_dataloader, loss_fn, model=None, optimizer=None, scheduler=None,
                       existing_model_path=None, device="cpu", num_epochs=1, eval_frequency=1000, is_siamese_network=False,
                       max_val_batches=10000, save_path=None, **loss_fn_kwargs):
    
    if not model and not existing_model_path:
        raise ValueError("Either a model or an existing model checkpoint path must be provided")
    if num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer")
    if eval_frequency <= 0:
        raise ValueError("eval_frequency must be a positive integer")
    
    if not optimizer:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)

    train_losses = []
    val_losses = []
    model_val_loss = float('inf')

    # Load model checkpoint if provided
    if existing_model_path:
        if not os.path.exists(existing_model_path):
            raise FileNotFoundError(f"Model checkpoint file not found: {existing_model_path}")
        
        print("Loading existing model and loss history")
        checkpoint = torch.load(existing_model_path)
        model = checkpoint['model']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        model_val_loss = checkpoint['model_val_loss']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'], to_map_location=device)
        if scheduler:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'], to_map_location=device)

    model = model.to(device)

    # Initialize GradScaler for mixed precision
    scaler = GradScaler(device=device)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        accumulated_train_loss = 0.0

        for i, batch in enumerate(tqdm(train_dataloader, desc="Training Batches")):
            
            if is_siamese_network:
                inputs1 = batch["input"]["input1"]
                inputs2 = batch["input"]["input2"]
                for key, value in inputs1.items():
                    inputs1[key] = value.to(device)
                for key, value in inputs2.items():
                    inputs2[key] = value.to(device)
                target = batch["target"].to(device)

                # Use autocast for mixed precision
                with autocast(device=device):
                    output1 = model(**inputs1)
                    output2 = model(**inputs2)
                    loss = loss_fn(output1, output2, target, **loss_fn_kwargs)
            else:
                inputs = batch["input"]
                for key, value in inputs.items():
                    inputs[key] = value.to(device)

                target = batch["target"].to(device)

                # Use autocast for mixed precision
                with autocast(device=device):
                    output = model(**inputs)
                    loss = loss_fn(output, target, **loss_fn_kwargs)

            optimizer.zero_grad()

            # Scale the loss for backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            accumulated_train_loss += loss.item()

            # Evaluate periodically
            if (i + 1) % eval_frequency == 0:
                avg_train_loss = accumulated_train_loss / eval_frequency
                train_losses.append(avg_train_loss)
                accumulated_train_loss = 0.0
                val_losses, model_val_loss = evaluate_and_save(model=model, optimizer=optimizer, val_dataloader=val_dataloader,
                                                               train_losses=train_losses, val_losses=val_losses, loss_fn=loss_fn,
                                                               device=device, model_val_loss=model_val_loss, is_siamese_network=is_siamese_network,
                                                               scheduler=scheduler, max_val_batches=max_val_batches, save_path=save_path, **loss_fn_kwargs)
        # End-of-epoch evaluation
        if (i + 1) % eval_frequency != 0:
            avg_train_loss = accumulated_train_loss / (i + 1)
            train_losses.append(avg_train_loss)
            val_losses, model_val_loss = evaluate_and_save(model=model, optimizer=optimizer, val_dataloader=val_dataloader,
                                                            train_losses=train_losses, val_losses=val_losses, loss_fn=loss_fn,
                                                            device=device, model_val_loss=model_val_loss, is_siamese_network=is_siamese_network,
                                                            scheduler=scheduler, max_val_batches=max_val_batches, save_path=save_path, **loss_fn_kwargs)

    return model, train_losses, val_losses, model_val_loss

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


if __name__ == "__main__":
    pass
      