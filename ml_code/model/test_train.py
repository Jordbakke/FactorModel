import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import tqdm
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

if __name__ == "__main__":
    pass
      