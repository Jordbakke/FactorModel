import os
import sys
import torch
import custom_loss_functions
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from end_to_end_model import EndToEndModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def plot_training_loss(training_loss:dict):
    # Extract keys and values from the dictionary
    x_values = list(training_loss.keys())
    y_values = list(training_loss.values())
    
    # Create the line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, marker='o')
  
    plt.xlabel('Iteration Count')
    plt.ylabel('Avg Training Loss')
    plt.title('Training loss vs iteration count')

    plt.grid(True)
    plt.show()

def train(end_to_end_model, custom_dataset, device, loss_fn=custom_loss_functions.MinEuclideanLoss,
          num_epochs=10,lr=1e-4, weight_decay=1e-4, batch_size=None,
          save_path=r"C:\repos\Deep-learning-trj\model\saved_model.pth", loss_display_frequency=10000):

    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(end_to_end_model.parameters(), lr=lr, weight_decay=weight_decay)

    end_to_end_model = end_to_end_model.to(device)

    training_loss = {}
    total_loss = 0.0
    iteration_count = 0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        end_to_end_model.train()  # Set model to training mode
        epoch_loss = 0.0
        # Progress bar for tracking training progress
        for i, batch in enumerate(tqdm(dataloader, desc="Training Batches")):
            # Load tensor to device
            previous_transaction_companies_data, portfolio_companies_data, target_transactions_data = batch
            previous_transaction_companies_data = {key: value.to(device) for key, value in previous_transaction_companies_data.items()}
            portfolio_companies_data = {key: value.to(device) for key, value in portfolio_companies_data.items()}
            target_transactions_data = {key: value.to(device) for key, value in target_transactions_data.items()}

            optimizer.zero_grad()
            transaction_prediction, target_transaction_embeddings = end_to_end_model(previous_transaction_companies_data,
                                                      portfolio_companies_data,
                                                      target_transactions_data)

            loss = loss_fn(transaction_prediction, target_transaction_embeddings)
            loss.backward() #backward prop to find the partial derivatives
            optimizer.step() #Iteratve through model parameters and update based on their gradient attribute set by loss.backward()
            
            total_loss += loss.item()
            iteration_count += 1

            # Log the loss every ith iteration
            if iteration_count % loss_display_frequency == 0:
                avg_loss = total_loss / iteration_count
                training_loss[iteration_count] = avg_loss
                plot_training_loss(training_loss)

            epoch_loss += loss.item()
        
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch + 1} Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(end_to_end_model.state_dict(), save_path)

    print("Training complete.")

    return training_loss


            

      