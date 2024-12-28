import torch
import math
from torch import nn
import numpy as np


def min_euclidean_distance(predicted_transactions, target_transactions, target_transactions_padding_mask=None):

    """
    Compute the minimum Euclidean distance between the predicted transaction and the target transactions for each sample
    """

    # cdist computes the pairwise Euclidean distances between the vectors for each batch.
    distances = torch.cdist(predicted_transactions, target_transactions)  # shape: (batch_size, num_future_transactions, num_predicted_transactions)
    
    # If a padding mask is provided, use it to ignore padded target transactions
    if target_transactions_padding_mask is not None:
        # Expand mask to match the shape of distances (batch_size, num_predicted_transactions, num_future_transactions)
        expanded_mask = target_transactions_padding_mask.unsqueeze(1).expand_as(distances)  # Expand along predicted_transactions dimension
        # Use out-of-place masked_fill to avoid in-place modification
        distances = distances.masked_fill(expanded_mask, float('inf'))  # Use masked_fill instead of masked_fill_

    # Compute the minimum distances, ignoring padded transactions
    min_distances, _ = torch.min(distances, dim=2)  # Along the num_future_transactions axis
    average_batch_distance = torch.mean(min_distances)
    return average_batch_distance

def embedding_correlation_cosine_loss(embedding1: torch.tensor, embedding2: torch.tensor, target_correlation_tensor: torch.tensor):

    """
    Compute the cosine similarity between the embeddings and compare it with the target correlation tensor
    """

    embedding1_norm = embedding1 / torch.linalg.vector_norm(embedding1, ord=2, dim=-1, keepdim=True)
    embedding2_norm = embedding2 / torch.linalg.vector_norm(embedding2, ord=2, dim=-1, keepdim=True)
    cosine_similarity_tensor = torch.matmul(embedding1_norm, embedding2_norm.transpose(1, 2))
    loss = nn.functional.mse_loss(cosine_similarity_tensor, target_correlation_tensor)
    
    return loss

if __name__ == "__main__":
    t1 = torch.ones(2, 1, 2)
    t2 = torch.ones(2, 1, 2)
    t2[0, 0, :] = t2[0, 0, :] + 1

    loss = embedding_correlation_cosine_loss(t1, t2, torch.ones(2, 1, 1))
    print(loss)



    