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

def cosine_dot_product_loss(predicted_cosine, target_cosine, predicted_dot_product, target_dot_product, cosine_weight, dot_product_weight):
    """
    Compute the cosine and dot product loss between the predicted and target cosine and dot product values
    """

    # Compute the cosine loss
    cosine_loss = nn.functional.mse_loss(predicted_cosine, target_cosine)

    # Compute the dot product loss
    dot_product_loss = nn.functional.mse_loss(predicted_dot_product, target_dot_product)

    # Combine the losses
    loss = cosine_weight * cosine_loss + dot_product_weight * dot_product_loss

    return loss

if __name__ == "__main__":
    t1 = torch.ones(3, 1, 2)
    t2 = torch.ones(3, 1, 2)
    t2[1, :, :] = 2
    t2[2, :, :] = -1


    