import torch
import pandas as pd

def infer(predicted_transactions, all_company_embeddings_df: pd.DataFrame):
    """
    Infer the company embeddings from the predicted transactions and the known company embeddings.
    :param predicted_transactions: The predicted transactions tensor. Shape: (batch_size, num_transactions, embedding_dim)
    :param all_company_embeddings: pandas dataframe with columns ['index', 'ticker', 'embedding']
    :return: The inferred company embeddings tensor. Shape: (batch_size, num_companies, embedding_dim)
    """
    # Compute the pairwise Euclidean distances between the predicted transactions and the known company embeddings
    all_company_embeddings = torch.tensor(all_company_embeddings_df["embedding"].values)

    distances = torch.cdist(predicted_transactions, all_company_embeddings)  # Shape: (batch_size, num_transactions, num_companies)
    
    # Compute the minimum distances and the corresponding indices
    min_distances, min_indices = torch.min(distances, dim=1)  # Along the num_transactions axis
    print(min_indices)

    #Use min_indices to get the corresponding company names
    return 1




    