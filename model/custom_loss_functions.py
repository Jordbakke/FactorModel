import torch
from torch import nn

class MinEuclideanLoss(nn.Module):
    def __init__(self):
        super(MinEuclideanLoss, self).__init__()

    def forward(self, predicted_transaction_company, future_transaction_companies_inc_current_data):

        # Compute the squared Euclidean distance between predicted with shape (1, d) and each horizontal vector in future with shape (n, d)
        distances = torch.cdist(predicted_transaction_company, future_transaction_companies_inc_current_data).squeeze(0)
        # Return the minimum distance
        min_distance = torch.min(distances)
        return min_distance
    
if __name__ == "__main__":
    t1 = torch.ones(1,3)
    t2 = torch.tensor([[1.0,1.0,1.0], [0.0,0.0,0.0], [2.0,2.0,2.0]])
    min_euclidean_loss = MinEuclideanLoss()
    print(min_euclidean_loss(t1, t2))