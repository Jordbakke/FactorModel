from model.utils import FFNN
from torch import nn

class CompanyFeaturesModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, output_dim, num_hidden_layers=4, activation_function=nn.GELU, dropout_prob=0.1, dropout_layer_frequency=2):
        super().__init__()
        self.ffnn = FFNN(embedding_dim=embedding_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers,
                 output_dim=output_dim, activation_function=activation_function, dropout_prob=dropout_prob,
                 dropout_layer_frequency=dropout_layer_frequency)
    
    def forward(self, x):
        return self.ffnn(x)
    
