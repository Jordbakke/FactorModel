import torch
import sys
sys.path.append(r"C:\repos\Deep-learning-trj")
from model import utils
from torch import nn

class MacroModel(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers,
                num_encoder_blocks=3, ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True):
        
        super().__init__()
        self.encoder= utils.Encoder(embedding_dim=embedding_dim, num_heads=num_heads,
                                    ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                    num_encoder_blocks=num_encoder_blocks, ffnn_dropout_prob=ffnn_dropout_prob,
                                    attention_dropout_prob=attention_dropout_prob,
                                    activation_function=activation_function, batch_first=batch_first)
        
    def forward(self, x):
        return self.encoder(x)
