import sys
import os
parent_dir = os.getenv("PARENT_DIR", os.getcwd())
if parent_dir:
    sys.path.append(parent_dir)
from ml_code.model import utils
from torch import nn

class FundamentalsModel(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, num_encoder_blocks,
                 activation_function=nn.GELU, ffnn_dropout_prob=0.1, attention_dropout_prob=0.1,
                 batch_first=True, max_seq_len=1000):
        
        super().__init__()

        self.encoder_cls = utils.EncoderCls(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                    num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                    activation_function=activation_function, ffnn_dropout_prob=ffnn_dropout_prob,
                                    attention_dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                    num_encoder_blocks=num_encoder_blocks, max_seq_len=max_seq_len)
    
    def forward(self, x, layer_normalization=True, key_padding_mask=None):
        return self.encoder_cls(x, layer_normalization=layer_normalization, key_padding_mask=key_padding_mask)

if __name__ == "__main__":
    embedding_dim = 800
    num_heads = 8
    ffnn_hidden_dim=800
    num_ffnn_hidden_layers = 2
    activation_function=nn.GELU
    dropout=0.1
    batch_first=True
    num_encoder_blocks=6
    fundamentals_model = FundamentalsModel(embedding_dim=embedding_dim,
                                        num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                        activation_function=activation_function, ffnn_dropout_prob=dropout,
                                        attention_dropout_prob=dropout,
                                        batch_first=batch_first, num_encoder_blocks=num_encoder_blocks
                                        )

    params = sum(p.numel() for p in fundamentals_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {params}")