import torch
import sys
sys.path.append(r"C:\repos\Deep-learning-trj")
from model import utils
from torch import nn
import torch.nn.functional as F
from torchinfo import summary

class FundamentalsModel(nn.Module):

    def __init__(self, embedding_dim, num_heads=6, num_decoder_layers=6,
                                          ffnn_hidden_dim=2048, dropout=0.1, activation_function=nn.GELU, batch_first=True,
                                          num_ffnn_hidden_layers=2, num_encoder_blocks=6):
        super(FundamentalsModel, self).__init__()
        self.positional_encoding = utils.PositionalEncoding(embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_encoder_blocks,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=ffnn_hidden_dim, dropout=dropout,
                                          activation=activation_function(), batch_first=batch_first)
        self.prepend_cls_vector = utils.PrependClsVector(embedding_dim)
        self.encoder = utils.Encoder(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers, ffnn_dropout_prob=dropout,
                                        attention_dropout_prob=dropout, activation_function=activation_function,
                                        batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                                        )
        
    def forward(self, fundamentals, missing_features_mask, tgt_key_padding_mask, src_key_padding_mask, memory_key_padding_mask):
        fundamentals = self.positional_encoding(fundamentals)
        missing_features_mask = self.positional_encoding(missing_features_mask)
        transformer_output = self.transformer(src=fundamentals, tgt=missing_features_mask, src_key_padding_mask=src_key_padding_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        encoder_input = self.prepend_cls_vector(transformer_output)
    
        encoder_key_padding_mask = F.pad(tgt_key_padding_mask, (0, 1))
         #add a zero at the beginning to account for the cls vector
        output = self.encoder(encoder_input, layer_normalization=True, key_padding_mask=encoder_key_padding_mask)
        return output[:, 0:1, :]

if __name__ == "__main__":
    
    model = FundamentalsModel(embedding_dim=300)
    fundamentals = torch.randn(3, 20, 300)
    fundamentals_key_padding_mask = torch.zeros(3, 20)
    fundamentals_missing_features_mask = torch.randn(3, 20, 300)
    fundamentals_missing_features_key_padding_mask = torch.zeros(3, 20)
    result = model(fundamentals = fundamentals, missing_features_mask=fundamentals_missing_features_mask,
                   tgt_key_padding_mask=fundamentals_missing_features_key_padding_mask, src_key_padding_mask=fundamentals_key_padding_mask,
                   memory_key_padding_mask=fundamentals_key_padding_mask)
    
    print(result.shape)
    # # Initialize Transformer
    # transformer = nn.Transformer(
    #     d_model=300,
    #     nhead=6,
    #     num_encoder_layers=6,
    #     num_decoder_layers=6,
    #     dim_feedforward=2048,
    #     dropout=0.1,
    #     activation=nn.GELU(),  # Correct activation function
    #     batch_first=True
    # )

    # # Dummy inputs
    # tensor = torch.randn(3, 20, 300)
    # tensor_key_padding_mask = torch.zeros(3, 20, dtype=torch.bool)
    # tensor_missing_features_mask = torch.randn(3, 20, 300)
    # tensor_missing_features_key_padding_mask = torch.zeros(3, 20, dtype=torch.bool)

    # # Forward pass
    # result = transformer(
    #     src=tensor,
    #     tgt=tensor_missing_features_mask,
    #     src_key_padding_mask=tensor_key_padding_mask,
    #     tgt_key_padding_mask=tensor_missing_features_key_padding_mask,
    #     memory_key_padding_mask=tensor_key_padding_mask
    # )
    # print(result.shape)  # Verify output
