import torch
import sys
sys.path.append(r"C:\repos\Deep-learning-trj")
from model import utils
from torch import nn
from torchinfo import summary

class FundamentalsModel(nn.Module):

    def __init__(self, embedding_dim, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                                          dim_feedforward=2048, dropout=0.1, activation=nn.GELU, batch_first=True,
                                          num_ffnn_hidden_layers=1, num_encoder_blocks=6):
        super(FundamentalsModel, self).__init__()
        self.positional_encoding = utils.PositionalEncoding(embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout,
                                          activation=activation, batch_first=batch_first)
        self.prepend_cls_vector = utils.PrependClsVector(embedding_dim)
        self.encoder = utils.Encoder(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=dim_feedforward,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers, ffnn_dropout_prob=dropout,
                                        attention_dropout_prob=dropout, activation_function=activation,
                                        batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                                        )
        
    def forward(self, fundamentals, missing_features_mask, tgt_mask, src_mask, memory_mask):
        fundamentals = self.positional_encoding(fundamentals)
        missing_features_mask = self.positional_encoding(missing_features_mask)
        transformer_output = self.transformer(fundamentals, missing_features_mask, tgt_mask=tgt_mask, src_mask=src_mask, memory_mask=memory_mask)
        
        encoder_input = self.prepend_cls_vector(transformer_output)
        
        encoder_key_padding_mask = torch.cat([torch.zeros(1, tgt_mask.shape[1]), tgt_mask], dim=1)
        output = self.encoder(encoder_input, layer_normalization=True, key_padding_mask=encoder_key_padding_mask)
        return output[:, 0:1, :]

if __name__ == "__main__":
    
    model = FundamentalsModel(embedding_dim=320)
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(num_parameters)