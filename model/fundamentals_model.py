import torch
import utils
from torch import nn

class FundamentalsModel(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, activation_function=nn.GELU, ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, batch_first=True, num_encoder_blocks=3, max_seq_len=1000, prepend_cls_vector=False):
        super(FundamentalsModel, self).__init__()
        
        if prepend_cls_vector:
            self.prepend_cls_vector = utils.PrependClsVector(embedding_dim)

        self.positional_encoding = utils.PositionalEncoding(embedding_dim, max_seq_len=max_seq_len)
        self.encoder = utils.Encoder(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                                      num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                                      activation_function=activation_function, ffnn_dropout_prob=ffnn_dropout_prob,
                                                      attention_dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                                      num_encoder_blocks=num_encoder_blocks)
        
    def forward(self, x, layer_normalization=False, key_padding_mask=None):
        if hasattr(self, 'prepend_cls_vector'):
            x = self.prepend_cls_vector(x)
            if key_padding_mask is not None: #add attention mask for the newly added cls vector.
                key_padding_mask = torch.cat([torch.zeros(key_padding_mask.shape[0], 1, dtype=key_padding_mask.dtype), key_padding_mask], dim=1)
                
        positional_embedded_x = self.positional_encoding(x)
        result = self.encoder(positional_embedded_x, layer_normalization=layer_normalization,
                              key_padding_mask=key_padding_mask
                              )
        return result

if __name__ == "__main__":
    pass

    a = FundamentalsModel(embedding_dim=1000, num_heads=4, ffnn_hidden_dim=400, num_ffnn_hidden_layers=3, activation_function=nn.GELU, ffnn_dropout_prob=0.1)

    t = torch.randn(2, 10, 1000)
    print(a(t).shape)   
