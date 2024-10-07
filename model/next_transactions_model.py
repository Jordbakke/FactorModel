import torch
import utils
from torch import nn
from torchinfo import summary

class NextTransactionsModel(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True,
                num_encoder_blocks=3):
        super(NextTransactionsModel, self).__init__()

        self.prepend_cls_vector = utils.PrependClsVector(embedding_dim)
        self.encoder = utils.Encoder(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers, ffnn_dropout_prob=ffnn_dropout_prob,
                                        attention_dropout_prob=attention_dropout_prob, activation_function=activation_function,
                                        batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                                        )
        
        self.ffnn = utils.FFNN(embedding_dim=embedding_dim, hidden_dim=ffnn_hidden_dim, output_dim=embedding_dim,
                               num_hidden_layers=num_ffnn_hidden_layers, dropout_prob=ffnn_dropout_prob)

    def forward(self, x, layer_normalization=True, key_padding_mask=None):
        # x is the output of the prev_trans_portf_model
        x = self.prepend_cls_vector(x)
        if key_padding_mask is not None: #add attention mask for the newly added cls vector.
            key_padding_mask = torch.cat([torch.zeros(key_padding_mask.shape[0], 1, dtype=key_padding_mask.dtype), key_padding_mask], dim=1)
        predicted_transaction = self.encoder(x, layer_normalization=layer_normalization,
                                             key_padding_mask=key_padding_mask)[:, 0:1, :]
        predicted_transaction = self.ffnn(predicted_transaction)
        return predicted_transaction

if __name__ == "__main__":
    model = NextTransactionsModel(embedding_dim=1566, num_heads=2, ffnn_hidden_dim=2048, num_ffnn_hidden_layers=3, num_encoder_blocks=3)

    t = torch.ones(3, 300, 1566)
    result = model(t)
    print(result.shape)
    