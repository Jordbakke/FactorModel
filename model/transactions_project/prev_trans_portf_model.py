
import torch
from torch import nn
from torchinfo import summary

class PreviousTransactionsPortfolioModel(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim,
                 num_encoder_layers=6, num_decoder_layers=6, dropout_prob=0.1,
                 activation_function="gelu", batch_first=True):
        super(PreviousTransactionsPortfolioModel, self).__init__()

        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=num_heads, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=ffnn_hidden_dim,
                                          dropout=dropout_prob,
                                          activation=activation_function, batch_first=batch_first)
        self.output_dim = embedding_dim
    
    def forward(self, encoder_input, decoder_input,
                src_key_padding_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):

        # Prev transactions as key_value vectors and portfolio companies as quer vectors. No need for mask
        output = self.transformer(src=encoder_input, tgt=decoder_input, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,
                                  )
        
        return output

if __name__ == "__main__":

    model = PreviousTransactionsPortfolioModel(embedding_dim=1544, num_heads=2, ffnn_hidden_dim=2048, num_enoder_layers=3, num_decoder_layers=3)
    prev_transactions = torch.ones(3, 3, 1544)
    port = torch.ones(3, 40, 1544)
    result = model(encoder_input=prev_transactions, decoder_input=port)
    print(result.shape)
