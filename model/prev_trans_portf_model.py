
import torch
import utils
from torch import nn
from torchinfo import summary

class PreviousTransactionsPortfolioModel(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True, num_encoder_blocks=3,
                num_decoder_blocks=3):
        super(PreviousTransactionsPortfolioModel, self).__init__()

        self.transformer = utils.Transformer(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                                        num_ffnn_hidden_layers=num_ffnn_hidden_layers, ffnn_dropout_prob=ffnn_dropout_prob,
                                                        attention_dropout_prob=attention_dropout_prob, activation_function=activation_function,
                                                        batch_first=batch_first, num_encoder_blocks=num_encoder_blocks, num_decoder_blocks=num_decoder_blocks,
                                                        )
    
    def forward(self, prev_transaction_companies, portfolio_companies, is_causal):
        # Prev transactions as key_value vectors and portfolio companies as quer vectors. No need for mask
        output = self.transformer(encoder_input = prev_transaction_companies,
                                  decoder_input = portfolio_companies, is_causal=is_causal)
        return output
    
if __name__ == "__main__":

    model = PreviousTransactionsPortfolioModel(embedding_dim=1544, num_heads=2, ffnn_hidden_dim=2048,
                                                 num_ffnn_hidden_layers=3, ffnn_dropout_prob=0.1,
                                                 attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True,
                                                 num_encoder_blocks=3, num_decoder_blocks=3)
    prev_transactions = torch.ones(3, 2, 1544)
    port = torch.ones(3, 40, 1544)
    result = model(prev_transaction_companies=prev_transactions, portfolio_companies=port, is_causal=False)
    print(result.shape)
