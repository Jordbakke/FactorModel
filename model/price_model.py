import torch
import utils
from torch import nn
from torchinfo import summary
from torch.utils.data import Dataset, DataLoader
    
class PriceModel(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, num_encoder_blocks,
                 activation_function=nn.GELU, ffnn_dropout_prob=0.1, attention_dropout_prob=0.1,
                 batch_first=True, max_seq_len=1000,
                 prepend_embedding_vector=False):
        
        super().__init__()
        if prepend_embedding_vector:
            self.prepend_embedding_vector = utils.PrependEmbeddingVector(embedding_dim)

        self.positional_encoding = utils.PositionalEncoding(embedding_dim, max_seq_len=max_seq_len)
        self.encoder = utils.Encoder(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                    num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                    activation_function=activation_function, ffnn_dropout_prob=ffnn_dropout_prob,
                                    attention_dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                    num_encoder_blocks=num_encoder_blocks,
                                    )
        
    def forward(self, x):
        if hasattr(self, 'prepend_embedding_vector'):
            x = self.prepend_embedding_vector(x)
        positional_embedded_x = self.positional_encoding(x)
        result = self.encoder(positional_embedded_x)
        return result

if __name__ == "__main__":

    embedding_dim = 2
    num_heads = 2
    ffnn_hidden_dim = 2
    num_ffnn_hidden_layers = 3
    activation_function = nn.GELU
    ffnn_dropout_prob = 0.1
    attention_dropout_prob = 0.1
    num_encoder_blocks = 3
    batch_first = True
    max_seq_len = 1000
    prepend_embedding_vector = True

    price_batch = torch.ones(2, 10, 2)
    model = PriceModel(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function,
                    ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob, batch_first=batch_first,
                    num_encoder_blocks=num_encoder_blocks, max_seq_len=max_seq_len, prepend_embedding_vector=prepend_embedding_vector)

    result = model(price_batch)
    print(result.shape)
    #summary(model, input_size=(2, 10, 2))

