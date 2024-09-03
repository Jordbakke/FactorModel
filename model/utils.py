import torch
import numpy as np
from torch import nn

class PrependEmbeddingVector(nn.Module):
    def __init__(self, embedding_dim):
        super(PrependEmbeddingVector, self).__init__()
        self.embedding_vector = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, x):
        batch_size = x.size(0)
        embedding_vector = self.embedding_vector.expand(batch_size, -1, -1)
        return torch.cat((embedding_vector, x), dim=1)  
    
class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=1000):
        super(PositionalEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        
        # Create positional encoding matrix using vectorized operations
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        
        PE = torch.zeros(max_seq_len, embedding_dim, requires_grad=False)
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
            
        PE = PE.unsqueeze(0)  # Add batch dimension
        self.register_buffer('PE', PE)  # Register buffer for positional encoding
    
    def forward(self, x):
        x = x + self.PE[:, :x.size(1), :]
        return x

class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_hidden_layers,
                 output_dim, activation_function=nn.GELU, dropout_prob=0.1, dropout_layer_frequency=2):
        super(FFNN, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.network = nn.ModuleList()
        self.network.append(nn.Linear(embedding_dim, hidden_dim))
        self.network.append(activation_function())
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        for i in range(num_hidden_layers - 1):
            self.network.append(nn.Linear(hidden_dim, hidden_dim))
            self.network.append(activation_function())
            if (i + 1) % dropout_layer_frequency == 0:
                self.network.append(self.dropout)
            
        self.network.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*self.network)

    def forward(self, x):
        return self.network(x) 

class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_prob=0.1, batch_first=True, force_inner_dimensions=False):
        super(MultiHeadSelfAttentionBlock, self).__init__()

        if embedding_dim % num_heads != 0:  # Check if the input dimension is divisible by the number of heads
            if not force_inner_dimensions:
                raise ValueError(f"Embedding dimension {embedding_dim} is not divisible by number of heads {num_heads}")
            new_embedding_dim = ((embedding_dim // num_heads) + 1) * num_heads  # Round up to the nearest multiple of num_heads
            self.linear_embedding_transform = nn.Linear(in_features=embedding_dim, out_features=new_embedding_dim)
            embedding_dim = new_embedding_dim

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def forward(self, x):
        if hasattr(self, 'linear_embedding_transform'):  # Increase input dimension if not divisible by num_heads
            x = self.linear_embedding_transform(x)
        
        attention_output, _ = self.multi_head_attention(query=x, key=x, value=x, need_weights=False)
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + x # Residual connection

        if attention_output.shape[-1] > 1:  # If embedding dim is just 1, normalization will set all values to 0, which is not desirable. Batch normalization in this case?
            normalized_attention_output = self.layer_norm(attention_output)
        else:
            normalized_attention_output = attention_output
        return normalized_attention_output

#Cross attention
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_prob=0.1, batch_first=True, force_inner_dimensions=False):
        super(MultiHeadAttentionBlock, self).__init__()

        if embedding_dim % num_heads != 0:  # Check if the input dimension is divisible by the number of heads
            if not force_inner_dimensions:
                raise ValueError(f"Embedding dimension {embedding_dim} is not divisible by number of heads {num_heads}")
            new_embedding_dim = ((embedding_dim // num_heads) + 1) * num_heads  # Round up to the nearest multiple of num_heads
            self.linear_embedding_transform = nn.Linear(in_features=embedding_dim, out_features=new_embedding_dim)
            embedding_dim = new_embedding_dim

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.multi_head_attention = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
    
    def forward(self, query_vector, key_value_vector):
        if hasattr(self, 'linear_embedding_transform'):  # Increase input dimension if not divisible by num_heads
            query_vector = self.linear_embedding_transform(query_vector)
            key_value_vector = self.linear_embedding_transform(key_value_vector)

        attention_output, _ = self.multi_head_attention(query=query_vector, key=key_value_vector, value=key_value_vector, need_weights=False)
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + query_vector
         # Residual connection
        if attention_output.shape[-1] > 1:  # If embedding dim is just 1, normalization will set all values to 0, which is not desirable. Batch normalization in this case?
            normalized_attention_output = self.layer_norm(attention_output)
        else:
            normalized_attention_output = attention_output
        return normalized_attention_output


class EncoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers = 2, ffnn_dropout_prob=0.1, attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True, force_inner_dimensions=False):
        super(EncoderBlock, self).__init__()

        
        self.multi_head_attention = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                                dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                                                force_inner_dimensions=force_inner_dimensions)

        self.embedding_dim = self.multi_head_attention.embedding_dim
        self.ffnn = FFNN(embedding_dim=embedding_dim, hidden_dim=ffnn_hidden_dim, num_hidden_layers=num_ffnn_hidden_layers, output_dim=embedding_dim, 
                         dropout_prob=ffnn_dropout_prob, activation_function=activation_function)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, x):
        attention_output = self.multi_head_attention(x)
        ffnn_output = self.ffnn(attention_output)
        ffnn_output = ffnn_output + attention_output  # Residual connection
        if ffnn_output.shape[-1] > 1:
            normalized_ffnn_output = self.layer_norm(ffnn_output)
        else:
            normalized_ffnn_output = ffnn_output
        return normalized_ffnn_output

class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True, num_encoder_blocks=3, force_inner_dimensions=False):
        super(Encoder, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        
        self.encoder_blocks.append(
            EncoderBlock(embedding_dim=embedding_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers, 
                             ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob, 
                             activation_function=activation_function, batch_first=batch_first, force_inner_dimensions=force_inner_dimensions)
        )

        self.embedding_dim = self.encoder_blocks[0].embedding_dim

        for i in range(num_encoder_blocks - 1):
            self.encoder_blocks.append(
                EncoderBlock(embedding_dim=embedding_dim, num_heads = num_heads, ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers, 
                                 ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob, 
                                 activation_function=activation_function, batch_first=batch_first, force_inner_dimensions=False)
            ) #num_heads for the subsequent block is a multiple of embedding_dim, so force_inner_dimensions=False

    def forward(self, x):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, ffnn_dropout_prob=0.1, attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True, force_inner_dimensions=False):
        super(DecoderBlock, self).__init__()
        
        self.multi_head_self_attention = MultiHeadSelfAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                            dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                                            force_inner_dimensions=force_inner_dimensions)
        
        self.embedding_dim = self.multi_head_self_attention.embedding_dim
        self.multi_head_cross_attention = MultiHeadAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                            dropout_prob=attention_dropout_prob, batch_first=batch_first,
                                                            force_inner_dimensions=False)
        
        self.ffnn = FFNN(embedding_dim=embedding_dim, hidden_dim=ffnn_hidden_dim, num_hidden_layers=num_ffnn_hidden_layers, output_dim=embedding_dim, 
                         dropout_prob=ffnn_dropout_prob, activation_function=activation_function)
        
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, encoder_output, decoder_input):
        self_attention_output = self.multi_head_self_attention(decoder_input)
        attention_output = self.multi_head_cross_attention(self_attention_output, encoder_output)
        ffnn_output = self.ffnn(attention_output)
        return self.layer_norm(ffnn_output + attention_output)  # Residual connection

class Decoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers=1,
                ffnn_dropout_prob=0.1, attention_dropout_prob=0.1, activation_function=nn.GELU,
                batch_first=True, num_decoder_blocks=3, force_inner_dimensions=False):
        super(Decoder, self).__init__()
        self.decoder_network = nn.ModuleList()
        self.decoder_network.append(DecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                                ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob,
                                                activation_function=activation_function, batch_first=batch_first,
                                                force_inner_dimensions=force_inner_dimensions))
        
        self.embedding_dim = self.decoder_network[0].embedding_dim

        for i in range(num_decoder_blocks - 1):
            self.decoder_network.append(DecoderBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                    ffnn_hidden_dim=ffnn_hidden_dim, num_ffnn_hidden_layers=num_ffnn_hidden_layers,
                                                    ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob,
                                                    activation_function=activation_function, batch_first=batch_first,
                                                    force_inner_dimensions=False))
            
    def forward(self, encoder_output, decoder_input):
        for decoder_block in self.decoder_network:
            decoder_input = decoder_block(encoder_output, decoder_input)
        return decoder_input

class Transformer(nn.Module):
        
        def __init__(self, encoder, decoder):
            super(Transformer, self).__init__()
            
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, encoder_input, decoder_input):
            encoder_output = self.encoder(encoder_input)
            decoder_output = self.decoder(encoder_output, decoder_input)
            
            return encoder_output, decoder_output

class DimensionAligner(nn.Module):
    """
    Class that aligns the dimensions of the input tensors by applying a linear transformation to each tensor and stacks them along the first dimension.
    """
    def __init__(self, *args, final_dim=None):
        super(DimensionAligner, self).__init__()

        input_dims = [arg.shape[-1] for arg in args]
        if final_dim is None:
            final_dim = max(input_dims)
        
        self.final_dim = final_dim
        
        self.linear_projection_layers = nn.ModuleList()
        for arg in args:
            self.linear_projection_layers.append(nn.Linear(arg.shape[-1], final_dim))
        
    def forward(self, *args, stacking_dim=1):
        aligned_tensors = []
        for i, arg in enumerate(args):
            
            aligned_tensors.append(self.linear_projection_layers[i](arg))
        stacked_tensors = torch.stack(aligned_tensors, dim=stacking_dim)

        return stacked_tensors.squeeze()
                                        
class MultiModalCombiner(nn.Module):
    """
    arg should be of shape (batch_size, 1, embedding_dim)
    """ 
    def __init__(self, *args, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, num_encoder_blocks,  ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True,
                force_inner_dimensions=True, final_dim=None):
        
        super(MultiModalCombiner, self).__init__()
        self.dimension_aligner = DimensionAligner(*args, final_dim=final_dim)
        self.prepend_embedding_vector = PrependEmbeddingVector(self.dimension_aligner.final_dim)
        self.encoder = Encoder(embedding_dim=self.dimension_aligner.final_dim, num_heads=num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                      num_ffnn_hidden_layers=num_ffnn_hidden_layers, activation_function=activation_function, ffnn_dropout_prob=ffnn_dropout_prob,
                                      attention_dropout_prob=attention_dropout_prob, batch_first=batch_first, num_encoder_blocks=num_encoder_blocks,
                                      force_inner_dimensions=force_inner_dimensions)
        

    def forward(self, *args): 
        
        aligned_tensor = self.dimension_aligner(*args)
       
        aligned_tensors_with_prepended_embedding_vector = self.prepend_embedding_vector(aligned_tensor)
        
        transformer_output = self.encoder(aligned_tensors_with_prepended_embedding_vector)
        return transformer_output[:, :1, :]
    
#Test the implementation
t1 = torch.ones(3, 1, 2)
t2 = torch.ones(3, 1, 4)
a = MultiModalCombiner(t1, t2, num_heads=2, ffnn_hidden_dim=4, num_ffnn_hidden_layers=3, num_encoder_blocks=3, ffnn_dropout_prob=0.1, attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True, force_inner_dimensions=True, final_dim=4)

# print(a(t1, t2).shape)

print(a(t1, t2))
