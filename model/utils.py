import torch
import numpy as np
from torch import nn

class PrependClsVector(nn.Module):
    def __init__(self, embedding_dim):
        super(PrependClsVector, self).__init__()
        self.cls_vector = nn.Parameter(torch.randn(1, embedding_dim))

    def forward(self, x):
        if len(x.shape) != 3:
            raise ValueError("Expected 3D tensor.")
        batch_size = x.size(0)
        cls_vector = self.cls_vector.expand(batch_size, -1, -1)
        return torch.cat([cls_vector, x], dim=1)
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_seq_len=1000):
        super(PositionalEncoding, self).__init__()
        
        if embedding_dim % 2 != 0:
            self.original_embedding_dim = embedding_dim
            embedding_dim = embedding_dim + 1
            
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
        if hasattr(self, "original_embedding_dim"):
            x = x + self.PE[:, :x.size(1), :-1]
        else:
            x = x + self.PE[:, :x.size(1), :]
        return x
class FFNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_hidden_layers,
                 output_dim, activation_function=nn.GELU, dropout_prob=0.1, dropout_layer_frequency=2):
        super(FFNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(activation_function())
        
        # Add hidden layers
        for i in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(activation_function())
            if (i + 1) % dropout_layer_frequency == 0:
                layers.append(nn.Dropout(dropout_prob))
                
        # Final output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        # Use Sequential to wrap the layers
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, x):
        return self.network(x)

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_prob=0.1,batch_first=True):
        super(MultiHeadAttentionBlock, self).__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension {embedding_dim} is not divisible by number of heads {num_heads}")

        self.num_heads = num_heads
        self.multi_head_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads, dropout=dropout_prob, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(embedding_dim)
    
    def forward(self, query, key, value, layer_normalization=True, key_padding_mask=None, apply_causal_mask=False):
        # Apply causal mask if required
        if apply_causal_mask:
            seq_len = query.size(1)  # Ensure the mask matches the sequence length
            causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(query.device)  # Match the device
        else:
            causal_mask = None
        
        # Compute multi-head attention
        attention_output, _ = self.multi_head_attention(query=query, key=key, value=value,
                                                        key_padding_mask=key_padding_mask,
                                                        attn_mask=causal_mask, need_weights=False)
        attention_output = self.dropout(attention_output)
        attention_output = attention_output + query  # Residual connection
        if layer_normalization:
            attention_output = self.layer_norm(attention_output)
        
        return attention_output
class EncoderBlock(nn.Module):

    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, ffnn_dropout_prob=0.1,
                 attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True):
        super(EncoderBlock, self).__init__()
        self.multi_head_attention_block = MultiHeadAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,
                                                                dropout_prob=attention_dropout_prob, batch_first=batch_first
                                                                )
        #embedding dim is the same as output embedding dim
        self.ffnn = FFNN(embedding_dim=embedding_dim, hidden_dim=ffnn_hidden_dim, num_hidden_layers=num_ffnn_hidden_layers,
                         output_dim=embedding_dim, dropout_prob=ffnn_dropout_prob, activation_function=activation_function)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x, layer_normalization=True, key_padding_mask=None):
        attention_output= self.multi_head_attention_block(query=x, key=x, value=x,
                                                          layer_normalization=layer_normalization, key_padding_mask=key_padding_mask,
                                                          apply_causal_mask=False)
        ffnn_output = self.ffnn(attention_output)
        ffnn_output = ffnn_output + attention_output  # Residual connection
        if ffnn_output.shape[-1] > 1:
            normalized_ffnn_output = self.layer_norm(ffnn_output)
        else:
            normalized_ffnn_output = ffnn_output
        return normalized_ffnn_output

class Encoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffnn_hidden_dim, num_ffnn_hidden_layers, num_encoder_blocks=3, ffnn_dropout_prob=0.1,
                attention_dropout_prob=0.1, activation_function=nn.GELU, batch_first=True):
        super(Encoder, self).__init__()
        
        self.encoder_blocks = nn.ModuleList()
        for i in range(num_encoder_blocks):
            self.encoder_blocks.append(
                EncoderBlock(embedding_dim=embedding_dim, num_heads = num_heads, ffnn_hidden_dim=ffnn_hidden_dim,
                                 num_ffnn_hidden_layers=num_ffnn_hidden_layers, 
                                 ffnn_dropout_prob=ffnn_dropout_prob, attention_dropout_prob=attention_dropout_prob, 
                                 activation_function=activation_function, batch_first=batch_first)
            )

    def forward(self, x, layer_normalization=True, key_padding_mask=None):
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, layer_normalization=layer_normalization,
                              key_padding_mask=key_padding_mask)
        return x
    
class HeadCombinationLayer(nn.Module):
    
    def __init__(self, input_dims: list, num_hidden_layers: int, final_dim: int, num_heads: int, activation_function=nn.GELU, dropout_prob=0.1,
                 dropout_layer_frequency=2, batch_first=True):
        super(HeadCombinationLayer, self).__init__()
        self.linear_projections = nn.ModuleList()
        self.num_vectors = len(input_dims)
        for input_dim in input_dims:
            self.linear_projections.append(FFNN(embedding_dim=input_dim, hidden_dim=final_dim, num_hidden_layers=num_hidden_layers, output_dim=final_dim,
                                                dropout_prob=dropout_prob, activation_function=activation_function, dropout_layer_frequency=dropout_layer_frequency))
        
        self.prepend_cls_vector = PrependClsVector(embedding_dim=final_dim)
        self.multi_head_attention = MultiHeadAttentionBlock(embedding_dim=final_dim, num_heads=num_heads,
                                                                dropout_prob=dropout_prob, batch_first=batch_first)
        self.final_dim = final_dim

    def forward(self, *args, layer_normalization=True, key_padding_mask=None):
        if len(args) != self.num_vectors:
            raise ValueError(f"Expected {self.num_vectors} input vectors, but got {len(args)}")
        aligned_tensors = []
        for i, arg in enumerate(args):
            aligned_tensors.append(self.linear_projections[i](arg))

        stacked_tensors = torch.stack(aligned_tensors, dim=1).squeeze()
        if stacked_tensors.dim() == 2: #happens when num transactions/companies is 1
            stacked_tensors = stacked_tensors.unsqueeze(0)
        stacked_tensors = self.prepend_cls_vector(stacked_tensors)
        stacked_tensors = self.multi_head_attention(query=stacked_tensors, key=stacked_tensors, value=stacked_tensors,
                                                    layer_normalization=layer_normalization, key_padding_mask=key_padding_mask,
                                                    apply_causal_mask=False)
        result = stacked_tensors[:, 0:1, :]
        return result #First company is first horizontal vector in the 2D tensor, second company is the second vector etc

if __name__ == "__main__":
    #Test the implementation
    t1 = torch.ones(3, 1, 2)
    t2 = torch.zeros(3, 1, 4)
    a = HeadCombinationLayer(input_dims=[2, 4], final_dim=3, num_heads=3, num_hidden_layers=2)
    result = a(t1, t2)
    print(result)
    print(result.shape)
