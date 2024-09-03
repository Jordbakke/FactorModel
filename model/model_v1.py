import torch
import utils
from torch import nn

class ModelV1(nn.Module):

    def __init__(self, embedding_model, transformer, output_layer):
        super(ModelV1, self).__init__()
        self.embedding_model = embedding_model
        self.transformer = transformer
        self.output_layer = output_layer

    def forward(self, fundamentals, company_description, prices):
        company_embeddings = self.embedding_model(fundamentals, company_description, prices)
        current_portfolio_embeddings = None

        transformer_output = self.transformer(company_embeddings, current_portfolio_embeddings)
        return self.output_layer(transformer_output)
        
