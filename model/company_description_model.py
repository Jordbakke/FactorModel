import utils
from torch import nn


class CompanyDescriptionModel(nn.Module):
    def __init__(self, hidden_dim, num_hidden_layers, output_dim, dropout_prob, activation_function, embedding_dim = 1000):
        super(CompanyDescriptionModel, self).__init__()
        self.ffnn = utils.FFNN(embedding_dim = embedding_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, output_dim=output_dim, dropout_prob=dropout_prob, activation_function=activation_function)
        
    def forward(self, company_description_embedding):
        return self.ffnn(company_description_embedding)


