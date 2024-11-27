import utils
from torch import nn
from torchinfo import summary

class CompanyDescriptionModel(nn.Module):
    def __init__(self, hidden_dim, num_hidden_layers, output_dim, dropout_prob=0.1, activation_function=nn.GELU, embedding_dim = 1536):
        super(CompanyDescriptionModel, self).__init__()
        self.ffnn = utils.FFNN(embedding_dim = embedding_dim, hidden_dim=hidden_dim, num_hidden_layers=num_hidden_layers, output_dim=output_dim, dropout_prob=dropout_prob, activation_function=activation_function)
        
    def forward(self, company_description_embedding):
        result = self.ffnn(company_description_embedding)
        return result


if __name__ == "__main__":
    company_embedding_model = CompanyDescriptionModel(hidden_dim=1536, embedding_dim=1536, num_hidden_layers=2, output_dim=1536)
    summary(company_embedding_model, input_size=[(1, 1536)])