from torch import nn


class EmbeddingModel(nn.Module):
    
    def __init__(self, fundamentals_model, company_description_model, price_model, head_combination_model):
        super(EmbeddingModel, self).__init__()
        self.fundamentals_model = fundamentals_model
        self.company_description_model = company_description_model
        self.price_model = price_model
        self.head_combination_model = head_combination_model
    
    def forward(self, fundamentals, company_description, prices):
        fundamentals_output = self.fundamentals_model(fundamentals)
        company_description_output = self.company_description_model(company_description)
        price_output = self.price_model(prices)
        
        return self.head_combination_model(fundamentals_output, company_description_output, price_output)