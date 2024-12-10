import utils
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torchinfo import summary

class CompanyDescriptionModel(nn.Module):
    def __init__(self, bert_model="bert_large_uncased", embedding_dim = 1024):
        super(CompanyDescriptionModel, self).__init__()
        self.bert_model= AutoModel.from_pretrained(bert_model)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids, attention_masks):
        output = self.bert_model(input_ids, attention_mask=attention_masks)
        cls_vectors = output.last_hidden_state[:, 0, :]
        cls_vectors = self.layer_norm(cls_vectors)
        return cls_vectors

if __name__ == "__main__":
    company_embedding_model = CompanyDescriptionModel(hidden_dim=1536, embedding_dim=1536, num_hidden_layers=2, output_dim=1536)
    summary(company_embedding_model, input_size=[(1, 1536)])