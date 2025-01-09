import sys
import os
import pandas as pd
sys.path.append(os.path.abspath("C:/repos/Deep-learning-trj"))
from data.code.llm_utils import HuggingFaceModel
from data.code import data_utils
from torch import nn
from sentence_transformers import SentenceTransformer, models

class CompanyDescriptionModel(nn.Module):
    def __init__(self, transformer_model="bert-large-uncased", embedding_dim=1024):
        super(CompanyDescriptionModel, self).__init__()
        
        pooling_layer = models.Pooling(
        word_embedding_dimension=embedding_dim,
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False
        )
        self.sbert_model = SentenceTransformer(modules=[models.Transformer(transformer_model, max_seq_length=512), pooling_layer])

    def forward(self, input_ids, attention_mask):
        return self.sbert_model({"input_ids": input_ids, "attention_mask": attention_mask})["sentence_embedding"].unsqueeze(1)

if __name__ == "__main__":

    data_utils.open_in_excel(pd.read_parquet(r"C:\repos\Deep-learning-trj\data\processed_data_storage\company_descriptions\company_descriptions.parquet"))






