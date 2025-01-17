import torch
from price_model import PriceModel
from company_description_model import CompanyDescriptionModel
from fundamentals_model import FundamentalsModel
from company_features_model import CompanyFeaturesModel
from macro_model import MacroModel
from model.utils import HeadCombinationLayer
from torch import nn

class CompanyEmbeddingModel(nn.Module):
    def __init__(self, price_model, fundamentals_model, company_description_model, company_features_model,
                 macro_model, head_combination_model):
        super(CompanyEmbeddingModel, self).__init__()
        self.price_model = price_model
        self.fundamentals_model = fundamentals_model
        self.macro_model = macro_model
        self.company_description_model = company_description_model
        self.company_features_model = company_features_model
        self.head_combination_model = head_combination_model
         
    def forward(self, price_tensor_batch, price_key_padding_mask_batch, fundamentals_tensor_batch, fundamentals_key_padding_mask_batch,
                fundamentals_missing_features_mask_batch, fundamentals_missing_features_key_padding_mask_batch,
                company_features_tensor_batch, company_description_input_ids_batch, company_description_key_padding_mask_batch,
                macro_tensor_batch, macro_key_padding_mask_batch):
        
        print("Macro tensor shape: ", macro_tensor_batch.shape)
        print("Price tensor shape: ", price_tensor_batch.shape)
        print("Fundamentals tensor shape: ", fundamentals_tensor_batch.shape)
        print("Company features tensor shape: ", company_features_tensor_batch.shape)
        print("Company description tensor shape: ", company_description_input_ids_batch.shape)
        
        macro_cls_tensor = self.macro_model(macro_tensor_batch, layer_normalization=True, key_padding_mask=macro_key_padding_mask_batch)
        price_cls_tensor1 = self.price_model(price_tensor_batch, layer_normalization=True, key_padding_mask=price_key_padding_mask_batch)
        fundamentals_cls_tensor1 = self.fundamentals_model(fundamentals_tensor_batch, missing_features_mask=fundamentals_missing_features_mask_batch,
                                                           tgt_key_padding_mask=fundamentals_missing_features_key_padding_mask_batch, src_key_padding_mask=fundamentals_key_padding_mask_batch,
                                                            memory_key_padding_mask=fundamentals_key_padding_mask_batch)
        company_description_cls_tensor1 = self.company_description_model(company_description_input_ids_batch, company_description_key_padding_mask_batch)
        company_features_tensor1 = self.company_features_model(company_features_tensor_batch)
        company_embedding1 = self.head_combination_model(price_cls_tensor1, fundamentals_cls_tensor1, company_description_cls_tensor1, company_features_tensor1, macro_cls_tensor)

        return company_embedding1
    
if __name__ == "__main__":
    pass
    
