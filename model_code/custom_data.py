import torch
import sys
import os
parent_dir = os.getenv("PARENT_DIR", os.getcwd())
if parent_dir:
    sys.path.append(parent_dir)
from ml_code.data_processing.data_utils import load_tensor
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, mapping_df, price_tensor_path, price_key_padding_mask_path, fundamentals_tensor_path, fundamentals_key_padding_mask_path,
                 fundamentals_missing_features_mask_path, fundamentals_missing_features_key_padding_mask_path, company_features_tensor_path,
                 company_description_input_ids_tensor_path, company_description_key_padding_mask_path, macro_tensor_path, macro_key_padding_mask_path,
                 target_tensor_path):
        
        for path in [price_tensor_path, price_key_padding_mask_path, fundamentals_tensor_path, fundamentals_key_padding_mask_path,
                     fundamentals_missing_features_mask_path, fundamentals_missing_features_key_padding_mask_path, company_features_tensor_path,
                     company_description_input_ids_tensor_path, company_description_key_padding_mask_path, macro_tensor_path, macro_key_padding_mask_path,
                     target_tensor_path]:
            if path is None:
                raise ValueError(f"Path is None")
        
        self.mapping_df = mapping_df
        self.price_tensor = load_tensor(price_tensor_path)
        self.price_key_padding_mask = load_tensor(price_key_padding_mask_path)
        self.fundamentals_tensor = load_tensor(fundamentals_tensor_path)
        self.fundamentals_key_padding_mask = load_tensor(fundamentals_key_padding_mask_path)
        self.fundamentals_missing_features_mask = load_tensor(fundamentals_missing_features_mask_path)
        self.fundamentals_missing_features_key_padding_mask = load_tensor(fundamentals_missing_features_key_padding_mask_path)
        self.company_features_tensor = load_tensor(company_features_tensor_path)
        self.company_description_input_ids_tensor = load_tensor(company_description_input_ids_tensor_path)
        self.company_description_key_padding_mask = load_tensor(company_description_key_padding_mask_path)
        self.macro_tensor = load_tensor(macro_tensor_path)
        self.macro_key_padding_mask = load_tensor(macro_key_padding_mask_path)
        self.target_tensor = load_tensor(target_tensor_path)

    def __len__(self):
        return len(self.mapping_df)

    def __getitem__(self, idx):

        selected_row = self.mapping_df.iloc[idx]

        #company1
        price_tensor_index1 = selected_row["prices_ticker_region1"]
        price_tensor1 = self.price_tensor[price_tensor_index1, :, :].unsqueeze(0)
        price_key_padding_mask1 = self.price_key_padding_mask[price_tensor_index1, :].unsqueeze(0)

        fundamentals_tensor_index1 = selected_row["fundamentals_ticker_region1"]
        fundamentals_tensor1 = self.fundamentals_tensor[fundamentals_tensor_index1, :, :].unsqueeze(0)
        fundamentals_key_padding_mask1 = self.fundamentals_key_padding_mask[fundamentals_tensor_index1, :].unsqueeze(0)
        fundamentals_missing_features_mask1 = self.fundamentals_missing_features_mask[fundamentals_tensor_index1, :].unsqueeze(0)
        fundamentals_missing_features_key_padding_mask1 = self.fundamentals_missing_features_key_padding_mask[fundamentals_tensor_index1, :].unsqueeze(0)

        company_features_tensor_index1 = selected_row["company_features_ticker_region1"]
        company_features_tensor1 = self.company_features_tensor[company_features_tensor_index1, :, :].unsqueeze(0)

        company_description_index1 = selected_row["company_description_ticker_region1"]
        company_description_input_ids1 = self.company_description_input_ids_tensor[company_description_index1, :].unsqueeze(0)
        company_description_key_padding_mask1 = self.company_description_key_padding_mask[company_description_index1, :].unsqueeze(0)

        #company2
        price_tensor_index2 = selected_row["prices_ticker_region2"]
        price_tensor2 = self.price_tensor[price_tensor_index2, :, :].unsqueeze(0)
        price_key_padding_mask2 = self.price_key_padding_mask[price_tensor_index2, :].unsqueeze(0)

        fundamentals_tensor_index2 = selected_row["fundamentals_ticker_region2"]
        fundamentals_tensor2 = self.fundamentals_tensor[fundamentals_tensor_index2, :, :].unsqueeze(0)
        fundamentals_key_padding_mask2 = self.fundamentals_key_padding_mask[fundamentals_tensor_index2, :].unsqueeze(0)
        fundamentals_missing_features_mask2 = self.fundamentals_missing_features_mask[fundamentals_tensor_index2, :].unsqueeze(0)
        fundamentals_missing_features_key_padding_mask2 = self.fundamentals_missing_features_key_padding_mask[fundamentals_tensor_index2, :].unsqueeze(0)

        company_features_tensor_index2 = selected_row["company_features_ticker_region2"]
        company_features_tensor2 = self.company_features_tensor[company_features_tensor_index2, :, :].unsqueeze(0)

        company_description_index2 = selected_row["company_description_ticker_region2"]
        company_description_input_ids2 = self.company_description_input_ids_tensor[company_description_index2, :].unsqueeze(0)
        company_description_key_padding_mask2 = self.company_description_key_padding_mask[company_description_index2, :].unsqueeze(0)

        #Common tensors
        macro_tensor_index = selected_row["macro"]
        macro_tensor = self.macro_tensor[macro_tensor_index, :, :].unsqueeze(0)
        macro_key_padding_mask = self.macro_key_padding_mask[macro_tensor_index, :].unsqueeze(0)

        target = self.target_tensor[idx].unsqueeze(0)

        tensor_dict = {
            "price_tensor1": price_tensor1,
            "price_key_padding_mask1": price_key_padding_mask1,
            "fundamentals_tensor1": fundamentals_tensor1,
            "fundamentals_key_padding_mask": fundamentals_key_padding_mask1,
            "fundamentals_missing_features_mask": fundamentals_missing_features_mask1,
            "fundamentals_missing_features_key_padding_mask": fundamentals_missing_features_key_padding_mask1,
            "company_features_tensor1": company_features_tensor1,
            "company_description_input_ids1": company_description_input_ids1,
            "company_description_key_padding_mask1": company_description_key_padding_mask1,
            "price_tensor2": price_tensor2,
            "price_key_padding_mask2": price_key_padding_mask2,
            "fundamentals_tensor2": fundamentals_tensor2,
            "fundamentals_key_padding_mask2": fundamentals_key_padding_mask2,
            "fundamentals_missing_features_mask2": fundamentals_missing_features_mask2,
            "fundamentals_missing_features_key_padding_mask2": fundamentals_missing_features_key_padding_mask2,
            "company_features_tensor2": company_features_tensor2,
            "company_description_input_ids2": company_description_input_ids2,
            "company_description_key_padding_mask2": company_description_key_padding_mask2,
            "macro_tensor": macro_tensor,
            "macro_key_padding_mask": macro_key_padding_mask,
            "target": target
        }

        return tensor_dict
    
    def custom_collate_fn(self, batch):
        
        price_tensor1_batch = []
        price_key_padding_mask1_batch = []
        fundamentals_tensor1_batch = []
        fundamentals_key_padding_mask1_batch = []
        fundamentals_missing_features_mask1_batch = []
        fundamentals_missing_features_key_padding_mask1_batch = []
        company_features_tensor1_batch = []
        company_description_input_ids1_batch = []
        company_description_key_padding_mask1_batch = []

        price_tensor2_batch = []
        price_key_padding_mask2_batch = []
        fundamentals_tensor2_batch = []
        fundamentals_key_padding_mask2_batch = []
        fundamentals_missing_features_mask2_batch = []
        fundamentals_missing_features_key_padding_mask2_batch = []
        company_features_tensor2_batch = []
        company_description_input_ids2_batch = []
        company_description_key_padding_mask2_batch = []

        macro_tensor_batch = []
        macro_key_padding_mask_batch = []
        target_batch = []

        for sample in batch:
            price_tensor1_batch.append(sample["price_tensor1"])
            price_key_padding_mask1_batch.append(sample["price_key_padding_mask1"])
            fundamentals_tensor1_batch.append(sample["fundamentals_tensor1"])
            fundamentals_key_padding_mask1_batch.append(sample["fundamentals_key_padding_mask"])
            fundamentals_missing_features_mask1_batch.append(sample["fundamentals_missing_features_mask"])
            fundamentals_missing_features_key_padding_mask1_batch.append(sample["fundamentals_missing_features_key_padding_mask"])
            company_features_tensor1_batch.append(sample["company_features_tensor1"])
            company_description_input_ids1_batch.append(sample["company_description_input_ids1"])
            company_description_key_padding_mask1_batch.append(sample["company_description_key_padding_mask1"])

            price_tensor2_batch.append(sample["price_tensor2"])
            price_key_padding_mask2_batch.append(sample["price_key_padding_mask2"])
            fundamentals_tensor2_batch.append(sample["fundamentals_tensor2"])
            fundamentals_key_padding_mask2_batch.append(sample["fundamentals_key_padding_mask2"])
            fundamentals_missing_features_mask2_batch.append(sample["fundamentals_missing_features_mask2"])
            fundamentals_missing_features_key_padding_mask2_batch.append(sample["fundamentals_missing_features_key_padding_mask2"])
            company_features_tensor2_batch.append(sample["company_features_tensor2"])
            company_description_input_ids2_batch.append(sample["company_description_input_ids2"])
            company_description_key_padding_mask2_batch.append(sample["company_description_key_padding_mask2"])

            macro_tensor_batch.append(sample["macro_tensor"])
            macro_key_padding_mask_batch.append(sample["macro_key_padding_mask"])
            target_batch.append(sample["target"])
            
        price_tensor1_batch = torch.cat(price_tensor1_batch)
        price_key_padding_mask1_batch = torch.cat(price_key_padding_mask1_batch)
        fundamentals_tensor1_batch = torch.cat(fundamentals_tensor1_batch)
        fundamentals_key_padding_mask1_batch = torch.cat(fundamentals_key_padding_mask1_batch)
        fundamentals_missing_features_mask1_batch = torch.cat(fundamentals_missing_features_mask1_batch)
        fundamentals_missing_features_key_padding_mask1_batch = torch.cat(fundamentals_missing_features_key_padding_mask1_batch)
        company_features_tensor1_batch = torch.cat(company_features_tensor1_batch)
        company_description_input_ids1_batch = torch.cat(company_description_input_ids1_batch)
        company_description_key_padding_mask1_batch = torch.cat(company_description_key_padding_mask1_batch)

        price_tensor2_batch = torch.cat(price_tensor2_batch)
        price_key_padding_mask2_batch = torch.cat(price_key_padding_mask2_batch)
        fundamentals_tensor2_batch = torch.cat(fundamentals_tensor2_batch)
        fundamentals_key_padding_mask2_batch = torch.cat(fundamentals_key_padding_mask2_batch)
        fundamentals_missing_features_mask2_batch = torch.cat(fundamentals_missing_features_mask2_batch)
        fundamentals_missing_features_key_padding_mask2_batch = torch.cat(fundamentals_missing_features_key_padding_mask2_batch)
        company_features_tensor2_batch = torch.cat(company_features_tensor2_batch)
        company_description_input_ids2_batch = torch.cat(company_description_input_ids2_batch)
        company_description_key_padding_mask2_batch = torch.cat(company_description_key_padding_mask2_batch)

        macro_tensor_batch = torch.cat(macro_tensor_batch)
        macro_key_padding_mask_batch = torch.cat(macro_key_padding_mask_batch)
        target_batch = torch.cat(target_batch)

        batch_dict = {"input": {"input1": {
            "price_tensor_batch": price_tensor1_batch,
            "price_key_padding_mask_batch": price_key_padding_mask1_batch,
            "fundamentals_tensor_batch": fundamentals_tensor1_batch,
            "fundamentals_key_padding_mask_batch": fundamentals_key_padding_mask1_batch,
            "fundamentals_missing_features_mask_batch": fundamentals_missing_features_mask1_batch,
            "fundamentals_missing_features_key_padding_mask_batch": fundamentals_missing_features_key_padding_mask1_batch,
            "company_features_tensor_batch": company_features_tensor1_batch,
            "company_description_input_ids_batch": company_description_input_ids1_batch,
            "company_description_key_padding_mask_batch": company_description_key_padding_mask1_batch,
            "macro_tensor_batch": macro_tensor_batch,
            "macro_key_padding_mask_batch": macro_key_padding_mask_batch
            }, "input2": {
            "price_tensor_batch": price_tensor2_batch,
            "price_key_padding_mask_batch": price_key_padding_mask2_batch,
            "fundamentals_tensor_batch": fundamentals_tensor2_batch,
            "fundamentals_key_padding_mask_batch": fundamentals_key_padding_mask2_batch,
            "fundamentals_missing_features_mask_batch": fundamentals_missing_features_mask2_batch,
            "fundamentals_missing_features_key_padding_mask_batch": fundamentals_missing_features_key_padding_mask2_batch,
            "company_features_tensor_batch": company_features_tensor2_batch,
            "company_description_input_ids_batch": company_description_input_ids2_batch,
            "company_description_key_padding_mask_batch": company_description_key_padding_mask2_batch,
            "macro_tensor_batch": macro_tensor_batch,
            "macro_key_padding_mask_batch": macro_key_padding_mask_batch},

        },
        "target": target_batch
        }

        return batch_dict
    
        


