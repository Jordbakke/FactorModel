
from data_utils import load_tensor
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, mapping_df, fundamentals_tensor_path, prices_tensor_path, company_features_tensor_path, macro_tensor_path,
                 company_description_input_ids_tensor_path, company_description_attention_mask_tensor_path, targets_tensor_path):
        
        self.mapping_df = mapping_df
        self.fundamentals = load_tensor(fundamentals_tensor_path)
        self.prices = load_tensor(prices_tensor_path)
        self.company_features = load_tensor(company_features_tensor_path)
        self.macro = load_tensor(macro_tensor_path)
        self.company_description_input_ids = load_tensor(company_description_input_ids_tensor_path)
        self.company_description_attention_mask = load_tensor(company_description_attention_mask_tensor_path)
        self.targets = load_tensor(targets_tensor_path)

    def __len__(self):
        return len(self.mapping_df)

    def __getitem__(self, idx):

        pass