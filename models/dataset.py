from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset

label2id = {"TRUE":0, "Fake":1}

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = str(self.texts[index])
        # label = int(self.labels[index])
        label = label2id.get(self.labels[index], -1)

        encoded_input = self.tokenizer.encode_plus(text, add_special_tokens=True, max_length=512, padding='max_length', truncation=True, return_tensors='pt')
        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }