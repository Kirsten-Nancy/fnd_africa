import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset

class NewsClassifier(nn.Module):
    def __init__(self, model_name, num_labels) -> None:
        super(NewsClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # TODO: fc network
        self.fc = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        logits = self.fc(pooled_output)
        return logits
    
label2id = {"TRUE":0, "Fake":1}

class CustomDataset(Dataset):
    def __init__(self, texts, labels) -> None:
        super().__init__()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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
