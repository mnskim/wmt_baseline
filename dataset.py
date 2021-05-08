import torch
from torch.utils.data import DataLoader, Dataset
import pdb

def preprocess_function(examples, tokenizer):
    inputs = [prefix + ex[source_lang] for ex in examples["translation"]]
    targets = [ex[target_lang] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


class TranslationDataset(Dataset):
    def __init__(self, raw_data, tokenizer, max_input_length, max_output_length, source_lang, target_lang):
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        src, tgt = self.raw_data[idx]

        #input_ids = self.tokenizer.encode(src, truncation=True, max_length=self.max_input_len)
        model_inputs = self.tokenizer(src, max_length=self.max_input_length, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            #output_ids = self.tokenizer.encode(tgt, truncation=True, max_length=self.max_output_len)
            labels = self.tokenizer(tgt, max_length=self.max_output_length, truncation=True)

        #pdb.set_trace()
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs
