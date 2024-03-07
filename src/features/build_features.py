import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import json
import os
from pathlib import Path
# from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence



# tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
code_corpus = os.path.join(PROCESSED_DATA_DIR, 'code_corpus.txt')

# def ai_tokenize_code(code_corpus_path):

#     tokenizer = Tokenizer(models.BPE())
#     tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
#     trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["[CLS]", "[SEP]", "[PAD]"])
#     tokenizer.train(files= [code_corpus_path], trainer = trainer)

#     return tokenizer

class CodeDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base") 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        buggy_code = self.tokenizer(example['buggy_code'], padding='max_length', truncation=True, return_tensors='pt')['input_ids']

        correct_patches = [
            self.tokenizer(patch, padding = 'max_length', truncation=True, return_tensors='pt')['input_ids']
            for patch in example['correct_patches'] if patch 
         ]
        incorrect_patches = [
            self.tokenizer(patch, padding = 'max_length', truncation=True, return_tensors='pt')['input_ids']
            for patch in example['incorrect_patches'] if patch 
         ]
        return buggy_code, correct_patches, incorrect_patches 
    

def custom_collate_fn(batch):
    # Extract elements 
    buggy_codes, correct_patches_list, incorrect_patches_list = zip(*batch)

    # Pad elements as needed (assuming individual patches already have same length)
    buggy_codes = pad_sequence(buggy_codes, batch_first=True, padding_value=0)

    correct_patches = [pad_sequence(patch_tensors, batch_first=True, padding_value=0) 
                       for patch_tensors in correct_patches_list if patch_tensors]
    correct_patches = pad_sequence(correct_patches, batch_first=True, padding_value=0)

    incorrect_patches = [pad_sequence(patch_tensors, batch_first=True, padding_value=0) 
                       for patch_tensors in incorrect_patches_list if patch_tensors]
    if incorrect_patches:
        incorrect_patches = pad_sequence(incorrect_patches, batch_first=True, padding_value=0)
    else:
        incorrect_patches = torch.tensor([])

    return buggy_codes, correct_patches, incorrect_patches

# class CodePatchDataset(Dataset):
#     def __init__(self, json_file, tokenizer):
#         with open(json_file, 'r') as f:
#             self.data = json.load(f)
#         self.tokenizer = tokenizer 

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         example = self.data[index]
#         buggy_code = self.tokenizer(example['buggy_code'], padding = 'max_length', truncation=True ,return_tensors='pt')
#         correct_patches = [
#             self.tokenizer(patch, padding = 'max_length', truncation=True, return_tensors='pt')
#             for patch in example['correct_patches'] if patch 
#          ]
#         incorrect_patches = [
#             self.tokenizer(patch, padding = 'max_length', truncation=True, return_tensors='pt')
#             for patch in example['incorrect_patches'] if patch 
#          ]
#         return buggy_code, correct_patches, incorrect_patches 
    
    # def __getitem__(self, index):
    #     example = self.data[index]

    #     # Handle buggy code with a check
    #     buggy_encoding = self.tokenizer.encode(example['buggy_code'])
    #     if buggy_encoding.ids:
    #         buggy_code = rnn_utils.pad_sequence([torch.tensor(buggy_encoding.ids)], batch_first=True, padding_value=0)
    #     else:
    #         return self.__getitem__(index + 1)

    #     # Handle correct and incorrect patches similarly 
    #     correct_patches = [
    #         rnn_utils.pad_sequence([torch.tensor((self.tokenizer.encode(patch)).ids)], batch_first=True, padding_value=0)
    #         for patch in example['correct_patches'] if patch 
    #     ]
    #     incorrect_patches = [
    #         rnn_utils.pad_sequence([torch.tensor((self.tokenizer.encode(patch)).ids)], batch_first=True, padding_value=0)
    #         for patch in example['incorrect_patches'] if patch
    #     ]
    #     return buggy_code, correct_patches, incorrect_patches 

    # def __getitem__(self, index):
    #         example = self.data[index]




    #     buggy_code = rnn_utils.pad_sequence(torch.tensor((self.tokenizer.encode(example['buggy_code'])).ids), batch_first= True, padding_value=0)[0]
    #     print("DONE")
    #     correct_patches = [rnn_utils.pad_sequence(torch.tensor((self.tokenizer.encode(patch)).ids), batch_first=True, padding_value=0)[0] for patch in example['correct_patches'] if patch]
    #     incorrect_patches = [rnn_utils.pad_sequence(torch.tensor((self.tokenizer.encode(patch)).ids), batch_first=True, padding_value=0) for patch in example['incorrect_patches'] if patch]
    #     return buggy_code, correct_patches, incorrect_patches 
    # def __getitem__(self, index):
    #     example = self.data[index]

    #     # Padding
    #     buggy_code = rnn_utils.pad_sequence([torch.tensor(example['buggy_code_tokens'])], 
    #                                         batch_first=True,
    #                                         padding_value=0)[0]

    #     correct_patches = [rnn_utils.pad_sequence([torch.tensor(patch)], 
    #                                               batch_first=True, 
    #                                               padding_value=0)[0] 
    #                        for patch in example['correct_patches_tokens']]

    #     incorrect_patches = [rnn_utils.pad_sequence([torch.tensor(patch)], 
    #                                                 batch_first=True, 
    #                                                 padding_value=0)[0] 
    #                          for patch in example['incorrect_patches_tokens']]

        # return buggy_code, correct_patches, incorrect_patches

# tokenizer = ai_tokenize_code(code_corpus)


data_path = os.path.join(PROCESSED_DATA_DIR, "tokenized_dataset.json")

dataset = CodeDataset(data_path) 
dataloader = DataLoader(dataset, batch_size=4, collate_fn=custom_collate_fn) 

for buggy_code, correct_patches, incorrect_patches in dataloader:
    print(buggy_code.shape)
    print(correct_patches.shape)
    print(incorrect_patches.shape)
    break