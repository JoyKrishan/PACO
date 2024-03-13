import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, BertTokenizer, BertLMHeadModel
import os 
import json
from pathlib import Path
import sys
from torch.nn.utils.rnn import pad_sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
code_corpus = os.path.join(PROCESSED_DATA_DIR, 'code_corpus.txt')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertLMHeadModel.from_pretrained('bert-base-uncased')
# model = T5ForConditionalGeneration.from_pretrained('t5-small')

class CodeDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]

        buggy_code =  example['buggy_code']
        correct_patches = example['correct_patches'] 
        incorrect_patches = example['incorrect_patches'] 

        return buggy_code, correct_patches, incorrect_patches 
    

def prepare_batch(batch):
    positives, negatives = [], []

    buggy_code =  batch['buggy_code']
    correct_patches = batch['correct_patches'] 
    incorrect_patches = batch['incorrect_patches']

    anchor = tokenizer.encode(batch['buggy_code'], return_tensors='pt', add_special_tokens=True)

    for correct_patch in correct_patches:
        pos_tokens = tokenizer.encode(correct_patch, return_tensors='pt', add_special_tokens=True)
        positives.append(pos_tokens)

    for incorrect_patch in incorrect_patches:
        neg_tokens = tokenizer.encode(incorrect_patch, return_tensors='pt', add_special_tokens=True)
        negatives.append(neg_tokens)

    return anchor, positives, negatives


def get_embeddings(model, input_data):
    print(model.config)
    if input_data.size(1) > model.config.max_position_embeddings:
        input_data = input_data[:, :, model.config.max_position_embeddings]
    embd = model.generate(input_data, max_length = 512)
    return embd


def triplet_loss(anchor_emb, positive_emb, negative_emb, margin=1.0):
    distance_positive = (anchor_emb - positive_emb).pow(2).sum(1)  
    distance_negative = (anchor_emb - negative_emb).pow(2).sum(1) 
    losses = torch.relu(distance_positive - distance_negative + margin)
    return losses.mean()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def train(data):
    for epoch in range(10): 
        for batch in data:
            pos_embeddings = []

            anchor, positives, negatives = prepare_batch(batch)
            anchor_embedding = get_embeddings(model, anchor)

            cos_sim = nn.CosineSimilarity(dim = 1)
            
            for pos in positives:
                pos_embd = get_embeddings(model, pos)
                pos_embeddings.append(pos_embd)
                print(cos_sim(pos_embd, anchor_embedding))
            # positive_embeddings = get_embeddings(model, positives) 
            # negative_embeddings = get_embeddings(model, negatives)
            sys.exit()

            loss = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 


data_path = os.path.join(PROCESSED_DATA_DIR, "tokenized_dataset.json")

dataset = CodeDataset(data_path) 
dataloader = DataLoader(dataset, batch_size=4, shuffle=True) 

with open(data_path, 'r') as f:
    data = json.load(f)

train(data)