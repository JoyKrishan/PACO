import torch.nn as nn
import torch

class SupConLossWithConsine(nn.Module):
    
    def __init__(self, device, margin = 0.5) -> None:
        super().__init__()
        self.margin = margin
        self.sim = nn.CosineSimilarity(dim = 2)
        self.device = device
        
    def forward(self, buggy_embd, patch_embd, labels):
        if buggy_embd.shape != patch_embd.shape:
            raise ValueError(f'Shape embd1: {buggy_embd} does not match shape embd2: {patch_embd}')
    
        batch_size = buggy_embd.shape[0]
        
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
    
        # positive patches
        anchor_label = torch.ones(batch_size).to(self.device)
        is_positive = anchor_label == labels
        positive_similarities = self.sim(buggy_embd, patch_embd)[is_positive]
        loss_positive = torch.clamp(self.margin - positive_similarities, min=0).pow(2).mean()

        # negative patches
        is_negative = ~is_positive
        negative_similarities = self.sim(buggy_embd, patch_embd)[is_negative]
        loss_negative = torch.clamp(negative_similarities + self.margin, min=0).pow(2).mean()

        return loss_positive + loss_negative