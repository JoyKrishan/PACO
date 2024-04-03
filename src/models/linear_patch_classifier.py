import torch.nn as nn
import torch.nn.functional as F  

class PatchClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim=1):  
        super().__init__()
        self.linear_layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),  
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
            )
        self.sim = nn.CosineSimilarity(dim = 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, buggy_embedding, patch_embedding):
        cos_similarities = self.sim(buggy_embedding, patch_embedding) 
        output = self.linear_layers(cos_similarities) 
        return self.sigmoid(output).squeeze() 
