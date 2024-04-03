import torch.nn as nn

class LSTMCodeEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.batch_norm_one = nn.BatchNorm1d(embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear_projection = nn.Linear(hidden_dim, 128) 
        
    def forward(self, code_tokens):
        embeddings = self.embedding(code_tokens)
        embeddings = self.batch_norm_one(embeddings) # Apply batch norm one
        output, (hidden, cell) = self.lstm(embeddings)  
        output = self.linear_projection(output)
        return output 