import torch
import torch.nn as nn


class LSTM(nn.Module):
    """LSTM Network for text classification."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, 
                 num_layers=2, num_classes=2, dropout=0.5):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        hidden_last = hidden[-1]
        
        # Dropout and FC
        out = self.dropout(hidden_last)
        out = self.fc(out)
        
        return out
