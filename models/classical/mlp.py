import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple MLP baseline for fairer comparison with hybrid quantum models."""

    def __init__(self, input_dim=28 * 28, hidden_dims=(128, 64), num_classes=10, dropout=0.3):
        super().__init__()
        h1, h2 = hidden_dims
        self.fc1 = nn.Linear(input_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
