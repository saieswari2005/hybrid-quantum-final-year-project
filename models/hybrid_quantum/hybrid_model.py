import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hybrid_quantum.vqc import VariationalQuantumCircuit


class HybridQuantumClassical(nn.Module):
    """Hybrid Quantum-Classical Neural Network."""
    
    def __init__(self, input_dim, num_classes=10, num_qubits=4, 
                 circuit_depth=2, classical_hidden=64):
        super(HybridQuantumClassical, self).__init__()
        
        self.num_qubits = num_qubits
        
        # Classical pre-processing layer
        self.fc_pre = nn.Linear(input_dim, num_qubits)
        
        # Quantum circuit
        self.quantum_layer = VariationalQuantumCircuit(num_qubits, circuit_depth)
        
        # Classical post-processing layers
        self.fc_post1 = nn.Linear(num_qubits, classical_hidden)
        self.fc_post2 = nn.Linear(classical_hidden, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Flatten if needed
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        
        # Classical pre-processing
        x = torch.tanh(self.fc_pre(x))
        
        # Normalize to [-pi, pi] for angle encoding
        x = x * torch.pi
        
        # Quantum layer
        x = self.quantum_layer(x)
        
        # Classical post-processing
        x = F.relu(self.fc_post1(x))
        x = self.dropout(x)
        x = self.fc_post2(x)
        
        return x
