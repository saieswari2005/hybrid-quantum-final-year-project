import pennylane as qml
import torch
import torch.nn as nn
import numpy as np


class VariationalQuantumCircuit(nn.Module):
    """Variational Quantum Circuit using PennyLane."""
    
    def __init__(self, num_qubits=4, circuit_depth=2):
        super(VariationalQuantumCircuit, self).__init__()
        
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=num_qubits)
        
        # Initialize quantum weights
        weight_shape = (circuit_depth, num_qubits, 3)
        self.q_weights = nn.Parameter(torch.randn(weight_shape) * 0.01)
        
        # Create quantum node
        self.qnode = qml.QNode(
            self._circuit,
            self.dev,
            interface='torch',
            diff_method='backprop'
        )
    
    def _circuit(self, inputs, weights):
        """Quantum circuit with angle encoding and variational layers."""
        
        # Angle encoding
        for i in range(self.num_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational layers
        for layer in range(self.circuit_depth):
            # Rotation gates
            for i in range(self.num_qubits):
                qml.Rot(weights[layer, i, 0], 
                       weights[layer, i, 1], 
                       weights[layer, i, 2], wires=i)
            
            # Entangling gates
            for i in range(self.num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            if self.num_qubits > 2:
                qml.CNOT(wires=[self.num_qubits - 1, 0])
        
        # Measurements
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
    
    def forward(self, x):
        """Forward pass through quantum circuit."""
        batch_outputs = []
        for i in range(x.size(0)):
            q_out = self.qnode(x[i], self.q_weights)
            if isinstance(q_out, (list, tuple)):
                q_out = torch.stack(q_out)
            batch_outputs.append(q_out)

        return torch.stack(batch_outputs, dim=0).to(device=x.device, dtype=x.dtype)
