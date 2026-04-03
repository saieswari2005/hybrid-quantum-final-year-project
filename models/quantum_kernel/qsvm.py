import pennylane as qml
import numpy as np
from sklearn.svm import SVC
from sklearn.base import BaseEstimator, ClassifierMixin


class QuantumKernelSVM(BaseEstimator, ClassifierMixin):
    """Quantum Kernel SVM using PennyLane."""
    
    def __init__(self, num_qubits=4, circuit_depth=2, C=1.0, kernel_type='iqp'):
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.C = C
        self.kernel_type = kernel_type
        
        # Create quantum device
        self.dev = qml.device('default.qubit', wires=num_qubits)
        self.kernel_qnode = qml.QNode(self._quantum_kernel_circuit, self.dev)
        
        # Initialize SVM
        self.svm = None
        self.X_train = None
    
    def _quantum_feature_map(self, x):
        """Quantum feature map circuit."""
        
        # Angle encoding
        for i in range(self.num_qubits):
            qml.RY(x[i], wires=i)
        
        # IQP-style entangling layers
        if self.kernel_type == 'iqp':
            for layer in range(self.circuit_depth):
                # Hadamard layer
                for i in range(self.num_qubits):
                    qml.Hadamard(wires=i)
                
                # ZZ interactions
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
                if self.num_qubits > 2:
                    qml.CZ(wires=[self.num_qubits - 1, 0])
                
                # Rotation layer
                for i in range(self.num_qubits):
                    qml.RZ(x[i], wires=i)
    
    def _quantum_kernel_circuit(self, x1, x2):
        """Compute quantum kernel between two samples."""
        
        # Apply feature map to first sample
        self._quantum_feature_map(x1)
        
        # Apply adjoint of feature map to second sample
        qml.adjoint(self._quantum_feature_map)(x2)
        
        return qml.probs(wires=range(self.num_qubits))
    
    def _compute_kernel_element(self, x1, x2):
        """Compute single kernel matrix element."""
        probs = self.kernel_qnode(x1, x2)
        # Kernel is the probability of measuring all zeros
        return probs[0]
    
    def _compute_kernel_matrix(self, X1, X2=None):
        """Compute full kernel matrix."""
        if X2 is None:
            X2 = X1
        
        K = np.zeros((len(X1), len(X2)))
        is_symmetric = X2 is X1

        if is_symmetric:
            for i in range(len(X1)):
                for j in range(i, len(X2)):
                    val = self._compute_kernel_element(X1[i], X2[j])
                    K[i, j] = val
                    K[j, i] = val
        else:
            for i in range(len(X1)):
                for j in range(len(X2)):
                    K[i, j] = self._compute_kernel_element(X1[i], X2[j])
        
        return K
    
    def fit(self, X, y):
        """Train the quantum kernel SVM."""
        
        # Ensure X has correct number of features
        if X.shape[1] != self.num_qubits:
            # Truncate or pad to match num_qubits
            if X.shape[1] > self.num_qubits:
                X = X[:, :self.num_qubits]
            else:
                padding = np.zeros((X.shape[0], self.num_qubits - X.shape[1]))
                X = np.hstack([X, padding])
        
        self.X_train = X.copy()
        
        # Compute kernel matrix for training data
        K_train = self._compute_kernel_matrix(X)
        
        # Train SVM with precomputed kernel
        self.svm = SVC(kernel='precomputed', C=self.C)
        self.svm.fit(K_train, y)
        
        return self
    
    def predict(self, X):
        """Predict using the quantum kernel SVM."""
        
        # Ensure X has correct number of features
        if X.shape[1] != self.num_qubits:
            if X.shape[1] > self.num_qubits:
                X = X[:, :self.num_qubits]
            else:
                padding = np.zeros((X.shape[0], self.num_qubits - X.shape[1]))
                X = np.hstack([X, padding])
        
        # Compute kernel matrix between test and train data
        K_test = self._compute_kernel_matrix(X, self.X_train)
        
        # Predict using SVM
        return self.svm.predict(K_test)
    
    def score(self, X, y):
        """Compute accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
