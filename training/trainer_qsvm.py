import numpy as np
import time
import psutil
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


class QSVMTrainer:
    """Trainer for Quantum Kernel SVM."""
    
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        
        # Tracking
        self.training_time = 0
        self.peak_memory = 0
        self.metadata = {}
        self.pca = None
        self.scaler = None
        self.test_subset_size = config.get('evaluation', {}).get('qsvm_test_subset', 500)

    def _flatten_if_needed(self, X):
        if len(X.shape) > 2:
            return X.reshape(X.shape[0], -1)
        return X

    def _fit_transform_features(self, X_train, X_val):
        """Fit PCA/scaler on train and transform train/val consistently."""
        num_qubits = self.model.num_qubits
        n_components = min(num_qubits, X_train.shape[1], X_train.shape[0])

        self.pca = PCA(n_components=n_components, random_state=self.config.get('random_seed', 42))
        X_train_pca = self.pca.fit_transform(X_train)
        X_val_pca = self.pca.transform(X_val)

        # Pad if PCA returns fewer dims than requested (very small sample edge case)
        if n_components < num_qubits:
            train_pad = np.zeros((X_train_pca.shape[0], num_qubits - n_components))
            val_pad = np.zeros((X_val_pca.shape[0], num_qubits - n_components))
            X_train_pca = np.hstack([X_train_pca, train_pad])
            X_val_pca = np.hstack([X_val_pca, val_pad])

        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train_scaled = self.scaler.fit_transform(X_train_pca)
        X_val_scaled = self.scaler.transform(X_val_pca)
        return X_train_scaled, X_val_scaled

    def _transform_features(self, X):
        """Apply train-fitted PCA/scaler to validation/test data."""
        X_pca = self.pca.transform(X)
        if X_pca.shape[1] < self.model.num_qubits:
            pad = np.zeros((X_pca.shape[0], self.model.num_qubits - X_pca.shape[1]))
            X_pca = np.hstack([X_pca, pad])
        return self.scaler.transform(X_pca)

    def _balanced_subset(self, X, y, max_samples):
        """Class-balanced subset for expensive QSVM test evaluation."""
        if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
            return X, y

        classes = np.unique(y)
        per_class = max(1, max_samples // len(classes))
        selected = []
        rng = np.random.default_rng(self.config.get('random_seed', 42))
        for c in classes:
            idx = np.where(y == c)[0]
            take = min(len(idx), per_class)
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())

        if len(selected) < max_samples:
            remaining = np.setdiff1d(np.arange(len(y)), np.array(selected, dtype=int), assume_unique=False)
            extra = min(len(remaining), max_samples - len(selected))
            if extra > 0:
                selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())

        selected = np.array(selected[:max_samples], dtype=int)
        return X[selected], y[selected]
    
    def train(self, train_data, val_data):
        """Train the QSVM."""
        
        # Extract data from TensorDataset
        X_train = train_data.tensors[0].numpy()
        y_train = train_data.tensors[1].numpy()
        
        X_val = val_data.tensors[0].numpy()
        y_val = val_data.tensors[1].numpy()
        
        X_train = self._flatten_if_needed(X_train)
        X_val = self._flatten_if_needed(X_val)

        # Train-fitted reduction and scaling improves signal quality and keeps splits consistent.
        X_train_reduced, X_val_reduced = self._fit_transform_features(X_train, X_val)
        
        process = psutil.Process(os.getpid())
        start_time = time.time()
        
        self.logger.info("Training Quantum Kernel SVM...")
        
        # Train
        self.model.fit(X_train_reduced, y_train)
        
        self.training_time = time.time() - start_time
        
        # Track memory
        mem_info = process.memory_info()
        self.peak_memory = mem_info.rss / 1024 / 1024  # MB
        
        # Evaluate on validation set
        val_score = self.model.score(X_val_reduced, y_val)
        
        self.logger.info(f"QSVM Training completed in {self.training_time:.2f}s")
        self.logger.info(f"Validation Accuracy: {val_score:.4f}")
        self.metadata.update({
            'num_qubits': self.model.num_qubits,
            'circuit_depth': self.model.circuit_depth,
            'qsvm_test_subset': self.test_subset_size
        })
        
        # Create history for consistency with other trainers
        self.history = {
            'train_loss': [0],  # Not applicable for SVM
            'train_acc': [self.model.score(X_train_reduced, y_train)],
            'val_loss': [0],
            'val_acc': [val_score]
        }
        
        return self.history
    
    def evaluate(self, test_data):
        """Evaluate on test set."""
        
        # Extract data
        X_test = test_data.tensors[0].numpy()
        y_test = test_data.tensors[1].numpy()
        
        X_test = self._flatten_if_needed(X_test)
        X_test, y_test = self._balanced_subset(X_test, y_test, self.test_subset_size)
        self.metadata['test_samples_evaluated'] = int(len(y_test))
        X_test_reduced = self._transform_features(X_test)
        
        # Predict
        predictions = self.model.predict(X_test_reduced)
        
        return predictions, y_test
