import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import psutil
import os


class HybridTrainer:
    """Trainer for hybrid quantum-classical models."""
    
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.device = torch.device(config['computation']['device'])
        self.model.to(self.device)
        
        # Training parameters
        self.lr = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.epochs = config['training']['epochs']
        self.patience = config['training']['early_stopping_patience']
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        
        # Tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        self.training_time = 0
        self.peak_memory = 0
        self.metadata = {}
        self.grad_history = {
            'quantum_grad_norm': [],
            'classical_grad_norm': []
        }
        self.test_subset_size = config.get('evaluation', {}).get('hybrid_test_subset', None)
    
    def train(self, train_data, val_data):
        """Train the hybrid model."""
        
        # Create data loaders (smaller batch size for quantum models)
        train_loader = DataLoader(train_data, batch_size=min(self.batch_size, 16), 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=min(self.batch_size, 16), 
                               shuffle=False, num_workers=0)
        
        best_val_acc = 0
        patience_counter = 0
        start_time = time.time()
        
        process = psutil.Process(os.getpid())
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Track memory
            mem_info = process.memory_info()
            self.peak_memory = max(self.peak_memory, mem_info.rss / 1024 / 1024)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
            
            if patience_counter >= self.patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        self.training_time = time.time() - start_time
        
        # Load best model
        if hasattr(self, 'best_state'):
            self.model.load_state_dict(self.best_state)

        self.metadata.update({
            'num_qubits': getattr(self.model, 'num_qubits', None),
            'circuit_depth': getattr(getattr(self.model, 'quantum_layer', None), 'circuit_depth', None),
            'hybrid_mean_quantum_grad_norm': float(np.mean(self.grad_history['quantum_grad_norm'])) if self.grad_history['quantum_grad_norm'] else 0.0,
            'hybrid_mean_classical_grad_norm': float(np.mean(self.grad_history['classical_grad_norm'])) if self.grad_history['classical_grad_norm'] else 0.0
        })
        
        return self.history
    
    def _train_epoch(self, loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()

            q_grad_sq = 0.0
            c_grad_sq = 0.0
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue
                grad_norm = param.grad.detach().norm(2).item()
                if 'quantum_layer' in name:
                    q_grad_sq += grad_norm ** 2
                else:
                    c_grad_sq += grad_norm ** 2
            self.grad_history['quantum_grad_norm'].append(q_grad_sq ** 0.5)
            self.grad_history['classical_grad_norm'].append(c_grad_sq ** 0.5)

            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate(self, loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, test_data):
        """Evaluate on test set."""
        if hasattr(test_data, 'tensors') and self.test_subset_size:
            x_test, y_test = test_data.tensors
            x_test, y_test = self._balanced_subset(x_test, y_test, int(self.test_subset_size))
            test_data = torch.utils.data.TensorDataset(x_test, y_test)
            self.metadata['test_samples_evaluated'] = int(len(y_test))

        test_loader = DataLoader(test_data, batch_size=min(self.batch_size, 16), 
                                shuffle=False, num_workers=0)
        
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.numpy())
        
        return np.array(all_preds), np.array(all_labels)

    def _balanced_subset(self, X, y, max_samples):
        if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
            return X, y
        y_np = y.detach().cpu().numpy() if torch.is_tensor(y) else np.array(y)
        classes = np.unique(y_np)
        per_class = max(1, max_samples // len(classes))
        rng = np.random.default_rng(self.config.get('random_seed', 42))
        selected = []
        for c in classes:
            idx = np.where(y_np == c)[0]
            take = min(len(idx), per_class)
            selected.extend(rng.choice(idx, size=take, replace=False).tolist())
        if len(selected) < max_samples:
            remaining = np.setdiff1d(np.arange(len(y_np)), np.array(selected, dtype=int))
            extra = min(len(remaining), max_samples - len(selected))
            if extra > 0:
                selected.extend(rng.choice(remaining, size=extra, replace=False).tolist())
        selected = np.array(selected[:max_samples], dtype=int)
        if torch.is_tensor(X):
            X = X[selected]
        else:
            X = X[selected]
        if torch.is_tensor(y):
            y = y[selected]
        else:
            y = y_np[selected]
        return X, y
