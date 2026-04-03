import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import time
import psutil
import os


class ClassicalTrainer:
    """Trainer for classical PyTorch models."""
    
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
    
    def train(self, train_data, val_data):
        """Train the model."""
        
        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=self.batch_size, 
                                 shuffle=True, num_workers=0)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, 
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
            self.peak_memory = max(self.peak_memory, mem_info.rss / 1024 / 1024)  # MB
            
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
                # Save best model
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
        test_loader = DataLoader(test_data, batch_size=self.batch_size, 
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
