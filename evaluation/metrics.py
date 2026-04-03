import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(predictions, labels, num_classes=10):
    """
    Compute comprehensive metrics for model evaluation.
    
    Args:
        predictions: Predicted labels
        labels: True labels
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    
    # Ensure arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Basic metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Determine average type based on number of classes
    if num_classes == 2:
        avg_type = 'binary'
    else:
        avg_type = 'macro'
    
    # F1 score
    f1 = f1_score(labels, predictions, average=avg_type, zero_division=0)
    
    # Precision and recall
    precision = precision_score(labels, predictions, average=avg_type, zero_division=0)
    recall = recall_score(labels, predictions, average=avg_type, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics


def compute_overfitting_gap(train_acc, val_acc):
    """
    Compute overfitting gap as difference between train and validation accuracy.
    
    Args:
        train_acc: Training accuracy
        val_acc: Validation accuracy
    
    Returns:
        Overfitting gap (positive value indicates overfitting)
    """
    return train_acc - val_acc


def aggregate_results(history, trainer, test_predictions, test_labels, 
                     dataset_name, dataset_size, model_name, num_classes):
    """
    Aggregate all results into a single dictionary.
    
    Args:
        history: Training history
        trainer: Trainer object
        test_predictions: Test set predictions
        test_labels: Test set labels
        dataset_name: Name of dataset
        dataset_size: Size of training set
        model_name: Name of model
        num_classes: Number of classes
    
    Returns:
        Dictionary of aggregated results
    """
    
    # Get final training and validation accuracy
    train_acc = history['train_acc'][-1] if history['train_acc'] else 0
    val_acc = history['val_acc'][-1] if history['val_acc'] else 0
    
    # Compute test metrics
    test_metrics = compute_metrics(test_predictions, test_labels, num_classes)
    
    # Compute overfitting gap
    overfitting_gap = compute_overfitting_gap(train_acc, val_acc)
    
    # Aggregate results
    results = {
        'dataset': dataset_name,
        'dataset_size': dataset_size,
        'model': model_name,
        'train_accuracy': train_acc,
        'val_accuracy': val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_f1_score': test_metrics['f1_score'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],
        'overfitting_gap': overfitting_gap,
        'training_time': trainer.training_time,
        'peak_memory_mb': trainer.peak_memory
    }

    # Include trainer-provided metadata (e.g., qubits/depth/test subset size) when available.
    metadata = getattr(trainer, 'metadata', None)
    if isinstance(metadata, dict):
        results.update(metadata)
    
    return results
