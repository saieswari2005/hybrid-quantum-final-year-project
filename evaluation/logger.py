import logging
import os
from datetime import datetime


def setup_logger(log_dir, experiment_name="quantum_ml_benchmark"):
    """
    Set up logger for the experiment.
    
    Args:
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
    
    Returns:
        Logger instance
    """
    
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Create logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers if any
    logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def log_experiment_start(logger, config):
    """Log experiment configuration at start."""
    
    logger.info("=" * 80)
    logger.info("QUANTUM ML BENCHMARKING EXPERIMENT")
    logger.info("=" * 80)
    logger.info(f"Random Seed: {config['random_seed']}")
    logger.info(f"Datasets: {[d['name'] for d in config['datasets']]}")
    logger.info(f"Dataset Sizes: {config['dataset_sizes']}")
    logger.info(f"Learning Rate: {config['training']['learning_rate']}")
    logger.info(f"Batch Size: {config['training']['batch_size']}")
    logger.info(f"Epochs: {config['training']['epochs']}")
    logger.info(f"Early Stopping Patience: {config['training']['early_stopping_patience']}")
    logger.info("=" * 80)


def log_dataset_info(logger, dataset_name, dataset_size, num_train, num_val, num_test):
    """Log dataset information."""
    
    logger.info("-" * 80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Target Training Size: {dataset_size}")
    logger.info(f"Actual Training Samples: {num_train}")
    logger.info(f"Validation Samples: {num_val}")
    logger.info(f"Test Samples: {num_test}")
    logger.info("-" * 80)


def log_model_training_start(logger, model_name, dataset_name, dataset_size):
    """Log start of model training."""
    
    logger.info("")
    logger.info("*" * 80)
    logger.info(f"Training {model_name} on {dataset_name} (n={dataset_size})")
    logger.info("*" * 80)


def log_model_results(logger, results):
    """Log model results."""
    
    logger.info("")
    logger.info("Results:")
    logger.info(f"  Train Accuracy: {results['train_accuracy']:.4f}")
    logger.info(f"  Val Accuracy: {results['val_accuracy']:.4f}")
    logger.info(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    logger.info(f"  Test F1 Score: {results['test_f1_score']:.4f}")
    logger.info(f"  Overfitting Gap: {results['overfitting_gap']:.4f}")
    logger.info(f"  Training Time: {results['training_time']:.2f}s")
    logger.info(f"  Peak Memory: {results['peak_memory_mb']:.2f} MB")
    logger.info("")


def log_experiment_end(logger):
    """Log experiment completion."""
    
    logger.info("=" * 80)
    logger.info("EXPERIMENT COMPLETED")
    logger.info("=" * 80)
