import os
import argparse
import copy
import traceback
import yaml
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import TensorDataset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix

# Import models
from models.classical.cnn import CNN
from models.classical.lstm import LSTM
from models.classical.mlp import MLP
from models.hybrid_quantum.hybrid_model import HybridQuantumClassical
from models.quantum_kernel.qsvm import QuantumKernelSVM

# Import trainers
from training.trainer_classical import ClassicalTrainer
from training.trainer_hybrid import HybridTrainer
from training.trainer_qsvm import QSVMTrainer

# Import evaluation
from evaluation.metrics import aggregate_results
from evaluation.plots import generate_paper_plots, plot_confusion_matrix_best_models
from evaluation.logger import setup_logger, log_experiment_start, log_dataset_info
from evaluation.logger import log_model_training_start, log_model_results, log_experiment_end


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    """Parse CLI arguments for reproducible experiment control."""
    parser = argparse.ArgumentParser(description="Quantum ML Benchmark runner")
    parser.add_argument('--config', default='config.yaml', help='Path to YAML config (default: config.yaml)')
    parser.add_argument('--seeds', default='0,1,2', help='Comma-separated seeds (default: 0,1,2)')
    parser.add_argument('--datasets', default='mnist,fashionmnist', help='Comma-separated dataset names')
    parser.add_argument('--sizes', default=None, help='Comma-separated dataset sizes (default: use config)')
    parser.add_argument(
        '--models',
        default=None,
        help='Comma-separated models from {cnn,mlp,lstm,hybrid_qc,qsvm} (default: config/models_to_run or all enabled)'
    )
    parser.add_argument('--run_ablation', action='store_true', help='Enable ablation grids for QML models')
    parser.add_argument('--skip_classical', action='store_true', help='Skip classical models (CNN/LSTM; MLP only if explicitly requested)')
    parser.add_argument('--summary_only', action='store_true', help='Skip training and rebuild summaries/plots from existing all_results.csv')
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs for this run')
    parser.add_argument('--patience', type=int, default=None, help='Override early stopping patience for this run')
    parser.add_argument('--hybrid_qubits', default=None, help='Ablation override for Hybrid-QC qubits, e.g. "2,4"')
    parser.add_argument('--hybrid_depths', default=None, help='Ablation override for Hybrid-QC depths, e.g. "1,2,3"')
    parser.add_argument('--qsvm_qubits', default=None, help='Ablation override for QSVM qubits, e.g. "2,4"')
    parser.add_argument('--qsvm_depths', default=None, help='Ablation override for QSVM depths, e.g. "1,2"')
    return parser.parse_args()


def _split_csv_arg(value, cast=str):
    if value is None or value == "":
        return None
    items = [v.strip() for v in value.split(',') if v.strip()]
    return [cast(v) for v in items]


def _normalize_dataset_name(name):
    mapping = {
        'mnist': 'MNIST',
        'fashionmnist': 'FashionMNIST',
        'fashion-mnist': 'FashionMNIST',
        'imdb': 'IMDB',
    }
    return mapping.get(name.lower(), name)


def configure_run_from_args(config, args, legacy_mode):
    """Apply CLI filters/overrides while preserving legacy no-arg behavior."""
    run_opts = {
        'legacy_mode': legacy_mode,
        'requested_seeds': [config.get('random_seed', 42)] if legacy_mode else (_split_csv_arg(args.seeds, int) or [0, 1, 2]),
    }

    if args.sizes:
        config['dataset_sizes'] = _split_csv_arg(args.sizes, int)

    if args.epochs is not None:
        config['training']['epochs'] = int(args.epochs)
    if args.patience is not None:
        config['training']['early_stopping_patience'] = int(args.patience)

    if not legacy_mode:
        requested_datasets = {_normalize_dataset_name(d) for d in (_split_csv_arg(args.datasets, str) or [])}
        if requested_datasets:
            config['datasets'] = [d for d in config['datasets'] if d['name'] in requested_datasets]

    if not legacy_mode:
        requested_models = _split_csv_arg(args.models, str)
        base_models = config.get('models_to_run', {})
        if requested_models is None:
            requested_models = [k for k, v in base_models.items() if v] if base_models else ['cnn', 'mlp', 'hybrid_qc', 'qsvm']

        requested_models = {m.strip().lower() for m in requested_models}
        models_to_run = {k: False for k in ['cnn', 'mlp', 'lstm', 'hybrid_qc', 'qsvm']}
        for key in models_to_run:
            if key in requested_models:
                models_to_run[key] = True

        if args.skip_classical:
            models_to_run['cnn'] = False
            models_to_run['lstm'] = False
            if 'mlp' not in requested_models:
                models_to_run['mlp'] = False

        config['models_to_run'] = models_to_run

    if args.run_ablation and not legacy_mode:
        config.setdefault('ablation', {})
        config['ablation']['hybrid_quantum'] = {
            'num_qubits': [2, 4, 6],
            'circuit_depth': [1, 2, 3]
        }
        config['ablation']['quantum_kernel'] = {
            'num_qubits': [2, 4],
            'circuit_depth': [1, 2]
        }
        # Practical defaults for runtime
        config['qsvm_max_size'] = min(int(config.get('qsvm_max_size', 100)), 100)
        config.setdefault('evaluation', {})
        config['evaluation']['qsvm_test_subset'] = int(config['evaluation'].get('qsvm_test_subset', 500))

    if not legacy_mode:
        if args.hybrid_qubits or args.hybrid_depths:
            config.setdefault('ablation', {}).setdefault('hybrid_quantum', {})
            if args.hybrid_qubits:
                config['ablation']['hybrid_quantum']['num_qubits'] = _split_csv_arg(args.hybrid_qubits, int)
            if args.hybrid_depths:
                config['ablation']['hybrid_quantum']['circuit_depth'] = _split_csv_arg(args.hybrid_depths, int)
        if args.qsvm_qubits or args.qsvm_depths:
            config.setdefault('ablation', {}).setdefault('quantum_kernel', {})
            if args.qsvm_qubits:
                config['ablation']['quantum_kernel']['num_qubits'] = _split_csv_arg(args.qsvm_qubits, int)
            if args.qsvm_depths:
                config['ablation']['quantum_kernel']['circuit_depth'] = _split_csv_arg(args.qsvm_depths, int)

    if not args.run_ablation and not legacy_mode:
        # Disable ablation expansion unless explicitly requested.
        config['_disable_ablation_expansion'] = True

    return config, run_opts


def create_directories():
    """Create necessary directories for results."""
    dirs = [
        'data/cache',
        'results/metrics',
        'results/plots',
        'results/logs'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def load_mnist_data():

    
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root='./data/cache', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data/cache', train=False, download=True, transform=transform
    )
    
    # Convert to tensors
    train_data = train_dataset.data.float().unsqueeze(1) / 255.0
    train_labels = train_dataset.targets
    test_data = test_dataset.data.float().unsqueeze(1) / 255.0
    test_labels = test_dataset.targets
    
    # Normalize
    train_data = (train_data - 0.1307) / 0.3081
    test_data = (test_data - 0.1307) / 0.3081
    
    return train_data, train_labels, test_data, test_labels


def load_fashion_mnist_data():
    """Load Fashion-MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(
        root='./data/cache', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data/cache', train=False, download=True, transform=transform
    )
    
    train_data = train_dataset.data.float().unsqueeze(1) / 255.0
    train_labels = train_dataset.targets
    test_data = test_dataset.data.float().unsqueeze(1) / 255.0
    test_labels = test_dataset.targets
    
    # Normalize
    train_data = (train_data - 0.5) / 0.5
    test_data = (test_data - 0.5) / 0.5
    
    return train_data, train_labels, test_data, test_labels


def load_imdb_data(config_fast):
    """Load IMDB dataset (simplified binary classification)."""
    # For simplicity, we'll create a synthetic text dataset
    # In a real implementation, you would load actual IMDB data
    
    vocab_size = config_fast['text']['vocab_size']
    max_len = config_fast['text']['max_sequence_length']
    
    # Create synthetic data (50% positive, 50% negative)
    n_train = 25000
    n_test = 25000
    
    train_data = torch.randint(1, vocab_size, (n_train, max_len))
    train_labels = torch.cat([torch.zeros(n_train // 2), torch.ones(n_train // 2)]).long()
    
    test_data = torch.randint(1, vocab_size, (n_test, max_len))
    test_labels = torch.cat([torch.zeros(n_test // 2), torch.ones(n_test // 2)]).long()
    
    # Shuffle
    train_indices = torch.randperm(n_train)
    train_data = train_data[train_indices]
    train_labels = train_labels[train_indices]
    
    test_indices = torch.randperm(n_test)
    test_data = test_data[test_indices]
    test_labels = test_labels[test_indices]
    
    return train_data, train_labels, test_data, test_labels


def create_small_dataset(train_data, train_labels, size, val_split=0.2):
    """Create small dataset by subsampling."""
    
    # Sample indices
    num_classes = len(torch.unique(train_labels))
    samples_per_class = size // num_classes
    
    sampled_indices = []
    for c in range(num_classes):
        class_indices = (train_labels == c).nonzero(as_tuple=True)[0]
        if len(class_indices) < samples_per_class:
            sampled_indices.extend(class_indices.tolist())
        else:
            sampled = class_indices[torch.randperm(len(class_indices))[:samples_per_class]]
            sampled_indices.extend(sampled.tolist())
    
    sampled_indices = sampled_indices[:size]  # Ensure exact size
    
    # Shuffle
    random.shuffle(sampled_indices)
    
    # Split into train and validation
    val_size = int(len(sampled_indices) * val_split)
    train_size = len(sampled_indices) - val_size
    
    train_idx = sampled_indices[:train_size]
    val_idx = sampled_indices[train_size:]
    
    train_subset = TensorDataset(train_data[train_idx], train_labels[train_idx])
    val_subset = TensorDataset(train_data[val_idx], train_labels[val_idx])
    
    return train_subset, val_subset


def create_balanced_subset_from_arrays(X, y, max_samples, seed):
    """Balanced subset helper for confusion-matrix comparisons."""
    if max_samples is None or max_samples <= 0 or len(y) <= max_samples:
        return X, y
    y_np = np.array(y)
    classes = np.unique(y_np)
    per_class = max(1, max_samples // len(classes))
    rng = np.random.default_rng(seed)
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
        X_sel = X[selected]
    else:
        X_sel = X[selected]
    return X_sel, y_np[selected]


def summarize_mean_std(results_df):
    """Aggregate metrics over seeds for report-ready summary."""
    if results_df.empty:
        return pd.DataFrame()

    group_cols = ['dataset', 'dataset_size', 'model']
    for col in ['num_qubits', 'circuit_depth']:
        if col in results_df.columns:
            group_cols.append(col)

    metric_cols = [
        'train_accuracy', 'val_accuracy', 'test_accuracy', 'test_f1_score',
        'test_precision', 'test_recall', 'overfitting_gap', 'training_time', 'peak_memory_mb'
    ]
    metric_cols = [c for c in metric_cols if c in results_df.columns]

    grouped = results_df.groupby(group_cols, dropna=False)
    summary = grouped[metric_cols].agg(['mean', 'std']).reset_index()
    summary.columns = [
        f"{a}_{b}" if b else a
        for a, b in summary.columns.to_flat_index()
    ]
    counts = grouped.size().reset_index(name='n_runs')
    summary = summary.merge(counts, on=group_cols, how='left')
    return summary


def train_cnn_model(train_data, val_data, test_data, config, logger):
    """Train CNN model."""
    
    model = CNN(
        num_classes=10,
        conv1_channels=config['models']['classical']['cnn']['conv1_channels'],
        conv2_channels=config['models']['classical']['cnn']['conv2_channels'],
        fc_hidden=config['models']['classical']['cnn']['fc_hidden'],
        dropout=config['models']['classical']['cnn']['dropout']
    )
    
    trainer = ClassicalTrainer(model, config, logger)
    history = trainer.train(train_data, val_data)
    
    # Plot training curves
    # plot_training_curves(history, "CNN", dataset_name, dataset_size, plots_dir)
    
    predictions, labels = trainer.evaluate(test_data)
    
    return history, trainer, predictions, labels


def train_lstm_model(train_data, val_data, test_data, config, logger):
    """Train LSTM model."""
    
    model = LSTM(
        vocab_size=config['text']['vocab_size'],
        embedding_dim=config['models']['classical']['lstm']['embedding_dim'],
        hidden_dim=config['models']['classical']['lstm']['hidden_dim'],
        num_layers=config['models']['classical']['lstm']['num_layers'],
        num_classes=2,
        dropout=config['models']['classical']['lstm']['dropout']
    )
    
    trainer = ClassicalTrainer(model, config, logger)
    history = trainer.train(train_data, val_data)
    
    predictions, labels = trainer.evaluate(test_data)
    
    return history, trainer, predictions, labels


def train_mlp_model(train_data, val_data, test_data, config, logger, input_dim, num_classes):
    """Train MLP baseline for fair comparison with hybrid model."""
    mlp_cfg = config['models']['classical'].get('mlp', {})
    hidden1 = mlp_cfg.get('hidden1', 128)
    hidden2 = mlp_cfg.get('hidden2', 64)
    dropout = mlp_cfg.get('dropout', 0.3)

    model = MLP(
        input_dim=input_dim,
        hidden_dims=(hidden1, hidden2),
        num_classes=num_classes,
        dropout=dropout
    )

    trainer = ClassicalTrainer(model, config, logger)
    history = trainer.train(train_data, val_data)
    predictions, labels = trainer.evaluate(test_data)
    return history, trainer, predictions, labels


def train_hybrid_model(train_data, val_data, test_data, config, logger, input_dim):
    """Train Hybrid Quantum-Classical model."""
    hybrid_cfg = config['models']['hybrid_quantum']
    model = HybridQuantumClassical(
        input_dim=input_dim,
        num_classes=10,
        num_qubits=hybrid_cfg['num_qubits'],
        circuit_depth=hybrid_cfg['circuit_depth'],
        classical_hidden=hybrid_cfg['classical_hidden']
    )
    
    trainer = HybridTrainer(model, config, logger)
    history = trainer.train(train_data, val_data)
    
    predictions, labels = trainer.evaluate(test_data)
    
    return history, trainer, predictions, labels


def train_qsvm_model(train_data, val_data, test_data, config, logger):
    """Train Quantum Kernel SVM."""
    qsvm_cfg = config['models']['quantum_kernel']
    model = QuantumKernelSVM(
        num_qubits=qsvm_cfg['num_qubits'],
        circuit_depth=qsvm_cfg['circuit_depth'],
        C=1.0,
        kernel_type=qsvm_cfg['kernel_type']
    )
    
    trainer = QSVMTrainer(model, config, logger)
    history = trainer.train(train_data, val_data)
    
    predictions, labels = trainer.evaluate(test_data)
    
    return history, trainer, predictions, labels


def get_model_variants(config, model_key):
    """Return list of (name_suffix, model_cfg) variants for ablation studies."""
    base_cfg = dict(config['models'][model_key])
    if config.get('_disable_ablation_expansion'):
        return [("", base_cfg)]
    ablation_cfg = config.get('ablation', {}).get(model_key, {})

    qubits_list = ablation_cfg.get('num_qubits', [base_cfg.get('num_qubits')])
    depth_list = ablation_cfg.get('circuit_depth', [base_cfg.get('circuit_depth')])

    variants = []
    for num_qubits in qubits_list:
        for circuit_depth in depth_list:
            cfg_variant = dict(base_cfg)
            cfg_variant['num_qubits'] = num_qubits
            cfg_variant['circuit_depth'] = circuit_depth
            if num_qubits == base_cfg.get('num_qubits') and circuit_depth == base_cfg.get('circuit_depth'):
                suffix = f"[q{num_qubits},d{circuit_depth}]"
            else:
                suffix = f"[q{num_qubits},d{circuit_depth}]"
            variants.append((suffix, cfg_variant))
    return variants


def run_experiment(dataset_name, dataset_type, dataset_size, config, logger, all_results, seed):
    """Run experiment for a specific dataset and size."""
    
    # Load data
    if dataset_name == "MNIST":
        train_data, train_labels, test_data, test_labels = load_mnist_data()
    elif dataset_name == "FashionMNIST":
        train_data, train_labels, test_data, test_labels = load_fashion_mnist_data()
    elif dataset_name == "IMDB":
        train_data, train_labels, test_data, test_labels = load_imdb_data(config)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create small dataset
    train_subset, val_subset = create_small_dataset(train_data, train_labels, dataset_size)
    test_dataset = TensorDataset(test_data, test_labels)
    
    log_dataset_info(logger, dataset_name, dataset_size, len(train_subset), len(val_subset), len(test_dataset))
    
    # Determine number of classes
    num_classes = len(torch.unique(train_labels))
    models_to_run = config.get('models_to_run', {})
    qsvm_max_size = config.get('qsvm_max_size', 100)

    def _record_result(model_label, train_fn, *train_args, **train_kwargs):
        try:
            log_model_training_start(logger, model_label, dataset_name, dataset_size)
            history, trainer, predictions, labels = train_fn(*train_args, **train_kwargs)
            results = aggregate_results(
                history, trainer, predictions, labels,
                dataset_name, dataset_size, model_label, num_classes
            )
            results['seed'] = seed
            log_model_results(logger, results)
            all_results.append(results)
        except Exception as e:
            logger.error(f"Model failure [{model_label}] on {dataset_name} (n={dataset_size}, seed={seed}): {str(e)}")
            logger.error(traceback.format_exc())
    
    # Train models based on dataset type
    if dataset_type == "image":
        # CNN
        if models_to_run.get('cnn', True):
            _record_result("CNN", train_cnn_model, train_subset, val_subset, test_dataset, config, logger)

        # MLP (parameter-matched baseline for fairer comparison with Hybrid-QC)
        if models_to_run.get('mlp', True):
            _record_result(
                "MLP", train_mlp_model, train_subset, val_subset, test_dataset, config, logger,
                input_dim=28*28, num_classes=num_classes
            )
        
        # Hybrid Quantum-Classical
        if models_to_run.get('hybrid_qc', True):
            original_hybrid_cfg = dict(config['models']['hybrid_quantum'])
            try:
                for suffix, hybrid_variant in get_model_variants(config, 'hybrid_quantum'):
                    config['models']['hybrid_quantum'] = hybrid_variant
                    model_label = f"Hybrid-QC{suffix}" if suffix else "Hybrid-QC"
                    _record_result(
                        model_label, train_hybrid_model, train_subset, val_subset, test_dataset, config, logger,
                        input_dim=28*28
                    )
            finally:
                config['models']['hybrid_quantum'] = original_hybrid_cfg
        
        # Quantum Kernel SVM (only for smallest sizes due to computational cost)
        if models_to_run.get('qsvm', True) and dataset_size <= qsvm_max_size:
            original_qsvm_cfg = dict(config['models']['quantum_kernel'])
            try:
                for suffix, qsvm_variant in get_model_variants(config, 'quantum_kernel'):
                    config['models']['quantum_kernel'] = qsvm_variant
                    model_label = f"QSVM{suffix}" if suffix else "QSVM"
                    _record_result(model_label, train_qsvm_model, train_subset, val_subset, test_dataset, config, logger)
            finally:
                config['models']['quantum_kernel'] = original_qsvm_cfg
    
    elif dataset_type == "text":
        # LSTM
        if models_to_run.get('lstm', True):
            _record_result("LSTM", train_lstm_model, train_subset, val_subset, test_dataset, config, logger)


def _parse_model_variant_label(model_label):
    if '[' not in model_label or ']' not in model_label:
        return model_label, None, None
    family, suffix = model_label.split('[', 1)
    suffix = suffix.rstrip(']')
    num_qubits = None
    circuit_depth = None
    for part in suffix.split(','):
        part = part.strip()
        if part.startswith('q'):
            num_qubits = int(part[1:])
        elif part.startswith('d'):
            circuit_depth = int(part[1:])
    return family.rstrip(), num_qubits, circuit_depth


def _find_best_models_for_confusion(summary_df):
    """Pick best classical, QSVM, and Hybrid entries for MNIST n=100 by mean test accuracy."""
    target = summary_df[(summary_df['dataset'] == 'MNIST') & (summary_df['dataset_size'] == 100)]
    if target.empty:
        return {}

    def pick(mask):
        subset = target[mask].sort_values('test_accuracy_mean', ascending=False)
        return None if subset.empty else subset.iloc[0].to_dict()

    return {
        'classical': pick(target['model'].isin(['CNN', 'MLP'])),
        'qsvm': pick(target['model'].str.contains('QSVM', regex=False)),
        'hybrid': pick(target['model'].str.contains('Hybrid-QC', regex=False)),
    }


def generate_confusion_matrix_artifact(config, logger, summary_df, out_path, seed=0):
    """Train and evaluate best models on MNIST n=100 and plot confusion matrices using a common balanced test subset."""
    selections = _find_best_models_for_confusion(summary_df)
    if not selections or any(v is None for v in selections.values()):
        logger.warning("Skipping confusion matrix artifact: best-model selections not available for MNIST n=100.")
        return False

    set_seed(seed)
    train_data, train_labels, test_data, test_labels = load_mnist_data()
    train_subset, val_subset = create_small_dataset(train_data, train_labels, 100)
    # Use same balanced subset size policy as QSVM for comparability.
    subset_size = int(config.get('evaluation', {}).get('qsvm_test_subset', 500))
    X_test_subset, y_test_subset = create_balanced_subset_from_arrays(test_data, test_labels.numpy(), subset_size, seed)
    if torch.is_tensor(X_test_subset):
        test_subset_ds = TensorDataset(X_test_subset, torch.tensor(y_test_subset))
    else:
        test_subset_ds = TensorDataset(torch.tensor(X_test_subset), torch.tensor(y_test_subset))

    predictions_map = {}
    labels_ref = None

    # Classical best
    best_classical = selections['classical']
    if best_classical['model'] == 'CNN':
        _, _, preds, labels = train_cnn_model(train_subset, val_subset, test_subset_ds, config, logger)
    else:
        _, _, preds, labels = train_mlp_model(train_subset, val_subset, test_subset_ds, config, logger, input_dim=28*28, num_classes=10)
    predictions_map[f"Best Classical ({best_classical['model']})"] = preds
    labels_ref = labels

    # Hybrid best
    _, q_h, d_h = _parse_model_variant_label(str(selections['hybrid']['model']))
    original_hybrid = dict(config['models']['hybrid_quantum'])
    try:
        if q_h is not None:
            config['models']['hybrid_quantum']['num_qubits'] = q_h
        if d_h is not None:
            config['models']['hybrid_quantum']['circuit_depth'] = d_h
        _, _, preds, labels = train_hybrid_model(train_subset, val_subset, test_subset_ds, config, logger, input_dim=28*28)
        predictions_map[f"Best Hybrid ({selections['hybrid']['model']})"] = preds
        labels_ref = labels
    finally:
        config['models']['hybrid_quantum'] = original_hybrid

    # QSVM best
    _, q_k, d_k = _parse_model_variant_label(str(selections['qsvm']['model']))
    original_qsvm = dict(config['models']['quantum_kernel'])
    try:
        if q_k is not None:
            config['models']['quantum_kernel']['num_qubits'] = q_k
        if d_k is not None:
            config['models']['quantum_kernel']['circuit_depth'] = d_k
        _, _, preds, labels = train_qsvm_model(train_subset, val_subset, test_subset_ds, config, logger)
        predictions_map[f"Best QSVM ({selections['qsvm']['model']})"] = preds
        labels_ref = labels
    finally:
        config['models']['quantum_kernel'] = original_qsvm

    if labels_ref is None or not predictions_map:
        return False

    cm_payload = []
    for title, preds in predictions_map.items():
        cm_payload.append((title, confusion_matrix(labels_ref, preds, labels=np.arange(10))))
    plot_confusion_matrix_best_models(cm_payload, out_path)
    return True


def main():
    """Main execution function."""
    args = parse_args()
    legacy_mode = len(os.sys.argv) == 1

    # Create directories
    create_directories()

    # Load configuration
    config = load_config(args.config)
    config, run_opts = configure_run_from_args(config, args, legacy_mode)

    # Setup logger
    logger = setup_logger('results/logs')
    log_experiment_start(logger, config)

    # Storage for all results
    all_results = []

    try:
        if not legacy_mode:
            # Exclude synthetic IMDB from default paper runs unless explicitly requested.
            requested_names = {d['name'] for d in config['datasets']}
            if 'IMDB' not in requested_names:
                logger.info("IMDB synthetic path excluded from this run (paper-mode default).")

        if args.summary_only:
            results_df = pd.read_csv('results/metrics/all_results.csv')
        else:
            for seed in run_opts['requested_seeds']:
                config['random_seed'] = int(seed)
                set_seed(int(seed))
                logger.info(f"\n{'#'*80}\nRunning seed {seed}\n{'#'*80}")

                for dataset_config in config['datasets']:
                    dataset_name = dataset_config['name']
                    dataset_type = dataset_config['type']

                    for dataset_size in config['dataset_sizes']:
                        logger.info(f"\n{'='*80}\n")
                        logger.info(f"Processing {dataset_name} with size {dataset_size} (seed={seed})")
                        try:
                            run_experiment(
                                dataset_name, dataset_type, dataset_size,
                                config, logger, all_results, seed=int(seed)
                            )
                        except Exception as e:
                            logger.error(f"Error in experiment {dataset_name} - {dataset_size} - seed {seed}: {str(e)}")
                            logger.error(traceback.format_exc())

            results_df = pd.DataFrame(all_results)
            if not results_df.empty:
                results_df.to_csv('results/metrics/all_results.csv', index=False)
                logger.info("\nSaved results to results/metrics/all_results.csv")

        if not results_df.empty:
            summary_df = summarize_mean_std(results_df)
            summary_df.to_csv('results/metrics/summary_mean_std.csv', index=False)
            logger.info("Saved results summary to results/metrics/summary_mean_std.csv")

            logger.info("\nGenerating paper-grade plots...")
            generate_paper_plots(results_df, summary_df, 'results/plots')
            confusion_path = os.path.join('results', 'plots', 'confusion_matrix_best_models.png')
            confusion_created = generate_confusion_matrix_artifact(copy.deepcopy(config), logger, summary_df, confusion_path, seed=0)
            if not confusion_created:
                logger.warning("Confusion matrix figure was not generated.")
            logger.info("Plots saved to results/plots/")

        log_experiment_end(logger)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
