import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import glob

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def plot_training_curves(history, model_name, dataset_name, dataset_size, save_dir):
    """Plot training and validation loss/accuracy curves."""
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{model_name} - Loss Curves\n{dataset_name} (n={dataset_size})', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title(f'{model_name} - Accuracy Curves\n{dataset_name} (n={dataset_size})', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f"{model_name}_{dataset_name}_{dataset_size}_curves.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_vs_size(results_df, save_dir):
    """Plot test accuracy vs dataset size for all models."""
    
    plt.figure(figsize=(12, 7))
    
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        for model in dataset_results['model'].unique():
            model_data = dataset_results[dataset_results['model'] == model]
            model_data = model_data.sort_values('dataset_size')
            
            plt.plot(model_data['dataset_size'], model_data['test_accuracy'], 
                    marker='o', linewidth=2, markersize=8, 
                    label=f"{dataset} - {model}")
    
    plt.xlabel('Training Dataset Size', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Test Accuracy vs Dataset Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    filepath = os.path.join(save_dir, 'accuracy_vs_size.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_f1_vs_size(results_df, save_dir):
    """Plot F1 score vs dataset size for all models."""
    
    plt.figure(figsize=(12, 7))
    
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        for model in dataset_results['model'].unique():
            model_data = dataset_results[dataset_results['model'] == model]
            model_data = model_data.sort_values('dataset_size')
            
            plt.plot(model_data['dataset_size'], model_data['test_f1_score'], 
                    marker='s', linewidth=2, markersize=8, 
                    label=f"{dataset} - {model}")
    
    plt.xlabel('Training Dataset Size', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1 Score vs Dataset Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    filepath = os.path.join(save_dir, 'f1_vs_size.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_overfitting_gap(results_df, save_dir):
    """Plot overfitting gap vs dataset size."""
    
    plt.figure(figsize=(12, 7))
    
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        for model in dataset_results['model'].unique():
            model_data = dataset_results[dataset_results['model'] == model]
            model_data = model_data.sort_values('dataset_size')
            
            plt.plot(model_data['dataset_size'], model_data['overfitting_gap'], 
                    marker='^', linewidth=2, markersize=8, 
                    label=f"{dataset} - {model}")
    
    plt.xlabel('Training Dataset Size', fontsize=12)
    plt.ylabel('Overfitting Gap (Train Acc - Val Acc)', fontsize=12)
    plt.title('Overfitting Gap vs Dataset Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    filepath = os.path.join(save_dir, 'overfitting_gap_vs_size.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_time(results_df, save_dir):
    """Plot training time vs dataset size."""
    
    plt.figure(figsize=(12, 7))
    
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        
        for model in dataset_results['model'].unique():
            model_data = dataset_results[dataset_results['model'] == model]
            model_data = model_data.sort_values('dataset_size')
            
            plt.plot(model_data['dataset_size'], model_data['training_time'], 
                    marker='D', linewidth=2, markersize=8, 
                    label=f"{dataset} - {model}")
    
    plt.xlabel('Training Dataset Size', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time vs Dataset Size', fontsize=14, fontweight='bold')
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    filepath = os.path.join(save_dir, 'training_time_vs_size.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_df, save_dir):
    """Plot model comparison across metrics."""
    
    # Group by model and compute mean across all datasets and sizes
    model_summary = results_df.groupby('model').agg({
        'test_accuracy': 'mean',
        'test_f1_score': 'mean',
        'training_time': 'mean',
        'peak_memory_mb': 'mean'
    }).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].bar(model_summary['model'], model_summary['test_accuracy'], 
                   color='steelblue', alpha=0.8)
    axes[0, 0].set_ylabel('Mean Test Accuracy', fontsize=11)
    axes[0, 0].set_title('Average Test Accuracy by Model', fontsize=12, fontweight='bold')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # F1 Score
    axes[0, 1].bar(model_summary['model'], model_summary['test_f1_score'], 
                   color='coral', alpha=0.8)
    axes[0, 1].set_ylabel('Mean F1 Score', fontsize=11)
    axes[0, 1].set_title('Average F1 Score by Model', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Training Time
    axes[1, 0].bar(model_summary['model'], model_summary['training_time'], 
                   color='lightgreen', alpha=0.8)
    axes[1, 0].set_ylabel('Mean Training Time (s)', fontsize=11)
    axes[1, 0].set_title('Average Training Time by Model', fontsize=12, fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Memory Usage
    axes[1, 1].bar(model_summary['model'], model_summary['peak_memory_mb'], 
                   color='mediumpurple', alpha=0.8)
    axes[1, 1].set_ylabel('Peak Memory (MB)', fontsize=11)
    axes[1, 1].set_title('Average Peak Memory by Model', fontsize=12, fontweight='bold')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filepath = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_time_tradeoff(results_df, save_dir):
    """Scatter plot of accuracy vs training time to visualize tradeoffs."""
    plt.figure(figsize=(11, 7))
    for dataset in results_df['dataset'].unique():
        dataset_results = results_df[results_df['dataset'] == dataset]
        for model in dataset_results['model'].unique():
            model_data = dataset_results[dataset_results['model'] == model]
            plt.scatter(
                model_data['training_time'],
                model_data['test_accuracy'],
                s=80,
                alpha=0.75,
                label=f"{dataset} - {model}"
            )

    plt.xscale('log')
    plt.xlabel('Training Time (s, log scale)', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Accuracy vs Training Time Tradeoff', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, loc='best')
    filepath = os.path.join(save_dir, 'accuracy_time_tradeoff.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()


def plot_qml_ablation_heatmaps(results_df, save_dir):
    """Heatmaps for QML models across qubits/depth if metadata is present."""
    required_cols = {'num_qubits', 'circuit_depth', 'test_accuracy', 'dataset', 'model'}
    if not required_cols.issubset(set(results_df.columns)):
        return

    qml_df = results_df[results_df['model'].str.contains('Hybrid-QC|QSVM', regex=True, na=False)].copy()
    if qml_df.empty:
        return

    qml_df = qml_df.dropna(subset=['num_qubits', 'circuit_depth'])
    if qml_df.empty:
        return

    qml_df['num_qubits'] = qml_df['num_qubits'].astype(int)
    qml_df['circuit_depth'] = qml_df['circuit_depth'].astype(int)
    qml_df['family'] = np.where(qml_df['model'].str.contains('QSVM'), 'QSVM', 'Hybrid-QC')

    for dataset in qml_df['dataset'].unique():
        for family in qml_df['family'].unique():
            subset = qml_df[(qml_df['dataset'] == dataset) & (qml_df['family'] == family)]
            if subset.empty:
                continue
            heat = subset.groupby(['num_qubits', 'circuit_depth'])['test_accuracy'].mean().reset_index()
            pivot = heat.pivot(index='num_qubits', columns='circuit_depth', values='test_accuracy')
            if pivot.empty:
                continue

            plt.figure(figsize=(6, 5))
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', cbar=True)
            plt.title(f'{family} Ablation Heatmap\n{dataset}', fontsize=12, fontweight='bold')
            plt.xlabel('Circuit Depth')
            plt.ylabel('Num Qubits')
            fname = f"ablation_heatmap_{family.lower().replace('-', '_')}_{dataset.lower()}.png"
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
            plt.close()


def generate_all_plots(results_df, save_dir):
    """Generate all plots."""
    
    plot_accuracy_vs_size(results_df, save_dir)
    plot_f1_vs_size(results_df, save_dir)
    plot_overfitting_gap(results_df, save_dir)
    plot_training_time(results_df, save_dir)
    plot_model_comparison(results_df, save_dir)
    plot_accuracy_time_tradeoff(results_df, save_dir)
    plot_qml_ablation_heatmaps(results_df, save_dir)


def _clean_pngs(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for path in glob.glob(os.path.join(save_dir, '*.png')):
        try:
            os.remove(path)
        except OSError:
            pass


def _plot_accuracy_vs_time_mean_std(summary_df, save_dir):
    if summary_df.empty:
        return
    df = summary_df.copy()
    needed = {'training_time_mean', 'test_accuracy_mean'}
    if not needed.issubset(df.columns):
        return

    plt.figure(figsize=(10, 7))
    for dataset in sorted(df['dataset'].dropna().unique()):
        subset_d = df[df['dataset'] == dataset]
        for model in sorted(subset_d['model'].dropna().unique()):
            rowset = subset_d[subset_d['model'] == model]
            x = rowset['training_time_mean'].values
            y = rowset['test_accuracy_mean'].values
            xerr = rowset['training_time_std'].fillna(0).values if 'training_time_std' in rowset else None
            yerr = rowset['test_accuracy_std'].fillna(0).values if 'test_accuracy_std' in rowset else None
            plt.errorbar(x, y, xerr=xerr, yerr=yerr, fmt='o', capsize=3, alpha=0.8, label=f'{dataset}-{model}')

    plt.xscale('log')
    plt.xlabel('Training Time Mean (s, log)')
    plt.ylabel('Test Accuracy Mean')
    plt.title('Accuracy vs Time (Mean ± Std)')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_vs_time_mean_std.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_qml_heatmap(summary_df, family, out_name, save_dir):
    if summary_df.empty:
        return
    df = summary_df.copy()
    if not {'model', 'num_qubits', 'circuit_depth', 'test_accuracy_mean'}.issubset(df.columns):
        return

    mask = df['model'].astype(str).str.contains(family, regex=False, na=False)
    qdf = df[mask & df['num_qubits'].notna() & df['circuit_depth'].notna()].copy()
    if qdf.empty:
        return

    qdf['num_qubits'] = qdf['num_qubits'].astype(int)
    qdf['circuit_depth'] = qdf['circuit_depth'].astype(int)
    heat = qdf.groupby(['num_qubits', 'circuit_depth'])['test_accuracy_mean'].mean().reset_index()
    pivot = heat.pivot(index='num_qubits', columns='circuit_depth', values='test_accuracy_mean')
    if pivot.empty:
        return

    plt.figure(figsize=(6, 5))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', cbar=True)
    plt.title(f'{family} Accuracy Heatmap (Mean over seeds/runs)')
    plt.xlabel('Circuit Depth')
    plt.ylabel('Num Qubits')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, out_name), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_runtime_scaling_qsvm(summary_df, save_dir):
    if summary_df.empty:
        return
    df = summary_df.copy()
    if not {'model', 'dataset', 'dataset_size', 'training_time_mean'}.issubset(df.columns):
        return
    qsvm = df[df['model'].astype(str).str.contains('QSVM', regex=False, na=False)].copy()
    if qsvm.empty:
        return

    # Aggregate over ablation settings to show practical scaling trend by n and dataset.
    agg = qsvm.groupby(['dataset', 'dataset_size'])[['training_time_mean']].mean().reset_index()
    agg_std = qsvm.groupby(['dataset', 'dataset_size'])[['training_time_mean']].std().reset_index().rename(
        columns={'training_time_mean': 'training_time_std_across_variants'}
    )
    agg = agg.merge(agg_std, on=['dataset', 'dataset_size'], how='left')

    plt.figure(figsize=(8, 6))
    for dataset in sorted(agg['dataset'].unique()):
        sub = agg[agg['dataset'] == dataset].sort_values('dataset_size')
        plt.errorbar(
            sub['dataset_size'],
            sub['training_time_mean'],
            yerr=sub['training_time_std_across_variants'].fillna(0),
            marker='o',
            capsize=4,
            linewidth=2,
            label=dataset
        )
    plt.xlabel('Training Dataset Size (n)')
    plt.ylabel('QSVM Training Time Mean (s)')
    plt.title('QSVM Runtime Scaling')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'runtime_scaling_qsvm.png'), dpi=150, bbox_inches='tight')
    plt.close()


def _plot_model_accuracy_bar(summary_df, save_dir):
    if summary_df.empty or 'test_accuracy_mean' not in summary_df.columns:
        return
    agg = summary_df.groupby('model')['test_accuracy_mean'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(agg.index), y=list(agg.values), color='steelblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Mean Test Accuracy')
    plt.title('Model Accuracy Summary (Mean Across Reported Runs)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'model_accuracy_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix_best_models(cm_payload, out_path):
    """Render three confusion matrices in one figure."""
    if not cm_payload:
        return
    n = len(cm_payload)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (title, cm) in zip(axes, cm_payload):
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=False, ax=ax)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_paper_plots(results_df, summary_df, save_dir):
    """Generate a minimal set of paper-ready plots and clean old plot files."""
    _clean_pngs(save_dir)
    _plot_accuracy_vs_time_mean_std(summary_df, save_dir)
    _plot_qml_heatmap(summary_df, 'QSVM', 'qml_heatmap_qsvm.png', save_dir)
    _plot_qml_heatmap(summary_df, 'Hybrid-QC', 'qml_heatmap_hybrid.png', save_dir)
    _plot_runtime_scaling_qsvm(summary_df, save_dir)
    _plot_model_accuracy_bar(summary_df, save_dir)
