# Quantum ML Benchmarking and Diagnostic System

This project benchmarks classical and quantum-integrated models on small-data tasks and explains *why* the quantum models succeed or fail using runtime, memory, and ablation diagnostics.

## What This Project Is (and Is Not)

- `Is`: a reproducible benchmarking + analysis framework for `CNN`, `MLP`, `LSTM`, `Hybrid-QC`, and `QSVM`
- `Is`: suitable for a final-year project/paper on fair QML evaluation and bottleneck analysis
- `Is not`: a claim of guaranteed quantum advantage over strong classical baselines

## Current Focus / Contributions

- Fairer image benchmarking with a new `MLP` baseline (`models/classical/mlp.py`)
- Hybrid QML training fix (gradient-safe batching in the VQC forward pass)
- Improved QSVM preprocessing:
  - train-fitted `PCA` to `num_qubits`
  - train-fitted scaling to `[0, pi]`
  - consistent val/test transforms
- Faster QSVM kernel computation:
  - QNode reuse
  - symmetric kernel matrix optimization
  - optional balanced test subset for evaluation
- QML ablation support (`num_qubits`, `circuit_depth`) from YAML config
- Tradeoff and ablation plots (when metadata is available)

## Project Structure

```text
models/
  classical/
    cnn.py
    mlp.py
    lstm.py
  hybrid_quantum/
    hybrid_model.py
    vqc.py
  quantum_kernel/
    qsvm.py
training/
  trainer_classical.py
  trainer_hybrid.py
  trainer_qsvm.py
evaluation/
  metrics.py
  plots.py
  logger.py
config.yaml
config_fast.yaml
config_qsvm_minimal.yaml
main.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Config Files

- `config.yaml`: default benchmark
- `config_fast.yaml`: quick iteration (QSVM disabled)
- `config_qsvm_minimal.yaml`: QSVM smoke-test mode (reduced complexity)

## Key Config Options

- `models_to_run`: enable/disable `cnn`, `mlp`, `lstm`, `hybrid_qc`, `qsvm`
- `qsvm_max_size`: only run QSVM up to a dataset size threshold
- `evaluation.qsvm_test_subset`: balanced test subset size for QSVM evaluation speed
- `ablation.hybrid_quantum.*`: lists of qubits/depth values to sweep
- `ablation.quantum_kernel.*`: lists of qubits/depth values to sweep

## Outputs

- `results/metrics/all_results.csv`: metrics + runtime/memory + (optional) ablation metadata
- `results/plots/`: plots including:
  - accuracy / F1 vs dataset size
  - overfitting gap
  - training time
  - model comparison
  - accuracy-time tradeoff
  - QML ablation heatmaps (if ablation metadata exists)
- `results/logs/`: run logs

## Important Scientific Notes

- Quantum models here run on classical simulators (PennyLane), not real quantum hardware.
- Real hardware is not guaranteed to improve results; noise often makes performance worse.
- This benchmark is designed to support honest conclusions about QML bottlenecks and tradeoffs.
