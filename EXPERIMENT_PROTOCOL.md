# Experiment Protocol

## Goal

Evaluate classical and quantum-integrated models fairly in small-data regimes and diagnose bottlenecks using accuracy/cost/ablation analysis.

## Recommended Procedure

1. Start with `config_fast.yaml` to validate code changes quickly.
2. Run `config_qsvm_minimal.yaml` for QSVM smoke tests.
3. Run ablations:
   - `ablation.hybrid_quantum.num_qubits: [2, 4, 6]`
   - `ablation.hybrid_quantum.circuit_depth: [1, 2, 3]`
   - `ablation.quantum_kernel.num_qubits: [2, 4]`
   - `ablation.quantum_kernel.circuit_depth: [1, 2]`
4. Repeat with multiple seeds (`3` minimum, `5` preferred).
5. Report mean and standard deviation.

## Fairness Rules

- Keep dataset subsampling procedure fixed per seed.
- Use the same train/val split for all models within a run.
- Compare Hybrid-QC against both `CNN` and `MLP`.
- Record runtime and memory, not just accuracy.
- Document QSVM test subset size when used.

## Notes for Defense / Paper

- Simulator results do not imply hardware advantage.
- Negative results are valid when methodology is fair and reproducible.
- Report bottlenecks explicitly (feature compression, kernel cost, optimization instability).
