# Model Checkpoints

This directory contains the trained model checkpoints required for evaluating the timing prediction models.

## Required Checkpoints

### 1. GNN Model

- **File:** `99.pth`
- **Expected path:**

```text
checkpoints/experiment_6train4test_no_gt/99.pth
```

This checkpoint contains the trained Graph Neural Network model used for timing prediction.

### 2. MLP Baseline Model

- **File:** `best_model.pth`
- **Expected path:**

```text
checkpoints/mlp_graph_baseline/best_model.pth
```

This checkpoint contains the trained MLP baseline model and its normalization parameters.

## Important Notes

- Trained checkpoint files are not currently included in this repository.
- The files must be obtained separately and placed in their respective directories.
- Checkpoint files may be large, so they are not committed directly to GitHub at this stage.
- The evaluation scripts expect the checkpoint paths specified above.

