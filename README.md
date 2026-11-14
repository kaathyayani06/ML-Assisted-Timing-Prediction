# Cross-Circuit Generalization of Graph Neural Networks for Timing Prediction: A Comparison with an MLP Baseline 

## Project Overview

This project explores a Machine Learning-Assisted Timing Prediction Framework for VLSI Physical Design.

The main objective is to investigate the cross-circuit generalization capability of Graph Neural Networks (GNNs) for timing prediction and compare their performance with a Multi-Layer Perceptron (MLP) baseline.

The framework predicts timing-related parameters, including arrival time and transformed slew, for unseen benchmark circuits.

## Dataset

This project uses the **8_rat dataset**, based on the [TimingPredict Dataset](https://github.com/TimingPredict/Dataset).

The experiments use ten benchmark circuits divided into training and testing sets.

### Training Circuits

- `aes_cipher`
- `blabla`
- `des`
- `genericfir`
- `spm`
- `usb`

### Testing Circuits

- `BM64`
- `cic_decimator`
- `synth_ram`
- `zipdiv`

**Dataset availability:**

- Dataset files are not included in this repository because of their size.
- Refer to [`data/README.md`](data/README.md) for dataset setup instructions.

## Models Implemented

### 1. Graph Neural Network (GNN)

The GNN model uses graph connectivity and message passing to learn relationships between circuit components and predict timing-related parameters.

### 2. Multi-Layer Perceptron (MLP) Baseline

The MLP model uses node-level features without explicitly utilizing graph connectivity.

It serves as a baseline for comparing the performance of graph-based timing prediction.

## Evaluation Metrics

The models are evaluated using the following metrics:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)
- Rank Correlation (where calculated in the corresponding evaluation experiment)

The evaluation includes per-channel and grouped comparisons of arrival time and transformed slew.

The endpoint-level comparison uses a common set of valid endpoints for both models.

## Repository Structure

```text
ML-Assisted-Timing-Prediction/
│
├── data/
│   └── README.md
│
├── checkpoints/
│   └── README.md
│
├── data_graph.py
├── model.py
├── train_gnn.py
├── train_mlp_graph.py
├── evaluate_endpoint_models.py
├── evaluate_endpoint_models_pre_audit.py
├── evaluate_mlp_graph.py
├── tee.py
│
├── requirements.txt
└── README.md
```

### File Descriptions

| File / Directory | Description |
|---|---|
| `data/` | Dataset setup instructions |
| `checkpoints/` | Model checkpoint requirements |
| `data_graph.py` | Graph data loading and feature construction |
| `model.py` | GNN and related model architectures |
| `train_gnn.py` | GNN training and evaluation |
| `train_mlp_graph.py` | MLP baseline training |
| `evaluate_endpoint_models.py` | Audited endpoint-level GNN and MLP comparison |
| `evaluate_endpoint_models_pre_audit.py` | Earlier pre-audit endpoint evaluation |
| `evaluate_mlp_graph.py` | Standalone MLP evaluation |
| `tee.py` | Training log utility |
| `requirements.txt` | Python package dependencies |

## Environment Setup

The experiments use the following software environment:

- Python 3.11.9
- PyTorch 2.1.2+cpu
- DGL 2.2.1
- NumPy 1.26.4
- SciPy 1.17.1

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

Before running the scripts, ensure that the dataset and required model checkpoints are placed in their expected directories.

Refer to:

- [`data/README.md`](data/README.md)
- [`checkpoints/README.md`](checkpoints/README.md)

## Research Objective

The project investigates whether graph-based learning can improve timing prediction across different VLSI circuits compared with a feature-based MLP baseline.

The study focuses on evaluating model generalization to unseen circuits and analyzing prediction performance across different timing-related parameters.

## Acknowledgement

This project builds upon the concepts and dataset provided by the [TimingPredict research work](https://github.com/TimingPredict/Dataset).

