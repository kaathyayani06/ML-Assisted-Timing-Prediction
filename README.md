# Cross-Circuit Generalization of Graph Neural Networks for Timing Prediction: A Comparison with an MLP Baseline

## Project Overview

This project investigates the use of Graph Neural Networks (GNNs) for timing prediction in VLSI physical design and compares their performance with a Multi-Layer Perceptron (MLP) baseline.

The primary objective is to study how well these models generalize to previously unseen circuits and evaluate their ability to predict timing-related parameters, including arrival time and transformed slew.

The project explores whether graph-based learning can capture circuit connectivity and provide useful timing predictions across different circuit designs.

## Dataset

The experiments use benchmark circuit data from the [TimingPredict Dataset](https://github.com/TimingPredict/Dataset).

The dataset contains 10 benchmark circuits divided into training and testing sets.

### Training Circuits

- aes_cipher
- blabla
- des
- genericfir
- spm
- usb

### Testing Circuits

- BM64
- cic_decimator
- synth_ram
- zipdiv

The dataset is approximately 4.14 GB and is not hosted in this repository. Refer to `data/README.md` for dataset information and setup instructions.

## Models Implemented

### 1. Graph Neural Network (GNN)

The GNN model uses circuit graph information to learn relationships between interconnected nodes and predict timing-related parameters.

### 2. Multi-Layer Perceptron (MLP) Baseline

The MLP model uses node-level features without explicitly processing graph connectivity. It serves as a baseline for evaluating the contribution of graph-based learning.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures the magnitude of prediction errors.
- **R-squared (R²):** Measures how well the predictions explain the variation in actual values.
- **Rank Correlation:** Evaluates the agreement between predicted and actual timing rankings.

Evaluation includes individual output channels, grouped timing parameters, and held-out circuit performance.

## Repository Structure

```text
ML-Assisted-Timing-Prediction/
│
├── README.md
├── requirements.txt
│
├── model.py
├── data_graph.py
├── train_gnn.py
├── train_mlp_graph.py
├── evaluate_endpoint_models.py
├── evaluate_endpoint_models_pre_audit.py
├── evaluate_mlp_graph.py
├── tee.py
│
├── data/
│   └── README.md
│
└── checkpoints/
    └── README.md
```

## Python Files Description

| File | Description |
|---|---|
| `model.py` | Defines the GNN and MLP model architectures. |
| `data_graph.py` | Handles graph dataset loading and preprocessing. |
| `train_gnn.py` | Trains and evaluates the GNN model. |
| `train_mlp_graph.py` | Trains the MLP baseline. |
| `evaluate_endpoint_models.py` | Performs audited endpoint-level comparison between GNN and MLP. |
| `evaluate_endpoint_models_pre_audit.py` | Contains the earlier, pre-audit endpoint evaluation. |
| `evaluate_mlp_graph.py` | Evaluates the MLP model independently. |
| `tee.py` | Redirects terminal output to log files. |

## Environment Requirements

The experiments were developed using:

- Python 3.11.9
- PyTorch 2.1.2+CPU
- DGL 2.2.1
- NumPy 1.26.4
- SciPy 1.17.1

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Key Research Objective

The central focus is cross-circuit generalization: evaluating how well models trained on six circuits perform on four unseen circuits.

The study compares graph-based learning against a feature-based MLP baseline to understand the benefits and limitations of incorporating circuit connectivity into timing prediction.

## Acknowledgement

We acknowledge the authors of the [TimingPredict dataset and framework](https://github.com/TimingPredict/Dataset), which provide the foundation for this work.
