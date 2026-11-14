# Cross-Circuit Generalization of Graph Neural Networks for Timing Prediction: A Comparison with an MLP Baseline

## Project Overview

This project focuses on machine learning-based timing prediction in VLSI physical design. It implements and compares two prediction models:

1. **Graph Neural Network (GNN):** Uses circuit connectivity and auxiliary net-delay and cell-delay training objectives.
2. **Multi-Layer Perceptron (MLP):** Uses node-level features to predict timing parameters.

The objective is to evaluate both models on different benchmark circuits and compare their prediction performance.

## Dataset

The project uses the open-source dataset from the TimingPredict project.

**Dataset Repository:**  
https://github.com/TimingPredict/Dataset

Ten benchmark circuits are used in the experiments.

**Training Circuits:**
- aes_cipher
- blabla
- des
- genericfir
- spm
- usb

**Test Circuits:**
- BM64
- cic_decimator
- synth_ram
- zipdiv

## Models Implemented

### 1. Graph Neural Network (GNN)
- Represents circuit data as graphs.
- Uses graph connectivity for timing prediction.
- Includes auxiliary net-delay and cell-delay training objectives.

### 2. MLP Baseline
- Uses ten node-level input features.
- Consists of five hidden layers.
- Predicts eight timing target channels.

## Evaluation Metrics

Both models are evaluated using:

- R² (Coefficient of Determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

Performance is analysed across all test circuits and individually for each circuit.

## Repository Contents

| File | Description |
|---|---|
| `data_graph.py` | Graph and feature construction |
| `model.py` | GNN model implementation |
| `train_gnn.py` | GNN training |
| `train_mlp_graph.py` | MLP training |
| `evaluate_endpoint_models.py` | Endpoint-level model evaluation |
| `evaluate_endpoint_models_pre_audit.py` | Pre-audit evaluation |
| `evaluate_mlp_graph.py` | MLP evaluation |
| `tee.py` | Supporting utility script |

## Key Observation

The comparison shows that model performance varies depending on the predicted timing target and the benchmark circuit. Neither model performs uniformly better across all evaluation cases.

## Acknowledgement

We acknowledge the original TimingPredict project for providing the dataset and reference resources used in this work.

**Dataset:** https://github.com/TimingPredict/Dataset
