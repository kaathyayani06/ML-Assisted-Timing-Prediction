# Cross-Circuit Generalization of Graph Neural Networks for Timing Prediction: A Comparison with an MLP Baseline

## 1. Project Overview

This project investigates machine learning-assisted timing prediction in VLSI physical design, focusing on the cross-circuit generalization capability of machine learning models.

Two models are implemented and evaluated:

- **Graph Neural Network (GNN):** Utilizes circuit graph connectivity and node-level timing features to predict timing parameters.
- **Multi-Layer Perceptron (MLP):** Uses node-level features as a baseline for comparison with the GNN model.

The objective is to analyse and compare their prediction performance on benchmark circuits that are not used during training.

## 2. Dataset

The experiments use benchmark circuit data from the open-source TimingPredict project.

**Dataset Source:** [TimingPredict Dataset](https://github.com/TimingPredict/Dataset)

A total of ten benchmark circuits are considered.

**Training Circuits (6):**
- aes_cipher
- blabla
- des
- genericfir
- spm
- usb

**Testing Circuits (4):**
- BM64
- cic_decimator
- synth_ram
- zipdiv

The testing circuits are held out during training to evaluate cross-circuit generalization.

## 3. Models Implemented

### 3.1 Graph Neural Network (GNN)

- Represents circuit information using graph structures.
- Utilizes graph connectivity and node-level features.
- Predicts eight timing-related target channels.
- Incorporates auxiliary net-delay and cell-delay training objectives.

### 3.2 Multi-Layer Perceptron (MLP) Baseline

- Uses ten node-level input features.
- Consists of five hidden layers.
- Predicts eight timing-related target channels.
- Serves as a baseline for comparison with the GNN model.

## 4. Evaluation Metrics

The models are evaluated using the following metrics:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Measures the magnitude of prediction errors.
- **R² Score:** Evaluates the goodness of fit.
- **Rank Correlation:** Measures the agreement between predicted and actual endpoint rankings.

Performance is analysed across individual benchmark circuits and timing target channels.

## 5. Repository Structure

```text
ML-Assisted-Timing-Prediction/
│
├── README.md
├── requirements.txt
│
├── data_graph.py
├── model.py
├── train_gnn.py
├── train_mlp_graph.py
├── evaluate_endpoint_models.py
├── evaluate_endpoint_models_pre_audit.py
├── evaluate_mlp_graph.py
└── tee.py
```

## 6. Environment Requirements

The project uses the following software environment:

| Component | Version |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.1.2+cpu |
| DGL | 2.2.1 |
| NumPy | 1.26.4 |
| SciPy | 1.17.1 |

## 7. Python Files Description

| File | Description |
|---|---|
| `data_graph.py` | Graph construction and feature preparation |
| `model.py` | GNN model implementation |
| `train_gnn.py` | GNN training |
| `train_mlp_graph.py` | MLP baseline training |
| `evaluate_endpoint_models.py` | Endpoint-level model evaluation |
| `evaluate_endpoint_models_pre_audit.py` | Pre-audit evaluation |
| `evaluate_mlp_graph.py` | MLP baseline evaluation |
| `tee.py` | Supporting utility script |

## 8. Key Observation

The experimental results indicate that model performance varies across timing target channels and benchmark circuits.

Neither model consistently outperforms the other across all evaluation cases. This highlights the importance of analysing prediction performance at both circuit and timing-channel levels when evaluating cross-circuit generalization.

## 9. Acknowledgement

We acknowledge the original TimingPredict project for providing the open-source benchmark dataset and reference resources used in this work.

**Dataset:** https://github.com/TimingPredict/Dataset
 
