# radio-astronomy-anomaly-detection
Enhanced Isolation Forest algorithms for radio astronomy anomaly detection. Evaluates four variants on MeerKAT survey data using expert-validated ground truth. Provides statistical performance benchmarks for human-in-the-loop discovery systems.
# Enhanced Isolation Forest for Radio Astronomy Anomaly Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.XXXX/XXXXX-blue)](https://doi.org/10.XXXX/XXXXX)

## 📖 Overview

This repository contains the implementation and evaluation of enhanced Isolation Forest algorithms for detecting scientifically interesting anomalies in radio astronomy data. The research uses expert-validated data from the MeerKAT Galaxy Cluster Legacy Survey (MGCLS) to establish performance baselines for four algorithmic variants.

## 🚀 Quick Start

### Installation
```bash
gh repo clone CS-UWC/radio-astronomy-anomaly-detection
cd radio-astronomy-anomaly-detection
pip install -r requirements.txt
Basic Usage
python
from enhanced_isolation_forest import CorrectedIsolationForest
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load and preprocess your data
X = load_astronomical_data()  # Your feature data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize enhanced algorithm
model = CorrectedIsolationForest(
    n_estimators=100,
    max_samples=256, 
    max_depth=10,
    split_method='pooled_gain',  # Options: 'standard', 'pooled_gain', 'kurtosis'
    scoring='depth',             # Options: 'depth', 'hypervolume'
    random_state=42
)

# Train and get anomaly scores
model.fit(X_scaled)
scores = model.decision_function(X_scaled)
📊 Algorithm Variants
1. Standard Enhanced
Baseline Isolation Forest with random splits

Reference: Liu et al. (2008)

2. Pooled Gain Variant
Feature-aware splitting using variance (70%) and kurtosis (30%)

Reference: Cortes (2021)

3. Kurtosis-based Variant
Kurtosis-weighted feature selection

Targets heavy-tailed distributions

Reference: Cortes (2021)

4. Hypervolume Variant
Geometric volume-based scoring instead of path lengths

Inspired by PAC learning theory

Reference: Dhouib et al. (2023)

📁 Project Structure
text
radio-astronomy-anomaly-detection/
├── notebooks/                          # Jupyter notebooks for analysis
│   ├── 01_enhanced_evaluation.ipynb    # Main evaluation pipeline
│   ├── 02_cumulative_detection.ipynb   # Cumulative analysis
│   └── 03_performance_table.ipynb      # Performance metrics
├── src/                                # Source code
│   ├── enhanced_isolation_forest.py    # Core algorithm implementation
│   ├── evaluation_metrics.py           # Evaluation framework
│   └── data_processing.py              # Data loading and preprocessing
├── data/                               # Data directory (not included)
│   ├── reduced_features_evaluation_set.csv
│   └── protege_catalogue.csv
├── results/                            # Generated figures and results
│   ├── Figure_PR_Curves_Enhanced.png
│   ├── Figure_ROC_Curves_Enhanced.png
│   ├── Figure_3A_Cumulative_Detection_Full_Set.png
│   ├── Figure_3B_Cumulative_Detection_First_200.png
│   └── Figure_4_Performance_Table_Large.png
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
└── environment.yml                     # Conda environment
🎯 Key Results
Performance Summary (MGCLS Dataset)
Algorithm	Avg Precision	ROC AUC	Precision@10	Anomalies@50
Pooled Gain	0.201	0.653	0.600	16.0
Standard Enhanced	0.198	0.677	0.400	14.0
Hypervolume	0.184	0.674	0.400	15.0
Kurtosis-based	0.155	0.677	0.200	10.0
Key Findings
Pooled Gain variant achieves best early detection (P@10 = 0.600)

6x improvement over random selection in top-10 precision

All variants show comparable binary classification (AUC ≈ 0.67)

Feature-aware splitting improves early-stage detection efficiency

🔧 Installation Details
Using pip
bash
pip install -r requirements.txt
Using conda
bash
conda env create -f environment.yml
conda activate radio-anomaly-detection
Data Requirements
Place the following files in the data/ directory:

reduced_features_evaluation_set.csv - 52-dimensional feature vectors

protege_catalogue.csv - Expert human interest scores

📈 Usage Examples
Complete Evaluation Pipeline
python
from src.enhanced_isolation_forest import CorrectedIsolationForest
from src.evaluation_metrics import run_robust_evaluation
from src.data_processing import load_and_prepare_data

# Load MGCLS data
X, object_ids, true_labels, expert_scores, scaler = load_and_prepare_data()

# Run comprehensive evaluation
results = run_robust_evaluation(X, true_labels, n_trials=10)

# Generate publication-ready figures
generate_all_figures(results, true_labels)
Custom Dataset Evaluation
python
# Apply to your own astronomical data
your_features = load_your_astronomical_data()
your_labels = get_expert_scores()  # Optional for evaluation

model = CorrectedIsolationForest(split_method='pooled_gain')
model.fit(your_features)
anomaly_scores = model.decision_function(your_features)

# Rank sources by anomaly score
ranked_indices = np.argsort(anomaly_scores)[::-1]
top_anomalies = your_features[ranked_indices[:10]]
