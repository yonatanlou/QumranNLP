# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
QumranNLP is a research project for authorial clustering of Qumran scrolls using semantic and statistical features. The project combines traditional NLP features with modern deep learning approaches including BERT embeddings and Graph Neural Networks (GNNs) for both supervised and unsupervised classification tasks.

## Environment Setup
```bash
# Create Python virtual environment (Python 3.10 recommended)
pyenv virtualenv 3.10.0 QumranNLP
pyenv activate QumranNLP
pip install -r requirements.txt

# Set PYTHONPATH for imports
export PYTHONPATH="${PYTHONPATH}:/path/to/QumranNLP"
```

## Common Commands

### Data Generation
```bash
# Generate DSS (Dead Sea Scrolls) dataset
python src/data_generation/dss_data_gen.py --chunk_size 100 --max_overlap 15 --pre_processing_tasks "[]" --output_file data/processed_data/dss/df_CHUNK_SIZE=100_MAX_OVERLAP=15_PRE_PROCESSING_TASKS=[]_2024_02_09.csv

# Generate Bible dataset
python src/data_generation/bible_data_gen.py --chunk_size 100 --max_overlap 10 --output_file data/processed_data/bible/df_CHUNK_SIZE=100_MAX_OVERLAP=10_2024_15_01.csv
```

### Baseline Experiments
```bash
# Run baseline classification experiments
python src/baselines/main.py --domain dss --results-dir experiments/dss/baselines --train-frac 0.7 --val-frac 0.1

# Run Bible baseline experiments
python src/baselines/bible_baseline.py --domain bible --results-dir experiments/bible/baselines --train-frac 0.7 --val-frac 0.1
```

### GNN Experiments
```bash
# Supervised GNN experiments
python src/gnn/hyperparameter_gnn_main.py --datasets dataset_scroll --domain dss --num-combined-graphs 2 --exp-name gcn_init --results-dir experiments/dss/gnn --is_supervised

# Unsupervised GNN experiments (GAE - Graph Auto-Encoder)
python src/gnn/hyperparameter_gnn_main.py --datasets dataset_scroll --domain dss --num-combined-graphs 2 --exp-name gae_init --results-dir experiments/dss/gnn
```

### Plot Generation
```bash
# Generate global comparison results
python src/plots_generation/global_results_two_plots.py

# Generate clustering within scroll plots
python src/plots_generation/clustering_within_scroll.py

# Generate sectarian clustering plots
python src/plots_generation/sectarian_clustering.py
```

### Full Experiment Pipeline
```bash
# Run complete experiment pipeline
./exp_runner.sh
```

## Architecture

### Core Components

**Data Processing Pipeline (`src/data_generation/`)**
- `dss_data_gen.py`: Main ETL for Dead Sea Scrolls data processing
- `bible_data_gen.py`: Bible dataset generation for validation
- Uses text-fabric package for text processing and chunking (default 100 words with 15 word overlap)

**Feature Engineering (`src/features/`)**
- `Starr/`: Implementation of Stamatatos Authorship Recognition features
- Combines traditional stylometric features with modern embeddings

**Baselines (`src/baselines/`)**
- `main.py`: Entry point for baseline experiments
- `embeddings.py`: Handles BERT embeddings (BEREL, DictaBERT, AlephBERT variants)
- `ml.py`: Machine learning models (LogisticRegression, LinearSVC, KNN, MLP)
- Supports supervised/unsupervised classification for scroll, composition, and sectarian tasks

**Graph Neural Networks (`src/gnn/`)**
- `model.py`: GCN and GAE implementations using PyTorch Geometric
- `adjacency.py`: Multiple adjacency matrix construction methods (tf-idf, trigram, starr features)
- `hyperparameter_gnn_main.py`: Hyperparameter tuning for GNN experiments
- Supports heterogeneous graphs with multiple edge types

**Visualization (`src/plots_generation/`)**
- `global_results_two_plots.py`: Comparative performance visualizations
- `clustering_within_scroll.py`: Within-scroll clustering analysis
- `sectarian_clustering.py`: Sectarian vs non-sectarian clustering

### Data Domains
- **DSS (Dead Sea Scrolls)**: Primary research data with sectarian/composition/scroll labels
- **Bible**: Hebrew Bible dataset used for validation and comparison

### Key Configuration Files
- `config.py`: Central configuration with data paths and domain-specific settings
- `base_utils.py`: Common utilities and decorators
- `logger.py`: Logging configuration

### Label Schemes
The project uses multiple labeling schemes stored in `data/qumran_labels.csv`:
- Sectarian classification (sectarian vs non-sectarian texts)
- Composition-level authorship
- Scroll-level clustering
- Genre classification

### Model Storage
- Trained GNN models stored in `models/` directory
- Pre-computed embeddings cached in `data/processed_data/{domain}/processed_vectorizers.pkl`
- Experiment results stored in `experiments/{domain}/` subdirectories

## Development Notes
- The codebase supports both DSS and Bible domains for comparative analysis
- GNN framework allows combining multiple adjacency matrix types
- All experiments use consistent train/validation/test splits with seed control
- Chunking strategy: 100-word segments with configurable overlap for stylometric analysis