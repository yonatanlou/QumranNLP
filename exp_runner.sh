#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/Users/yonatanlou/dev/QumranNLP"
readonly csv_name=/Users/yonatanlou/dev/QumranNLP/data/processed_data/dss/df_CHUNK_SIZE=100_MAX_OVERLAP=15_PRE_PROCESSING_TASKS=[]_2024_02_09.csv
#data_generation runner for making the data:
#python data_generation/dss_data_gen.py --chunk_size 100 --max_overlap 15 --pre_processing_tasks "[]" --output_file $csv_name


# run for baselines
#python src/baselines/main.py --domain dss --results-dir experiments/baselines --train-frac 0.7 --val-frac 0.1

# run for supervised GNN
#python src/gnn/hyperparameter_gnn_main.py --dataset all --domain dss --num-combined-graphs 2 --exp-name gcn_init --results-dir experiments/gnn --is_supervised

# run for unsupervised GNN
python src/gnn/hyperparameter_gnn_main.py --dataset all --domain dss --num-combined-graphs 1 --exp-name gvae_init --results-dir experiments/gnn