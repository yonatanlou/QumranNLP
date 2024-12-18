#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/Users/yonatanlou/dev/QumranNLP"
readonly dss_csv_name='data/processed_data/dss/df_CHUNK_SIZE=100_MAX_OVERLAP=15_PRE_PROCESSING_TASKS=[]_2024_02_09.csv'
readonly bible_csv_name='data/processed_data/bible/df_CHUNK_SIZE=100_MAX_OVERLAP=10_2024_15_01.csv'

###DSS
#run for creating the data:
#python data_generation/dss_data_gen.py --chunk_size 100 --max_overlap 15 --pre_processing_tasks "[]" --output_file $dss_csv_name

# run for baselines
#python src/baselines/main.py --domain dss --results-dir experiments/dss/baselines --train-frac 0.7 --val-frac 0.1

# run for supervised GNN
#python src/gnn/hyperparameter_gnn_main.py --datasets all --domain dss --num-combined-graphs 2 --exp-name gcn_init --results-dir experiments/dss/gnn --is_supervised

# run for unsupervised GNN
python src/gnn/hyperparameter_gnn_main.py --datasets dataset_composition,dataset_scroll --domain dss --num-combined-graphs 2 --exp-name gave_init --results-dir experiments/dss/gnn

# run for training all GVAE models

# run for producing clustering within scroll plots

# run for producing sectarian umap plots

###Bible
#run for creating the data:
#python src/data_generation/bible_data_gen.py --chunk_size 100 --max_overlap 10 --output_file $bible_csv_name

# run for baselines
#python src/baselines/bible_baseline.py --domain bible --results-dir experiments/bible/baselines --train-frac 0.7 --val-frac 0.1

# run for supervised GNN
#python src/gnn/hyperparameter_gnn_main.py --datasets all --domain bible --num-combined-graphs 1 --exp-name gcn_init --results-dir experiments/bible/gnn --is_supervised

# run for unsupervised GNN
#python src/gnn/hyperparameter_gnn_main.py --datasets all --domain bible --num-combined-graphs 1 --exp-name gave_init --results-dir experiments/bible/gnn