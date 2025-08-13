#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/Users/yonatanlou/dev/QumranNLP"
readonly dss_csv_name='data/processed_data/dss/df_CHUNK_SIZE=100_MAX_OVERLAP=0_PRE_PROCESSING_TASKS=[]_2024_02_09.csv'
readonly bible_csv_name='data/processed_data/bible/df_CHUNK_SIZE=100_MAX_OVERLAP=0_2024_15_01.csv'

###DSS
#run for creating the data:
python src/data_generation/dss_data_gen.py --chunk_size 100 --max_overlap 0 --pre_processing_tasks "[]" --output_file $dss_csv_name

# # run for baselines
python src/baselines/main.py --domain dss --results-dir experiments/dss/bert_cls/baselines --train-frac 0.7 --val-frac 0.1

# # run for supervised GNN
# python src/gnn/hyperparameter_gnn_main.py --datasets dataset_scroll --domain dss --num-combined-graphs 2 --exp-name gcn_init --results-dir experiments/dss/bert_cls/gnn --is_supervised

# # run for unsupervised GNN
python src/gnn/hyperparameter_gnn_main.py --datasets dataset_scroll --domain dss --num-combined-graphs 2 --exp-name gae_init --results-dir experiments/dss/bert_cls/gnn


# # run for producing global results
# python src/plots_generation/global_results_two_plots.py
# # run for producing clustering within scroll plots
# python src/plots_generation/clustering_within_scroll.py
# # run for producing sectarian clustering plots
# python src/plots_generation/sectarian_clustering.py

###Bible
#python src/data_generation/bible_data_gen.py --chunk_size 100 --max_overlap 10 --output_file $bible_csv_name
#python src/baselines/bible_baseline.py --domain bible --results-dir experiments/bible/baselines --train-frac 0.7 --val-frac 0.1
#python src/gnn/hyperparameter_gnn_main.py --datasets all --domain bible --num-combined-graphs 1 --exp-name gcn_init_v2 --results-dir experiments/bible/gnn --is_supervised
#python src/gnn/hyperparameter_gnn_main.py --datasets all --domain bible --num-combined-graphs 1 --exp-name gae_init_v2 --results-dir experiments/bible/gnn