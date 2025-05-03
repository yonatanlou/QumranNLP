# QumranNLP
🎉 Our paper was accepted to the **ALP 2025** workshop at **NAACL 2025**!  
📄 [Integrating Semantic and Statistical Features for Authorial Clustering of Qumran Scrolls](https://aclanthology.org/2025.alp-1.2/)

## Development
Create a new env (however you want, here a suggestion):
```shell
cd /path/to/cloned/repo
pyenv virtualenv 3.10.0 QumranNLP
pyenv activate QumranNLP
pip install -r requirements.txt
```

## Reproducibility
Reproducibility in the academic world is pretty sucks. Ive tried to make this whole repo reproducible as much as i can.
If you are reading this, you are probably wish to use the Qumran data, so the data pipelines will be pretty easy to use (`src/ETL/main_ETL.py`).
In case you are here for the machine learning, algorithms or visualizations, Ive tried to make it robust as i can to the data.
In any case, all of the scripts can be easily run with the `exp_runner.sh` script.



## Data
I'm using the [ETCBC/dss](https://github.com/ETCBC/dss/tree/master) package (built on [text-fabric](https://github.com/annotation/text-fabric/)).
This repo contains the original transcriptions by [Matrin Abegg](https://en.wikipedia.org/wiki/Martin_Abegg) as well (data/texts/abegg).

For generating data, you need to run the `src/main_ETL.py` script.
It will run over all of the scrolls (bib and nonbib), will generate starr features and save two dataframes:
1. Full data (no filtering).
2. Filtered data (you can specify which rules do you want to apply using the `filter_df_by_rules` function). The rules for now are:
   1. Books greater then 300 words.
   2. Hebrew books.
   3. Each book divided into 100 words chunks.





---
## Repo structure

├── data\
├── experiments\
├── models\
├── notebooks\
├── reports\
├── src\

1. Data - contains the most updated processed data (under processed_data), the qumran_labels.csv contains the labels by scrolls (by multiple labeling schemes: sectarian/composition/genre, etc...).
2. Experiments - contains the results of multiple experiments.
3. Models - contains some trained models (mainly GNN's, the fine-tuned models are stored in HF).
4. Notebooks - contains alot of research notebooks.
5. Reports - contains the results of most of the experiments.
6. Src - contains the code for the main ETL, feature engineering, experiments, model training.


---
## Running Research
### Topic modeling
After trying multiple methods for getting the optimal number of topics 
( LDA with coherence and perplexity, NMF optimization by Gal Gilad method, HDP), we decided that the optimal number is somewhere between 10-20.
For now, we will proceed without it.
![NMF topic modeling](reports/plots/nmf_topic_modeling.png "NMF topic modeling")


### Global tuning params
Two different researches for determine the optimal `chunk_size` and the `pre_processing` scheme.
For evaluating each parameter, we checked those scheme on supervised and unsupervised classification for the scroll and composition level.
That means running the `src/ETL/main_ETL.py` for generating data, and then running `make_baselines_results` for each of the tasks (`src/baselines/main.py`).
* Chunk size research: [1.1-select_best_chunk_size.ipynb](notebooks%2F1.1-select_best_chunk_size.ipynb) (code in branch `new-chunking-scheme` )
* Pre processing research: [1.1-select_best_pre_processing_scheme.ipynb](notebooks%2F1.1-select_best_pre_processing_scheme.ipynb) (code in branch `create-pre-processing-schemes-19-08` )

### Fine-tuning
I made the fine-tuning via masked LM scheme with 15% random masking. 
The code was run with colab [fine-tuning-bert-maskedLM.ipynb](https://colab.research.google.com/drive/1N60StbssmT7ssd7ykXP9apKdaVdBa8-7?usp=sharing) for the easy to use GPU 😅. 

### GNN
For implementing different structures in the GNN, ive created a framework which can combine different edge types together (heterogeneous graph didn't work so well, i believe this structure is too complex for this data).
So each node x is a chunk of text represented by a vector of dimension 768 (from different BERT models).
The edges can constructed via various methods, when the scheme is to define some feature space of the nodes, taking the cosine similiarity between each node, and taking only edges that are most similar (practically zeroing out the <0.99 quantile of the adj matrix).

#### Supervised classification
We can see that for the global tasks (scroll, composition and sectarian classification) the GNN always outperform the rest of the methods.
![Global tasks comparison supervised](reports/plots/global_results/dss_supervised_composition.png "Global tasks comparison")
![Global tasks comparison supervised](reports/plots/global_results/dss_supervised_Scroll.png "Global tasks comparison")



Interesting to see which types of adjacency matrices perform best:
![Different adj](experiments/dss/gnn/comparsion_plot_all_tasks_different_adj.png "Different adj")

#### Unsupervised classification
For the unsupervised setting, i've used the GAE (Graph auto-encoder) algorithm.
Its a classic encoder-decoder framework which can work with graphs.
Ive trained the model for each one of our potential embedders, when the graph was built using tf-idf/starr/trigram.
The clustering is made by hierarchical clustering (agglomerative), number of clusters as number of unique labels per task.
The following plots shows the difference between different models for the top per each metric:
![Global tasks comparison supervised](reports/plots/global_results/dss_unsupervised_composition.png "Global tasks comparison")

## Bible validation
I validate the results of the DSS with the Hebrew bible dataset (same chunking technique, embeddings, etc...).
The results are compared with the DSS results which is nice.
![Global tasks comparison supervised](reports/plots/global_results/bible_unsupervised_book.png "Global tasks comparison")



----- 
Tasks:
* Posts:
  * Easy implementation of GNN with supervised and unsupervised context. 
  * How to use GNN for text classification with different adj matrices.
