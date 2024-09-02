# QumranNLP
## Development
Create a new env (however you want, here a suggestion):
```shell
cd /path/to/cloned/repo
pyenv virtualenv 3.10.0 QumranNLP
pyenv activate QumranNLP
pip install -r requirements.txt
```

## Reproducibility
Reproducibility in the academic workd is pretty sucks. Ive tried to make this whole repo reproducible as much as i can.
If you are reading this, you are probably wish to use the Qumran data, so the data pipelines will be pretty easy to use (`src/ETL/main_ETL.py`).
In case you are here for the machine learning, algorithms or visualizations, Ive tried to make it robust as i can to the data.
In any case, i will post a guide so how to use your own data in the future.



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


## Running Research
### Topic modeling
After trying multiple methods for getting the optimal number of topics 
( LDA with coherence and perplexity, NMF optimization by Gal Gilad method, HDP), we decided that the optimal number is somewhere between 10-20.
For now, we will proceed without it.

### Global tuning params
Two different researches for determine the optimal `chunk_size` and the `pre_processing` scheme.
For evaluating each parameter, we checked those scheme on supervised and unsupervised classification for the scroll and composition level.
That means running the `src/ETL/main_ETL.py` for generating data, and then running `make_baselines_results` for each of the tasks (`src/baselines/main.py`).
* Chunk size research: [1.1-select_best_chunk_size.ipynb](notebooks%2F1.1-select_best_chunk_size.ipynb) (code in branch `new-chunking-scheme` )
* Pre processing research: [1.1-select_best_pre_processing_scheme.ipynb](notebooks%2F1.1-select_best_pre_processing_scheme.ipynb) (code in branch `create-pre-processing-schemes-19-08` )




## Running Tasks:
 

Tasks:
* Secterian classification - 
  * Update results: one row per composition
  * Interactive plots for UMAP with plotly

* Clustering at the scroll level:
  * In any case, before starting, consult with Roded. 
  * We have a few scrolls of interest: 1QS, Hodayot and 1QM.
  * Each scroll can be segmented to different parts (we have labels - mail from Jonathan).
  * We wish to find the best algorithm for clustering those scroll to the right segments.
  * First i will show the results with some naive methods (results means: dasgupta score, dendogram):
    * Using bert embeddings + agglomerate clustering.
    * Using different embeddings + different clustering.
  * We want to show that we can improve those results with a GNN. need to think how to do it. what i have in mind:
    * Check some methods for unsupervisd clustering with GNN.
    * Graph reconstruction from GNN outputs.
    * Check the review article for semi-supervised learning with GNN.
  * 
----- 
Could be nice in the future:
* Unsupervised gnn.
* Guide on how to use your own data (not Qumran).
* Medium posts (unsupervised clustering).

More optional things to consider:
1. More open questions from Jonathan:
Clustering of Serech Hayachad (חוקים, שירה וכו)
Clustering for CD.
Check if there different מזמורים in Hodayot.
Is Temple Scroll is sectarian?
Compare Instruction and Mysteries.
Compare pesharim and catena florilegium.
