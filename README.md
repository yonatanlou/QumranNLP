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


**Running Research**:
Summary of topic modeling:
After trying multiple methods for getting the optimal number of topics 
( LDA with coherence and perplexity, NMF optimization by Gal Gilad method, HDP), we decided that the optimal number is somewhere between 10-20.
For now, we will proceed without it.

**Running Tasks**:


Tasks:
* Write the progress report for the ministry of science.
* PPT for seminar.
* GNN - make sure the afj_mat is good (1 on the diagonal)
* Make sure that normalizing is better than not.
* GNN - apply GAT.
* GNN - applied to sectarian classification all of the above.
* After showing what is the best model, can send this model results to Jonathan.
* Guide on how to use your own data (not Qumran).
* Medium posts (unsupervised clustering).

* Implementing unsupervised learning with GNN.

More optional things to consider:
1. More open questions from Jonathan:
Clustering of Serech Hayachad (חוקים, שירה וכו)
Clustering for CD.
Check if there different מזמורים in Hodayot.
Is Temple Scroll is sectarian?
Compare Instruction and Mysteries.
Compare pesharim and catena florilegium.
