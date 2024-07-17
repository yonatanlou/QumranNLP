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
1. Fine-tuning - finish fine tuning for composition classification, sectarian classification and MaskedLM.
2. Write the progress report for the ministry of science.
3. PPT for seminar.
2. Baselines - make sure i dont have data leakage. prepare a pipeline for all baselines.
3. GNN - using the same train test data for the GNN and see if im getting better at classification.
4. GNN - apply topic modeling graph, remove the bert graph.
5. GNN - apply GAT. 
6. Guide on how to use your own data (not Qumran).
7. Medium posts (unsupervised clustering).

More optional things to consider:
1. More open questions from Jonathan:
Clustering of Serech Hayachad (חוקים, שירה וכו)
Clustering for CD.
Check if there different מזמורים in Hodayot.
Is Temple Scroll is sectarian?
Compare Instruction and Mysteries.
Compare pesharim and catena florilegium.
