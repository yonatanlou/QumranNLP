# QumranNLP
## Development
Create a new env (however you want, here a suggestion):
```shell
cd /path/to/cloned/repo
pyenv virtualenv 3.10.0 QumranNLP
pip install -r requirements.txt
```

## Data
I'm using the [ETCBC/dss](https://github.com/ETCBC/dss/tree/master) package (built on [text-fabric](https://github.com/annotation/text-fabric/)).
This repo contains the original transcriptions by [Matrin Abegg](https://en.wikipedia.org/wiki/Martin_Abegg) as well (data/texts/abegg).




Running Tasks:


Data:
1. I need to decide in which data to use:
   1. Origin text (dss_bib/dss_nonbib) from abegg: 
   This is the original text by abegg, parsed by us.
      1. Pros: The code is done, can be used to get starr features. High quality.
      2. Cons: Will be hard to reproduce for other researchers, not updated. Not open source this code means that we could have some errors. Dont have lexemes.
   2. Text-fabric:
   They are using also the origin text (dss_bib/dss_nonbib) from abegg, but the parsing is by them [tfFromAbegg.py](https://github.com/ETCBC/dss/blob/master/programs/tfFromAbegg.py):
      1. Pros: Highly reproducible, Probably can be updated pretty easily. have lexemes.
      2. Cons: I need to re-code the starr features.

2. After deciding which one to use, i need to be able to extract the lexemes from this text.
3. EDA - duplicates of the scrolls.
Topic modeling:
1. Using NMF instead of LDA -  Gal Gilad on my web page.


Clustering:
1. Add the starr features to the unsupervised pipeline.
2. NMF.
2. Cluster by daniel method but between books (compositions), define different number of similiarties.
2. Construct a baseline.

Ask jonathan:
1. The decision to move forward with the text-fabric data.
2. What is lexeme.
3. Re-do topic modeling. for validation - what does he need?
4. Clusteroing what me and roded spoke yesterdau

Tasks:
1. Finish the EDA - Make sure the settings of the data are clear.
2. Re-organize the whole repo
3. start topic modeling from the begining
4. 