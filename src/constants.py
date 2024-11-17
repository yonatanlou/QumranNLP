BERT_MODELS = [
    "dicta-il/BEREL",
    "onlplab/alephbert-base",
    "dicta-il/dictabert",
    "dicta-il/MsBERT",
    "yonatanlou/BEREL-finetuned-DSS-maskedLM",
    "yonatanlou/alephbert-base-finetuned-DSS-maskedLM",
    "yonatanlou/dictabert-finetuned-DSS-maskedLM",
]

UNSUPERVISED_METRICS = ["jaccard", "dasgupta", "silhouette", "clustering_accuracy"]
DSS_OPTIONAL_DATASETS = ["dataset_composition", "dataset_scroll", "dataset_sectarian"]
BIBLE_OPTIONAL_DATASETS = ["dataset_book"]
OPTIONAL_DATASET_NAMES = {
    "dss": DSS_OPTIONAL_DATASETS,
    "bible": BIBLE_OPTIONAL_DATASETS,
}
