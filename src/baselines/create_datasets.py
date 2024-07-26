import pickle

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


class QumranDataset:
    def __init__(self, df, label, train_frac, val_frac, processed_vectorizers):
        self.label = label
        self.df = self.process_df_by_label(df)
        self.texts = df["text"].tolist()
        self.labels = self.df[self.label]
        self.n_labels = self.labels.nunique()
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = 1 - train_frac
        self.processed_vectorizers = processed_vectorizers
        self.embeddings = None
        self.split_data()

    def split_data(self, random_state=42):
        self.df = self.df.reset_index(drop=True)
        train, test = train_test_split(
            self.df,
            test_size=self.test_frac,
            random_state=random_state,
            stratify=self.df[self.label],
        )
        train, val = train_test_split(
            train,
            test_size=self.val_frac,
            random_state=random_state,
            stratify=train[self.label],
        )
        train_mask, val_mask, test_mask = self.create_masks(
            len(self.df),
            train.index.to_list(),
            val.index.to_list(),
            test.index.to_list(),
        )
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    def split_data_for_cross_validation(self, n_splits=5, random_state=42):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        X = self.df.index.values
        y = self.df[self.label].values

        for train_index, val_index in skf.split(X, y):
            train_mask, val_mask, _ = self.create_masks(
                len(self.df),
                train_index.tolist(),
                val_index.tolist(),
                []
            )
            yield train_mask, val_mask


    def create_masks(self, data_length, train_indices, val_indices, test_indices):
        train_mask = np.zeros(data_length, dtype=bool)
        val_mask = np.zeros(data_length, dtype=bool)
        test_mask = np.zeros(data_length, dtype=bool)

        # Efficiently set elements to True using vectorized assignment
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        return train_mask, val_mask, test_mask

    def process_df_by_label(self, df):
        df["original_index"] = range(len(df))
        if self.label == "sectarian":
            df = df[df[self.label] != "unknown"]

        df = df.dropna(subset=[self.label])
        self.relevant_idx_to_embeddings = df["original_index"].to_list()
        return df

    def load_embeddings(self, vectorizer_type):
        embeddings = self.processed_vectorizers[vectorizer_type][
            self.relevant_idx_to_embeddings
        ]
        self.embeddings = embeddings
        return embeddings

    def __repr__(self):
        return f"num chunks: {self.df.shape[0]}, task:{self.label}"

    def __getstate__(self):
        # Return the state of the object for pickling
        state = self.__dict__.copy()
        # Ensure processed_vectorizers is serializable
        state["processed_vectorizers"] = {
            key: vectorizer for key, vectorizer in self.processed_vectorizers.items()
        }
        return state

    def __setstate__(self, state):
        # Restore the state of the object
        self.__dict__.update(state)


def save_dataset_for_finetuning(path, dataset):
    dataset_for_fine_tuning = {
        "train": dataset.df.iloc[dataset.train_mask, :],
        "val": dataset.df.iloc[dataset.val_mask, :],
        "test": dataset.df.iloc[dataset.test_mask],
    }
    with open(path, "wb") as f:
        pickle.dump(dataset_for_fine_tuning, f)
