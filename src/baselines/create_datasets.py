import pickle

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

MIN_SAMPLES_PER_CLASS = 4
CLASSES_TO_REMOVE_DUE_TO_COMPARING_CHUNK_SIZE = {
    "composition": ["the_rule_of_the_congregation", "Barkhi_Nafshi"],
    "book": [
        "4Q219",
        "11Q12",
        "11Q13",
        "4Q200",
        "4Q257",
        "4Q259",
        "1QSa",
        "4Q522",
        "4Q387",
        "4Q274",
        "4Q434",
        "4Q432",
        "4Q429",
        "4Q321a",
        "4Q422",
        "4Q158",
        "4Q161",
        "4Q366",
        "4Q415",
        "4Q368",
        "4Q400",
        "4Q397",
        "4Q365a",
        "4Q367",
    ],
    "section": ["notImplemented"],
}


class QumranDataset:
    def __init__(
        self,
        df,
        label,
        train_frac,
        val_frac,
        processed_vectorizers,
        specific_scrolls=None,
    ):
        print(f"Creating dataset - {label}...")
        self.label = label
        self.specific_scrolls = specific_scrolls
        df["original_index"] = range(len(df))
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

    def custom_split(self, samples, min_val_test_samples=1, random_state=42):
        """
        Split the samples into train, val, and test sets, ensuring at least 1 sample for val and test, and 2 for train.
        This function is designed for classes with a very low number of samples.
        For larger sample sizes, it uses a classic train-test split.

        Args:
            samples (list): List of sample indices to split.
            min_val_test_samples (int): Minimum number of samples for validation and test sets. Default is 1.
            random_state (int): Seed for random number generator. Default is 42.

        Returns:
            tuple: Three lists containing indices for train, validation, and test sets.

        Raises:
            ValueError: If there aren't enough samples to perform the split.
        """
        total_samples = len(samples)
        min_total_samples = (
            2 + 2 * min_val_test_samples
        )  # 2 for train, 1 each for val and test

        if total_samples < min_total_samples:
            raise ValueError(
                f"Not enough samples to split. Got {total_samples}, need at least {min_total_samples}."
            )

        # For larger sample sizes, use classic train-test split
        if total_samples >= int(1 / self.val_frac * self.train_frac):
            train_val, test = train_test_split(
                samples, test_size=1 - self.train_frac, random_state=random_state
            )
            train, val = train_test_split(
                train_val, test_size=self.val_frac, random_state=random_state
            )
            return train, val, test

        # Calculate ideal split based on ratio
        val_ratio = self.train_frac * self.val_frac
        ideal_train = int(total_samples * self.train_frac)
        ideal_val = int(total_samples * val_ratio)
        ideal_test = total_samples - ideal_train - ideal_val

        # Ensure minimum samples in each split
        train_samples = max(ideal_train, 2)  # At least 2 samples for train
        val_samples = max(ideal_val, min_val_test_samples)
        test_samples = max(ideal_test, min_val_test_samples)

        # Adjust if we've overallocated
        total_allocated = train_samples + val_samples + test_samples
        while total_allocated > total_samples:
            if train_samples > 2:
                train_samples -= 1
            elif test_samples > min_val_test_samples:
                test_samples -= 1
            elif val_samples > min_val_test_samples:
                val_samples -= 1
            total_allocated = train_samples + val_samples + test_samples

        # Shuffle the samples
        np.random.seed(random_state)
        shuffled_samples = np.random.permutation(samples)

        # Split the samples
        train = shuffled_samples[:train_samples]
        val = shuffled_samples[train_samples : train_samples + val_samples]
        test = shuffled_samples[train_samples + val_samples :]
        assert len(train) + len(val) + len(test) == total_samples

        return train, val, test

    def split_data(self, random_state=42):
        """
        Split the dataset into train, validation, and test sets.
        For classes with very low data, the custom_split function handles those classes,
        ensuring that in the worst case (4 samples per class), we have at least 2 samples in train
        and 1 sample each in validation and test.
        """
        self.df = self.df.reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No classes have enough samples for splitting.")

        # Create a new DataFrame with only classes that have enough samples
        initial_class_counts = self.df[self.label].value_counts()
        valid_classes = initial_class_counts[
            initial_class_counts >= MIN_SAMPLES_PER_CLASS
        ].index
        removed_classes = initial_class_counts[
            initial_class_counts < MIN_SAMPLES_PER_CLASS
        ]

        # Print information about removed classes
        if len(removed_classes) > 0:
            print("Classes removed due to insufficient samples:")
            for class_label, count in removed_classes.items():
                print(f"  - Class '{class_label}': {count} samples")
            total_removed_samples = removed_classes.sum()
            print(f"Total samples removed: {total_removed_samples}")
        else:
            print(
                f"No classes were removed. All classes have at least {MIN_SAMPLES_PER_CLASS} samples."
            )

        ## Only made for comparing chunk sizes
        # self.df = self.df[
        #     ~(
        #         self.df[self.label].isin(
        #             CLASSES_TO_REMOVE_DUE_TO_COMPARING_CHUNK_SIZE[self.label]
        #         )
        #     )
        # ]
        self.df = self.df[self.df[self.label].isin(valid_classes)]
        self.df = self.process_df_by_label(self.df)
        self.df = self.df.reset_index(drop=True)
        # Reset the DataFrame
        self.labels = self.df[self.label]
        self.n_labels = self.labels.nunique()

        print(
            f"Remaining classes (after removing nulls by label={self.label}: {self.n_labels}"
        )
        print(f"Remaining samples: {len(self.df)}")

        if len(self.df) == 0:
            raise ValueError(
                "No classes have enough samples for splitting after filtering."
            )

        train_indices, val_indices, test_indices = [], [], []

        for class_label in self.df[self.label].unique():
            class_data = self.df[self.df[self.label] == class_label]
            class_indices = class_data.index.tolist()
            train, val, test = self.custom_split(
                class_indices, random_state=random_state
            )

            train_indices.extend(train)
            val_indices.extend(val)
            test_indices.extend(test)

        self.train_mask, self.val_mask, self.test_mask = self.create_masks(
            len(self.df), train_indices, val_indices, test_indices
        )

        print(f"Final split sizes:")
        print(f"  - Train: {sum(self.train_mask)} samples")
        print(f"  - Validation: {sum(self.val_mask)} samples")
        print(f"  - Test: {sum(self.test_mask)} samples")

    #     def split_data(self, random_state=42):
    #
    #         # class_counts = self.df[self.label].value_counts()
    #         # min_samples = int(1 / min(self.test_frac, self.val_frac))
    #         # print(f"dropping {list(class_counts[class_counts < min_samples].index)} due to low number of samples")
    #         # valid_classes = class_counts[class_counts >= min_samples].index
    #         # self.df = self.df[self.df[self.label].isin(valid_classes)]
    #
    # #         custom_labels_to_remove = {"composition": ['Pseudo_Jeremiah',
    # #                                                    '4QMMT',
    # #                                                    'the_rule_of_the_blessings',
    # #                                                    'Berakhot',
    # #                                                    'the_rule_of_the_congregation',
    # #                                                    'Barkhi_Nafshi'], "book": ['4Q270', '1QpHab', '4Q317', '4Q249z', '4Q525', '4Q163', '4Q319', '4Q417', '4Q416', '4Q427', '4Q403', '4Q496', '4Q372', '11Q17', '4Q321', '4Q428', '3Q15', '4Q216', '4Q271', '4Q269', '4Q267', '4Q258', '4Q176', '4Q177', '4Q286', '4Q422', '4Q423', '4Q252', '4Q251', '4Q221', '4Q174', '4Q171', '4Q524', '4Q169', '11Q11', '1QSa', '1QSb', '4Q256', '4Q385a', '4Q394', '4Q379', '4Q265', '4Q378', '4Q391', '1Q22', '4Q320', '11Q12', '4Q321a', '11Q13', '4Q365a', '4Q366', '4Q522', '4Q367', '4Q161', '4Q158', '4Q400', '4Q219', '4Q200', '4Q259', '4Q434', '4Q432', '4Q429', '4Q261', '4Q387', '4Q274', '4Q257', '4Q415', '4Q397', '4Q368'],
    # #                                    "section": ["nothing"]
    # # }
    # #         print(f"df shape before removing samples {len(self.df)}")
    # #         self.df = self.df[~self.df[self.label].isin(custom_labels_to_remove[self.label])]
    # #         print(f"df shape after removing samples{len(self.df)}")
    # #         self.df = self.df[~self.df[self.label].isin(custom_labels_to_remove[self.label])]
    #         self.labels = self.df[self.label]
    #         self.n_labels = self.labels.nunique()
    #         self.df = self.process_df_by_label(self.df)
    #         self.df = self.df.reset_index(drop=True)
    #         if len(self.df) == 0:
    #             raise ValueError("No classes have enough samples for splitting.")
    #         train, test = train_test_split(
    #             self.df,
    #             test_size=self.test_frac,
    #             random_state=random_state,
    #             stratify=self.df[self.label],
    #         )
    #         train, val = train_test_split(
    #             train,
    #             test_size=self.val_frac,
    #             random_state=random_state,
    #             stratify=train[self.label],
    #         )
    #         train_mask, val_mask, test_mask = self.create_masks(
    #             len(self.df),
    #             train.index.to_list(),
    #             val.index.to_list(),
    #             test.index.to_list(),
    #         )
    #         self.train_mask = train_mask
    #         self.val_mask = val_mask
    #         self.test_mask = test_mask
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
        """
        The embeddings are pre-calculated per sample, here we are preserving the original index so we can
        get the relevant embedding per sample.
        """
        if self.label == "section":
            df = df[df[self.label] != "unknown"]
        if self.specific_scrolls:
            df = df[df["book"].isin(self.specific_scrolls)]
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
