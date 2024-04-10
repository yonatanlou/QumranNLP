from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import numpy as np
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from config import BASE_DIR
import pandas as pd
import os
from logger import get_logger
from src.features.BERT.bert import aleph_bert_preprocessing
from src.hierarchial_clustering.clustering_utils import generate_books_dict
from src.parsers import parser_data


class BertClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        # self.bert.required_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.softmax(linear_output)
        return final_layer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, text_list, label_list, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels = label_list
        self.texts = [
            self.tokenizer(
                text,
                padding="max_length",
                max_length=200,
                truncation=True,
                return_tensors="pt",
            )
            for text in text_list
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])
        return batch_texts, batch_y


def train(
    model,
    train_dataset,
    val_dataset,
    learning_rate,
    epochs,
    batch_size,
    run_name,
    logger,
    models_dir,
):
    writer = SummaryWriter(
        os.path.join(
            models_dir,
            run_name,
            f"{run_name}_tensorboard_{datetime.now().strftime('%Y-%m-%d')}.log",
        )
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_val_accuracy = 0
    for epoch_num in range(epochs):
        model.train()
        total_loss_train, total_acc_train = 0, 0
        for batch_idx, (batch_data, batch_labels) in enumerate(tqdm(train_dataloader)):
            batch_labels = batch_labels.to(device)
            mask = batch_data["attention_mask"].to(device)
            input_ids = batch_data["input_ids"].squeeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, mask)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss_train += loss.item()
            total_acc_train += (outputs.argmax(dim=1) == batch_labels).sum().item()

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"Epoch: {epoch_num+1}/{epochs}, Step: {batch_idx+1}/{len(train_dataloader)}, Loss: {total_loss_train / (batch_idx + 1):.4f}"
                )

        # Validation
        model.eval()
        total_loss_val, total_acc_val = 0, 0
        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                batch_labels = batch_labels.to(device)
                mask = batch_data["attention_mask"].to(device)
                input_ids = batch_data["input_ids"].squeeze(1).to(device)

                outputs = model(input_ids, mask)
                loss = criterion(outputs, batch_labels)

                total_loss_val += loss.item()
                total_acc_val += (outputs.argmax(dim=1) == batch_labels).sum().item()
        val_accuracy = total_acc_val / len(val_dataset)

        # Logging to TensorBoard
        writer.add_scalars(
            "Loss",
            {
                "Train": total_loss_train / len(train_dataset),
                "Validation": total_loss_val / len(val_dataset),
            },
            epoch_num,
        )
        writer.add_scalars(
            "Accuracy",
            {"Train": total_acc_train / len(train_dataset), "Validation": val_accuracy},
            epoch_num,
        )

        logger.info(
            f"Epoch: {epoch_num + 1}, Training Loss: {total_loss_train / len(train_dataset):.4f}, Training Accuracy: {total_acc_train / len(train_dataset):.4f}, Validation Loss: {total_loss_val / len(val_dataset):.4f}, Validation Accuracy: {val_accuracy:.4f}"
        )

        # Checkpointing based on best validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model_path = os.path.join(
                models_dir,
                run_name,
                f"best_model_{run_name}_{datetime.now().strftime('%Y-%m-%d')}.pth",
            )
            torch.save(model.state_dict(), best_model_path)
            logger.info(
                f"New best model saved with validation accuracy: {best_val_accuracy:.4f}"
            )

    writer.close()
    logger.info("Training complete.")


#
# BATCH_SIZE = 32
# EPOCHS = 5
# model = BertClassifier()
# tokenizer = BertTokenizer.from_pretrained("onlplab/alephbert-base")
# LR = 1e-6
# section_type = ["bib", "nonbib"]
# # section_type = ['sectarian_texts', 'non_sectarian_texts']
# train_data, train_label = np.array([]), np.array([], dtype=int)
# val_data, val_label = np.array([]), np.array([], dtype=int)
# test_data, test_label = np.array([]), np.array([], dtype=int)
#
# size = [0 for i in section_type]
# for i in range(len(section_type)):
#     all_data = np.array([])
#     all_labels = np.array([], dtype=int)
#     section = section_type[i]
#     book_dict, book_to_section = generate_books_dict([None], "books_to_read.yaml")
#     data = parser_data.get_dss_data(book_dict, section)
#     for book_name, book_data in data.items():
#         if len(book_data) < 50:
#             print(f"{book_name} have less than 50 samples")
#             continue
#         book_scores = [section, book_name]
#         samples, sample_names = parser_data.get_samples(book_data)
#         preprocessed_samples = aleph_bert_preprocessing(samples)
#         labels = [i for _ in range(len(samples))]
#         all_data = np.concatenate((all_data, preprocessed_samples))
#         all_labels = np.concatenate((all_labels, labels))
#     size[i] = len(all_data)
#     idx = np.arange(len(all_data))
#     np.random.shuffle(idx)
#     test_size = int(len(all_data) * 0.15)
#     test_data, test_label = np.concatenate(
#         (test_data, all_data[:test_size])
#     ), np.concatenate((test_label, all_labels[:test_size]))
#     val_data, val_label = np.concatenate(
#         (val_data, all_data[test_size : 2 * test_size])
#     ), np.concatenate((val_label, all_labels[test_size : 2 * test_size]))
#     train_data, train_label = np.concatenate(
#         (train_data, all_data[2 * test_size :])
#     ), np.concatenate((train_label, all_labels[2 * test_size :]))
#
# print(size)
# train_dataset = Dataset(train_data, train_label)
# val_dataset = Dataset(val_data, val_label)
# test_dataset = Dataset(test_data, test_label)
#
# train(model, train_dataset, val_dataset, LR, EPOCHS, BATCH_SIZE)
