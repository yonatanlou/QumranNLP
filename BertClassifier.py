import torch
from torch import nn
from transformers import BertModel, BertTokenizer
import numpy as np
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('onlplab/alephbert-base')
        # self.bert.required_grad_(False)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.softmax(linear_output)
        return final_layer


class Dataset(torch.utils.data.Dataset):
    tokenizer = BertTokenizer.from_pretrained('onlplab/alephbert-base')

    def __init__(self, text_list, label_list):
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.labels = label_list
        self.texts = [self.tokenizer(text, padding='max_length', max_length=200, truncation=True,
                                     return_tensors="pt") for text in text_list]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        batch_texts = self.texts[idx]
        batch_y = np.array(self.labels[idx])
        return batch_texts, batch_y


def train(model, train, val, learning_rate, epochs):
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=32)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0

        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input.data['attention_mask'].to(device)
            input_id = train_input.data['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            batch_loss = criterion(output, train_label)
            total_loss_train += batch_loss.item()

            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():

            for val_input, val_label in val_dataloader:
                val_label = val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

        print(
            f'Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Val Loss: {total_loss_val / len(val_data): .3f} \
                | Val Accuracy: {total_acc_val / len(val_data): .3f}')

#
# EPOCHS = 5
# model = BertClassifier()
# tokenizer = BertTokenizer.from_pretrained('onlplab/alephbert-base')
# LR = 1e-6
# section_type = ['sectarian_texts', 'non_sectarian_texts']
#
# train_size = 557
# val_size = 100
# test_size = 100
#
# train_data, train_label = np.array([]), np.array([], dtype=int)
# val_data, val_label = np.array([]), np.array([], dtype=int)
# test_data, test_label = np.array([]), np.array([], dtype=int)
#
# size = [0 for i in section_type]
# for i in range(len(section_type)):
#     all_data = np.array([])
#     all_labels = np.array([], dtype=int)
#     section = section_type[i]
#     data = parser_data.get_dss_data("data/yamls/all_sectarian_texts.yaml", section=section)
#     for book_name, book_data in data.items():
#         if len(book_data) < 100:
#             print(book_name)
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
#     test_data, test_label = np.concatenate((test_data, all_data[:test_size])), np.concatenate((test_label, all_labels[:test_size]))
#     val_data, val_label = np.concatenate((val_data, all_data[test_size: 2*test_size])), np.concatenate((val_label, all_labels[test_size: 2*test_size]))
#     train_data, train_label = np.concatenate((train_data, all_data[2*test_size:])), np.concatenate((train_label, all_labels[2*test_size:]))

# print(size)
# train_dataset = Dataset(train_data, train_label)
# val_dataset = Dataset(val_data, val_label)
# test_dataset = Dataset(test_data, test_label)
#
# train(model, train_dataset, val_dataset, LR, EPOCHS)

