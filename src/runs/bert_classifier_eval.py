import os

import numpy as np
from tqdm import tqdm

from config import BASE_DIR
import torch
import pandas as pd

from src.models.BertClassifier import Dataset, BertClassifier

RUN_NAME = "alephbert-base_sectarian_from_books_to_read_tf_data_sampled"
MODEL_NAME = "onlplab/alephbert-base"
models_dir = os.path.join(BASE_DIR, "models", "bert_classifier")
model = BertClassifier(MODEL_NAME)
model.load_state_dict(
    torch.load(
        f"{models_dir}/{RUN_NAME}/best_model_alephbert-base_sectarian_from_books_to_read_tf_data_sampled_2024-04-09.pth"
    )
)
test_data_np = np.load(f"{models_dir}/{RUN_NAME}/test_data.npy", allow_pickle=True)
test_labels_np = np.load(f"{models_dir}/{RUN_NAME}/test_label.npy", allow_pickle=True)

test_dataset = Dataset(test_data_np, test_labels_np, MODEL_NAME)
unique, counts = np.unique(test_dataset.classes(), return_counts=True)
print(f"{len(test_dataset)=}")
print(f"{dict(zip(unique, counts))=}")


def evaluate_model(model, test_dataset, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_labels = []

    with torch.no_grad():  # Disable gradient computation
        for i, (text, label) in enumerate(tqdm(test_dataset)):
            input_ids = text["input_ids"].to(device).squeeze(1)
            attention_mask = text["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)
            pred_label = torch.argmax(outputs, dim=1)

            predictions.append(pred_label.item())
            true_labels.append(label.item())

    # Convert predictions and true labels into a DataFrame
    df_results = pd.DataFrame({"True Label": true_labels, "Prediction": predictions})
    return df_results


# Assuming you have a `test_dataset` and a `model` loaded and ready to use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)  # Ensure the model is on the right device
df_results = evaluate_model(model, test_dataset, device)

# Display the DataFrame or save it to a CSV file
print(df_results.head())  # Display the first few rows
from sklearn.metrics import classification_report

print(classification_report(df_results["True Label"], df_results["Prediction"]))
df_results.to_csv(
    f"{models_dir}/{RUN_NAME}/test_predictions.csv", index=False
)  # Save to CSV if needed
