import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
import warnings


class GCN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, lr):
        super().__init__()
        torch.manual_seed(42)
        self.kwargs = {"dim_in": dim_in, "dim_h": dim_h, "dim_out": dim_out, "lr": lr}
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_h)
        self.gcn3 = GCNConv(dim_h, dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

    def forward(self, x, edge_index, edge_attr):
        h = self.gcn1(x, edge_index, edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.gcn2(h, edge_index, edge_attr)
        h = torch.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)

        h = self.gcn3(h, edge_index, edge_attr)
        return h, F.log_softmax(h, dim=1)

    def get_embeddings(self, x, edge_index, edge_attr):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            h = F.dropout(x, p=0.5, training=False)
            h = self.gcn1(h, edge_index, edge_attr)
            h = torch.relu(h)

            return h


class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out, lr, heads=8):
        super().__init__()
        torch.manual_seed(42)
        self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
        self.gat2 = GATv2Conv(dim_h * heads, dim_out, heads=1)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=5e-4)

    def forward(self, x, edge_index):
        h = F.dropout(x, p=0.5, training=self.training)
        h = self.gat1(h, edge_index)
        h = F.elu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.gat2(h, edge_index)
        return h, F.log_softmax(h, dim=1)


def accuracy(y_pred, y_true):
    """Calculate accuracy."""
    return torch.sum(y_pred == y_true) / len(y_true)


def accuracy(pred_y, y):
    return ((pred_y == y).sum() / len(y)).item()


def calculate_metrics(y_pred, y_test):
    warnings.filterwarnings(
        "ignore"
    )  # annoying warning UndefinedMetricWarning: Precision is ill-defined and being set to 0.0

    metric_type = ["micro", "macro", "weighted"]
    weighted_metrics = {}
    for metric in metric_type:
        precision = precision_score(y_test, y_pred, average=metric)
        recall = recall_score(y_test, y_pred, average=metric)
        f1 = f1_score(y_test, y_pred, average=metric)
        weighted_metrics[f"{metric}_precision"] = precision
        weighted_metrics[f"{metric}_recall"] = recall
        weighted_metrics[f"{metric}_f1"] = f1

    warnings.filterwarnings("default")
    return weighted_metrics


def train(model, data, epochs, patience=20, verbose=True):
    stats_best = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0

    model.train()
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        h, out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
        val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])

        model.eval()
        test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
        pred_y = out[data.test_mask].argmax(dim=1)
        true_y = data.y[data.test_mask]
        weighted_metrics = calculate_metrics(pred_y, true_y)
        test_acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])

        if verbose:
            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}% | Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%"
                )

        if val_loss < best_val_loss:
            stats_best = []
            best_val_loss = val_loss
            best_model = model.state_dict()
            patience_counter = 0
            metrics = {
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_acc": acc,
                "val_loss": val_loss.item(),
                "val_acc": val_acc,
                "test_loss": test_loss.item(),
                "test_acc": test_acc,
            }
            metrics.update(weighted_metrics)
            stats_best.append(metrics)
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")

            break

    model.load_state_dict(best_model)
    print(
        f'test_acc: {stats_best[0].get("test_acc")}, weighted_f1: {stats_best[0].get("weighted_f1")}'
    )
    return model, stats_best


def test(model, data):
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    # print(classification_report(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], target_names=label_encoder.classes_))
    return acc
