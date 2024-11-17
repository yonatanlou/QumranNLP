import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F

# from torch_geometric import nn
from torch_geometric.nn import GCNConv, GATv2Conv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    silhouette_score,
)
import warnings

from base_utils import measure_time
from src.baselines.create_datasets import QumranDataset
from src.baselines.features import get_linkage_matrix
from src.baselines.ml import (
    get_clusterer,
    unsupervised_evaluation,
    unsupervised_optimization,
)
from sknetwork.hierarchy import dasgupta_score as calculate_dasgupta_score
import scipy.sparse as sp

from src.constants import UNSUPERVISED_METRICS
from src.baselines.utils import calculate_jaccard_unsupervised


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


@measure_time
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


@measure_time
def train_gvae(
    model, data, optimizer, epochs, dataset, adjacency_matrix_tmp, verbose=True
):
    best_epoch = 0
    unsupervised_metric = "silhouette"
    best_stats = unsupervised_evaluation(
        dataset, data.x, adjacency_matrix_tmp, clustering_algo="agglomerative"
    )
    best_model_state = None
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed_x, mu, logvar = model(data.x, data.edge_index, data.edge_attr)
        loss = vae_loss(reconstructed_x, data.x, mu, logvar)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            _, mu, _ = model(data.x, data.edge_index, data.edge_attr)
        metrics = unsupervised_optimization(
            dataset, mu, clustering_algo="agglomerative"
        )

        if metrics[unsupervised_metric] >= best_stats[unsupervised_metric]:
            best_stats = metrics
            best_epoch = epoch
            best_model_state = model.state_dict()

        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3} | Loss: {loss:.3f} | "
                + " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            )
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model state loaded (epoch {best_epoch})")
    else:
        print("GNN didnt improve from epoch 0")
    model.eval()
    with torch.no_grad():
        _, mu, _ = model(data.x, data.edge_index, data.edge_attr)
        best_stats_all_metrics = unsupervised_evaluation(
            dataset, mu, adjacency_matrix_tmp, clustering_algo="agglomerative"
        )

    best_stats_all_metrics.update({"epoch": best_epoch})
    return model, [best_stats_all_metrics]


def test(model, data):
    model.eval()
    _, out = model(data.x, data.edge_index)
    acc = accuracy(out.argmax(dim=1)[data.test_mask], data.y[data.test_mask])
    # print(classification_report(data.y[data.test_mask], out.argmax(dim=1)[data.test_mask], target_names=label_encoder.classes_))
    return acc


class GVAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(GVAE, self).__init__()
        torch.manual_seed(42)
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        }

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, edge_index, edge_attr):
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z, edge_index, edge_attr)
        return reconstructed_x, mu, logvar


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2_mu = GCNConv(hidden_dim, latent_dim)
        self.gc2_logvar = GCNConv(hidden_dim, latent_dim)

    def forward(self, x, edge_index, edge_attr):
        hidden = F.relu(self.gc1(x, edge_index, edge_attr))
        mu = self.gc2_mu(hidden, edge_index, edge_attr)
        logvar = self.gc2_logvar(hidden, edge_index, edge_attr)
        return mu, logvar


class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.gc1 = GCNConv(latent_dim, hidden_dim)
        self.gc2 = GCNConv(hidden_dim, output_dim)

    def forward(self, z, edge_index, edge_attr):
        hidden = F.relu(self.gc1(z, edge_index, edge_attr))
        return self.gc2(hidden, edge_index, edge_attr)


def vae_loss(reconstructed_x, x, mu, logvar):
    reconstruction_loss = F.mse_loss(reconstructed_x, x, reduction="sum")
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + kl_divergence
