import torch
from torch.nn import functional as F, Linear
from torch_geometric.nn import GCNConv, GATv2Conv, HeteroConv, SAGEConv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import warnings

from base_utils import measure_time

from src.baselines.ml import (
    get_clusterer,
    unsupervised_evaluation,
    unsupervised_optimization,
)
from torch.nn import Linear, Dropout, BatchNorm1d
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import global_mean_pool


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, dropout_rate=0.85):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # First layer with input-specific channel sizes
        first_conv = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')  # You can experiment with different aggregation methods
        self.convs.append(first_conv)
        self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 1):
            conv = HeteroConv({
                edge_type: SAGEConv((hidden_channels, hidden_channels), hidden_channels)
                for edge_type in metadata[1]
            }, aggr='sum')
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm1d(hidden_channels))

        # Dropout layer
        self.dropout = Dropout(p=dropout_rate)

        # Final layers
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.lin2 = Linear(hidden_channels // 2, out_channels)

        # Regularization
        self.l2_lambda = 1e-5

    def forward(self, x_dict, edge_index_dict):
        # Iterate through convolution layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x_dict = conv(x_dict, edge_index_dict)

            # Apply batch normalization and activation
            x_dict = {
                key: batch_norm(F.leaky_relu(x, negative_slope=0.1))
                for key, x in x_dict.items()
            }

            # Apply dropout
            x_dict = {
                key: self.dropout(x)
                for key, x in x_dict.items()
            }

        # Final classification layers
        x = x_dict['bert']  # Assuming 'author' is the target node type
        x = F.leaky_relu(self.lin1(x), negative_slope=0.1)
        x = self.dropout(x)
        x = self.lin2(x)

        return x

    def loss(self, pred, target):
        # Cross-entropy loss with L2 regularization
        ce_loss = F.cross_entropy(pred, target)

        # L2 regularization
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())

        return ce_loss + self.l2_lambda * l2_loss

class DMGI(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, num_relations):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels, out_channels) for _ in range(num_relations)])
        self.M = torch.nn.Bilinear(out_channels, out_channels, 1)
        self.Z = torch.nn.Parameter(torch.empty(num_nodes, out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.M.weight)
        self.M.bias.data.zero_()
        torch.nn.init.xavier_uniform_(self.Z)

    def forward(self, x, edge_indices):
        pos_hs, neg_hs, summaries = [], [], []
        for conv, edge_index in zip(self.convs, edge_indices):
            pos_h = F.dropout(x, p=0.5, training=self.training)
            pos_h = conv(pos_h, edge_index).relu()
            pos_hs.append(pos_h)

            neg_h = F.dropout(x, p=0.5, training=self.training)
            neg_h = neg_h[torch.randperm(neg_h.size(0), device=neg_h.device)]
            neg_h = conv(neg_h, edge_index).relu()
            neg_hs.append(neg_h)

            summaries.append(pos_h.mean(dim=0, keepdim=True))

        return pos_hs, neg_hs, summaries

    def loss(self, pos_hs, neg_hs, summaries,clustering_loss):
        loss = 0.
        for pos_h, neg_h, s in zip(pos_hs, neg_hs, summaries):
            s = s.expand_as(pos_h)
            loss += -torch.log(self.M(pos_h, s).sigmoid() + 1e-15).mean()
            loss += -torch.log(1 - self.M(neg_h, s).sigmoid() + 1e-15).mean()

        pos_mean = torch.stack(pos_hs, dim=0).mean(dim=0)
        neg_mean = torch.stack(neg_hs, dim=0).mean(dim=0)

        pos_reg_loss = (self.Z - pos_mean).pow(2).sum()
        neg_reg_loss = (self.Z - neg_mean).pow(2).sum()
        loss += 0.001 * (pos_reg_loss - neg_reg_loss)
        loss += clustering_loss*2
        print(f"0.001*({pos_reg_loss=}-{neg_reg_loss=})+ {clustering_loss=}*2 = {loss=}")

        return loss

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
def train(model, data, epochs, lr, patience=20, verbose=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    stats_best = []
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model = None
    patience_counter = 0

    model.train()
    for epoch in range(epochs + 1):
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(out[data["bert"].train_mask], data["bert"].y[data["bert"].train_mask])
        acc = accuracy(out[data["bert"].train_mask].argmax(dim=1), data["bert"].y[data["bert"].train_mask])
        loss.backward()
        optimizer.step()

        val_loss = criterion(out[data["bert"].val_mask], data["bert"].y[data["bert"].val_mask])
        val_acc = accuracy(out[data["bert"].val_mask].argmax(dim=1), data["bert"].y[data["bert"].val_mask])

        model.eval()
        test_loss = criterion(out[data["bert"].test_mask], data["bert"].y[data["bert"].test_mask])
        pred_y = out[data["bert"].test_mask].argmax(dim=1)
        true_y = data["bert"].y[data["bert"].test_mask]
        weighted_metrics = calculate_metrics(pred_y, true_y)
        test_acc = accuracy(out[data["bert"].test_mask].argmax(dim=1), data["bert"].y[data["bert"].test_mask])

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
# @measure_time
# def train(model, data, epochs, patience=20, verbose=True):
#     stats_best = []
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = model.optimizer
#
#     best_val_loss = float("inf")
#     best_model = None
#     patience_counter = 0
#
#     model.train()
#     for epoch in range(epochs + 1):
#         optimizer.zero_grad()
#         h, out = model(data.x, data.edge_index, data.edge_attr)
#         loss = criterion(out[data.train_mask], data.y[data.train_mask])
#         acc = accuracy(out[data.train_mask].argmax(dim=1), data.y[data.train_mask])
#         loss.backward()
#         optimizer.step()
#
#         val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
#         val_acc = accuracy(out[data.val_mask].argmax(dim=1), data.y[data.val_mask])
#
#         model.eval()
#         test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
#         pred_y = out[data.test_mask].argmax(dim=1)
#         true_y = data.y[data.test_mask]
#         weighted_metrics = calculate_metrics(pred_y, true_y)
#         test_acc = accuracy(out[data.test_mask].argmax(dim=1), data.y[data.test_mask])
#
#         if verbose:
#             if epoch % 10 == 0:
#                 print(
#                     f"Epoch {epoch:>3} | Train Loss: {loss:.3f} | Train Acc: {acc * 100:>6.2f}% | Val Loss: {val_loss:.2f} | Val Acc: {val_acc * 100:.2f}% | Test Loss: {test_loss:.2f} | Test Acc: {test_acc * 100:.2f}%"
#                 )
#
#         if val_loss < best_val_loss:
#             stats_best = []
#             best_val_loss = val_loss
#             best_model = model.state_dict()
#             patience_counter = 0
#             metrics = {
#                 "epoch": epoch,
#                 "train_loss": loss.item(),
#                 "train_acc": acc,
#                 "val_loss": val_loss.item(),
#                 "val_acc": val_acc,
#                 "test_loss": test_loss.item(),
#                 "test_acc": test_acc,
#             }
#             metrics.update(weighted_metrics)
#             stats_best.append(metrics)
#         else:
#             patience_counter += 1
#
#         if patience_counter >= patience:
#             print(f"Early stopping at epoch {epoch}")
#
#             break
#
#     model.load_state_dict(best_model)
#     print(
#         f'test_acc: {stats_best[0].get("test_acc")}, weighted_f1: {stats_best[0].get("weighted_f1")}'
#     )
#     return model, stats_best


@measure_time
def train_dmgi(
    model, data, optimizer, epochs, dataset, adjacency_matrix_tmp, verbose=True
):
    best_epoch = 0
    unsupervised_metric = "silhouette"
    best_stats = unsupervised_evaluation(
        dataset,  data['bert'].x, adjacency_matrix_tmp, clustering_algo="agglomerative"
    )
    best_model_state = None
    clustering_weight = 3
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        x = data['bert'].x
        edge_indices = data.edge_index_dict.values()
        pos_hs, neg_hs, summaries = model(x, edge_indices)
        with torch.no_grad():
            metrics = unsupervised_optimization(
                dataset, model.Z, clustering_algo="agglomerative"
            )
        clustering_loss = 1 - metrics[unsupervised_metric]
        loss = model.loss(pos_hs, neg_hs, summaries,clustering_loss)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            mu = model.Z
            metrics = unsupervised_evaluation(
                dataset, mu, adjacency_matrix_tmp, clustering_algo="agglomerative"
            )
        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3} | Loss: {loss:.3f} | | "
                + " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            )
        if metrics[unsupervised_metric] > best_stats[unsupervised_metric]:
            best_stats = metrics
            best_epoch = epoch
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model state loaded (epoch {best_epoch})")
    else:
        print("GNN didnt improve from epoch 0")

    print(best_stats)

    best_stats.update({"epoch": best_epoch})
    return model, [best_stats]

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
    clustering_weight = 3
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed_x, mu, logvar = model(data.x, data.edge_index, data.edge_attr)
        if (
            optimizer.defaults["lr"] < 0.001
        ):  # smaller loss needs smaller learning rates
            recon_loss = F.mse_loss(reconstructed_x, data.x, reduction="mean")
            kl_los = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        else:
            recon_loss = F.mse_loss(reconstructed_x, data.x, reduction="sum")
            kl_los = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        with torch.no_grad():
            metrics = unsupervised_optimization(
                dataset, mu, clustering_algo="agglomerative"
            )
        clustering_loss = 1 - metrics[unsupervised_metric]
        loss = recon_loss + kl_los + clustering_weight * clustering_loss
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            _, mu, _ = model(data.x, data.edge_index, data.edge_attr)
            metrics = unsupervised_evaluation(
                dataset, mu, adjacency_matrix_tmp, clustering_algo="agglomerative"
            )
        if verbose and epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3} | Loss: {loss:.3f} | recon_loss: {recon_loss:.3f} | kl_loss: {kl_los:.3f} | clustering_loss: {clustering_weight * clustering_loss:.3f} | "
                + " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
            )
        if metrics[unsupervised_metric] > best_stats[unsupervised_metric]:
            best_stats = metrics
            best_epoch = epoch
            best_model_state = model.state_dict()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model state loaded (epoch {best_epoch})")
    else:
        print("GNN didnt improve from epoch 0")

    print(best_stats)

    best_stats.update({"epoch": best_epoch})
    return model, [best_stats]


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

        # Add batch normalization layers
        self.bn_encoder = torch.nn.BatchNorm1d(latent_dim)
        self.bn_decoder = torch.nn.BatchNorm1d(hidden_dim)

        self.kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        }

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, edge_index, edge_attr):
        mu, logvar = self.encoder(x, edge_index, edge_attr)
        mu = self.bn_encoder(mu)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decoder(z, edge_index, edge_attr)
        return reconstructed_x, mu, logvar


class Encoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2_mu = GCNConv(hidden_dim, latent_dim)
        self.gc2_logvar = GCNConv(hidden_dim, latent_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        hidden = F.relu(self.gc1(x, edge_index, edge_attr))
        hidden = self.dropout(hidden)
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
