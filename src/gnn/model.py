import torch
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)
import warnings


from base_utils import measure_time
from config import BASE_DIR
from logger import get_logger

from src.baselines.ml import (
    unsupervised_evaluation,
)
from datetime import datetime

date = datetime.today().strftime("%Y-%m-%d")
logger = get_logger(__name__, f"{BASE_DIR}/logs/{__name__}_{date}.log")


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
def train_gcn(model, data, epochs, patience=20, verbose=True):
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
def train_gae(
    model, data, optimizer, epochs, dataset, adjacency_matrix_tmp, verbose=True
):
    counter = 0
    patience = 2
    min_delta = 0.001
    min_validation_loss = float("inf")

    best_epoch = 0
    unsupervised_metric = "auc"
    best_stats = unsupervised_evaluation(
        dataset, data.x, adjacency_matrix_tmp, clustering_algo="agglomerative"
    )
    best_stats.update({"auc": 0.0, "ap": 0.0})
    print(f"starting with {best_stats=}")
    best_model_state = None
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index, data.edge_attr)
        loss = model.recon_loss(z, data.edge_index)
        if hasattr(model, "kl_loss"):
            loss = loss + (1 / data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            model.eval()
            z = model.encode(data.x, data.edge_index, data.edge_attr)
            neg_edge_index = negative_sampling(data.edge_index, z.size(0))
            auc, ap = model.test(z, data.edge_index, neg_edge_index)
            metrics = unsupervised_evaluation(
                dataset, z, adjacency_matrix_tmp, clustering_algo="agglomerative"
            )
            metrics["auc"] = auc
            metrics["ap"] = ap

            print(
                f"epoch: {epoch} | loss: {loss :.3f} |",
                " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()]),
            )
            # Check for improvement
            validation_loss = loss.item()
            if validation_loss < min_validation_loss - min_delta:
                min_validation_loss = validation_loss
                counter = 0
            else:
                counter += 1

            # Early stopping
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
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


#####################################################################################
############## New GVAE model #######################################################
#####################################################################################

EPS = 1e-15
MAX_LOGSTD = 10
from torch_geometric.nn.inits import reset
from torch import Tensor
from torch_geometric.utils import negative_sampling
from typing import Optional, Tuple
import torch

from torch.nn import Module


class EncoderGAE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(EncoderGAE, self).__init__()
        self.gc1 = GCNConv(input_dim, hidden_dim)
        self.gc2_mu = GCNConv(hidden_dim, latent_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr):
        hidden = F.relu(self.gc1(x, edge_index, edge_attr))
        hidden = self.dropout(hidden)
        mu = self.gc2_mu(hidden, edge_index, edge_attr)

        return mu


class InnerProductDecoder(torch.nn.Module):
    def forward(
        self,
        z: Tensor,
        edge_index: Tensor,
        sigmoid: bool = True,
    ) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            edge_index (torch.Tensor): The edge indices.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.

    Args:
        encoder (torch.nn.Module): The encoder module.
        decoder (torch.nn.Module, optional): The decoder module. If set to
            :obj:`None`, will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """

    def __init__(
        self, input_dim, hidden_dim, latent_dim, decoder: Optional[Module] = None
    ):
        super().__init__()
        torch.manual_seed(42)
        self.encoder = EncoderGAE(input_dim, hidden_dim, latent_dim)
        self.bn_encoder = torch.nn.BatchNorm1d(latent_dim)

        self.decoder = InnerProductDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)
        self.kwargs = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
        }

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        reset(self.encoder)
        reset(self.decoder)

    def forward(self, *args, **kwargs) -> Tensor:  # pragma: no cover
        r"""Alias for :meth:`encode`."""
        return self.encoder(*args, **kwargs)

    def encode(self, *args, **kwargs) -> Tensor:
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs) -> Tensor:
        r"""Runs the decoder and computes edge probabilities."""
        return self.decoder(*args, **kwargs)

    def recon_loss(
        self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Optional[Tensor] = None
    ) -> Tensor:
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to train against.
            neg_edge_index (torch.Tensor, optional): The negative edges to
                train against. If not given, uses negative sampling to
                calculate negative edges. (default: :obj:`None`)
        """
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + EPS
        ).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(
            1 - self.decoder(z, neg_edge_index, sigmoid=True) + EPS
        ).mean()

        return pos_loss + neg_loss

    def test(
        self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor
    ) -> Tuple[Tensor, Tensor]:
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (torch.Tensor): The positive edges to evaluate
                against.
            neg_edge_index (torch.Tensor): The negative edges to evaluate
                against.
        """
        from sklearn.metrics import average_precision_score, roc_auc_score

        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

    def get_embeddings(self, x, edge_index, edge_attr):
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            return self.encoder(x, edge_index, edge_attr)
