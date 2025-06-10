

import math
from typing import Literal, Optional

import torch
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer
from tensordict import TensorDict
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.nn import DirGNNConv, GCNConv, GINConv

from gfn.actions import GraphActionType

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Literal



class GaussianIndexer:

    def __init__(self, index_dim: int):
        self.index_dim = index_dim

    def __call__(self, seed: Optional[int] = None) -> torch.Tensor:
        if seed is not None:
            np.random.seed(seed)
        z = np.random.randn(self.index_dim)  # N(0, I)
        return torch.tensor(z, dtype=torch.float32)


# ===== MLP_ENN Model =====
'''class MLP_ENN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        index_dim: int = 7,
        hidden_dim: int = 256,
        epinet_hidden_dim: int = 128,
        epinet_layers: int = 2,
        trunk: Optional[nn.Module] = None,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh", "elu"]] = "relu",
        add_layer_norm: bool = False,
        prior_scale: float = 1.0,
        stop_gradient: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.index_dim = index_dim
        self.stop_gradient = stop_gradient
        self.prior_scale = prior_scale
        if trunk is not None:
            self.trunk = trunk
        else:
            base_layers = [nn.Linear(input_dim, hidden_dim)]
            ...
            self.trunk = nn.Sequential(*base_layers)

        # ===== Activation =====
        if activation_fn == "elu":
            activation = nn.ELU
        elif activation_fn == "relu":
            activation = nn.ReLU
        elif activation_fn == "tanh":
            activation = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        # ===== Base network (Trunk) =====
        base_layers = [nn.Linear(input_dim, hidden_dim)]
        if add_layer_norm:
            base_layers.append(nn.LayerNorm(hidden_dim))
        base_layers.append(activation())

        for _ in range(n_hidden_layers - 1):
            base_layers.append(nn.Linear(hidden_dim, hidden_dim))
            if add_layer_norm:
                base_layers.append(nn.LayerNorm(hidden_dim))
            base_layers.append(activation())

        self.trunk = nn.Sequential(*base_layers)
        self.base_head = nn.Linear(hidden_dim, output_dim)

        # ===== Epinet networks (train and frozen prior) =====
        def make_epinet():
            layers = [nn.Linear(hidden_dim, epinet_hidden_dim), activation()]
            for _ in range(epinet_layers - 1):
                layers.append(nn.Linear(epinet_hidden_dim, epinet_hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(epinet_hidden_dim, output_dim * index_dim))
            return nn.Sequential(*layers)

        self.epinet_train = make_epinet()
        self.epinet_prior = make_epinet()

        # Freeze prior network
        for param in self.epinet_prior.parameters():
            param.requires_grad = False

    def forward(self, preprocessed_states: torch.Tensor, z: torch.Tensor):
        """
        preprocessed_states: [B, input_dim] - input batch after preprocessing
        z: [index_dim] - epistemic index for stochasticity
        """
        if preprocessed_states.dtype != torch.float:
            preprocessed_states = preprocessed_states.float()  # TODO: handle precision.

        features = self.trunk(preprocessed_states)

        if self.stop_gradient:
            features = features.detach()

        base_out = self.base_head(features)

        def project(epinet_net, features: torch.Tensor, detach: bool = False):
            if detach:
                features = features.detach()
            epinet_out = epinet_net(features)  # [B, C * Z]
            epinet_out = epinet_out.view(-1, self.output_dim, self.index_dim)  # [B, C, Z]
            return torch.einsum('bcz,z->bc', epinet_out, z)  # [B, C]

        epi_train = project(self.epinet_train, features, detach=False)
        epi_prior = project(self.epinet_prior, features, detach=True)

        final_train = base_out + epi_train
        final_prior = self.prior_scale * epi_prior

        return final_train + final_prior'''

class MLP_ENN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        index_dim: int = 6,
        hidden_dim: int = 256,
        epinet_hidden_dim: int = 128,
        epinet_layers: int = 2,
        trunk: Optional[nn.Module] = None,
        n_hidden_layers: Optional[int] = 2,
        activation_fn: Optional[Literal["relu", "tanh", "elu"]] = "relu",
        add_layer_norm: bool = False,
        prior_scale: float = 1,
        stop_gradient: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim

        self.index_dim = index_dim
        self.stop_gradient = stop_gradient
        self.prior_scale = prior_scale

        #Activations
        if activation_fn == "elu":
            activation = nn.ELU
        elif activation_fn == "relu":
            activation = nn.ReLU
        elif activation_fn == "tanh":
            activation = nn.Tanh
        else:
            raise ValueError(f"Unsupported activation_fn: {activation_fn}")

        # Trunk
        if trunk is not None:
            self.trunk = trunk
        else:
            base_layers = [nn.Linear(input_dim, hidden_dim)]
            if add_layer_norm:
                base_layers.append(nn.LayerNorm(hidden_dim))
            base_layers.append(activation())

            for _ in range(n_hidden_layers - 1):
                base_layers.append(nn.Linear(hidden_dim, hidden_dim))
                if add_layer_norm:
                    base_layers.append(nn.LayerNorm(hidden_dim))
                base_layers.append(activation())

            self.trunk = nn.Sequential(*base_layers)

        self.base_head = nn.Linear(hidden_dim, output_dim)

        #  Epinet networks with learnable network and prior network
        def make_epinet():
            layers = [nn.Linear(hidden_dim, epinet_hidden_dim), activation()]
            for _ in range(epinet_layers - 1):
                layers.append(nn.Linear(epinet_hidden_dim, epinet_hidden_dim))
                layers.append(activation())
            layers.append(nn.Linear(epinet_hidden_dim, output_dim * index_dim))
            return nn.Sequential(*layers)

        self.epinet_train = make_epinet()
        self.epinet_prior = make_epinet()

        # Freeze prior network
        for param in self.epinet_prior.parameters():
            param.requires_grad = False

    def forward(self, preprocessed_states: torch.Tensor):

        if preprocessed_states.dtype != torch.float:
            preprocessed_states = preprocessed_states.float()

        # Sample z from standard Gaussian
        z = torch.randn(self.index_dim, dtype=torch.float32, device=preprocessed_states.device)

        features = self.trunk(preprocessed_states)
        if self.stop_gradient:
            features = features.detach()

        base_out = self.base_head(features)

        def project(epinet_net, features: torch.Tensor, detach: bool = False):
            if detach:
                features = features.detach()
            epinet_out = epinet_net(features)
            epinet_out = epinet_out.view(-1, self.output_dim, self.index_dim)
            return torch.einsum('bcz,z->bc', epinet_out, z)

        epi_train = project(self.epinet_train, features, detach=False)
        epi_prior = project(self.epinet_prior, features, detach=True)

        final_train = base_out + epi_train
        final_prior = self.prior_scale * epi_prior

        return final_train + final_prior


