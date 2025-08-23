

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
import torch.nn.functional as F
from typing import Optional, Literal
from gfn.utils.modules_dropout import MLP  


class MLP_Dr(MLP):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        K: int = 6,
        trunk: Optional[nn.Module] = None,
        n_hidden_layers: Optional[int] = 2,
        dropout_rate: float = 0.3,
        activation_fn: Optional[Literal["relu", "tanh", "elu"]] = "relu",
        add_layer_norm: bool = False,
        #dropout_prob: float = 0.1,  # Not used yet, but can be added in trunk if needed
    ):
        # Call MLP's constructor without K
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            activation_fn=activation_fn,
            dropout_rate= dropout_rate,
            add_layer_norm=add_layer_norm,
        )

        self.K = K
        # Replace single last layer with K separate output heads
        self.heads = nn.ModuleList([
            nn.Linear(self._hidden_dim, output_dim) for _ in range(K)
        ])

        # Initialize a random head index
        self.current_head_index = torch.randint(0, self.K, (1,)).item()

        @property
        def input_dim(self):
            return self._input_dim

        @property
        def output_dim(self):
            return self._output_dim

        @input_dim.setter
        def input_dim(self, value: int):
            self._input_dim = value

        @output_dim.setter
        def output_dim(self, value: int):
            self._output_dim = value

    def reset_policy_head(self):
        """Randomly select a new policy head for the current batch."""
        self.current_head_index = torch.randint(0, self.K, (1,)).item()

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward pass using the currently selected policy head."""
        if preprocessed_states.dtype != torch.float:
            preprocessed_states = preprocessed_states.float()

        x = self.trunk(preprocessed_states)  # Shared hidden layers
        out = self.heads[self.current_head_index](x)  # Selected head
        return out



      
