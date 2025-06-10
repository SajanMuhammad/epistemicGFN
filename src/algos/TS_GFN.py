"""This file contains some examples of modules that can be used with GFN."""

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
from gfn.utils.modules_dropout import MLP  # Assuming MLP is in a separate file


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


class Tabular(nn.Module):
    """Implements a tabular policy.

    This class is only compatible with the EnumPreprocessor.

    Attributes:
        table: a tensor with dimensions [n_states, output_dim].
        device: the device that holds this policy.
    """

    def __init__(self, n_states: int, output_dim: int) -> None:
        """
        Args:
            n_states (int): Number of states in the environment.
            output_dim (int): Output dimension.
        """

        super().__init__()

        self.table = torch.zeros(
            (n_states, output_dim),
            dtype=torch.float,
        )

        self.table = nn.parameter.Parameter(self.table)
        self.device = None

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the tabular policy.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the tabular policy. The shape of the tensor should be (*batch_shape, 1).
        Returns: a tensor of shape (*batch_shape, output_dim).
        """
        if self.device is None:
            self.device = preprocessed_states.device
            self.table = self.table.to(self.device)
        assert preprocessed_states.dtype == torch.long
        outputs = self.table[preprocessed_states.squeeze(-1)]
        return outputs


class DiscreteUniform(nn.Module):
    """Implements a uniform distribution over discrete actions.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(self, output_dim: int) -> None:
        """Initializes the uniform function approximiator.

        Args:
            output_dim (int): Output dimension. This is typically n_actions if it
                implements a Uniform PF, or n_actions-1 if it implements a Uniform PB.
        """
        super().__init__()
        self.output_dim = output_dim

    def forward(self, preprocessed_states: torch.Tensor) -> torch.Tensor:
        """Forward method for the uniform distribution.

        Args:
            preprocessed_states: a batch of states appropriately preprocessed for
                ingestion by the uniform distribution. The shape of the tensor should be (*batch_shape, input_dim).

        Returns: a tensor of shape (*batch_shape, output_dim).
        """
        out = torch.zeros(*preprocessed_states.shape[:-1], self.output_dim).to(
            preprocessed_states.device
        )
        return out


class LinearTransformer(nn.Module):
    """The Linear Transformer module.

    Implements Transformers are RNNs: Fast Autoregressive Transformers with Linear
        Attention. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François
        Fleuret, ICML 2020.

    Expresses self-attention as a linear dot-product of kernel feature maps and makes
    use the associativity property of matrix products to reduce the complexity of the
    attention computation from O(n^2) to O(n).

    Implementation from https://github.com/lucidrains/linear-attention-transformer.

    Args:
        dim: The dimension of the input.
        depth: The depth of the transformer.
        max_seq_len: The maximum sequence length.
        n_heads: The number of attention heads.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        max_seq_len: int,
        n_heads: int = 8,
        causal: bool = False,
    ):
        super().__init__()
        assert isinstance(dim, int) and dim > 0, "dim must be a positive integer"
        assert isinstance(depth, int) and depth > 0, "depth must be a positive integer"
        assert (
            isinstance(max_seq_len, int) and max_seq_len > 0
        ), "max_seq_len must be a positive integer"

        self.module = LinearAttentionTransformer(
            dim,
            depth,
            max_seq_len,
            heads=n_heads,
            causal=causal,
            dim_head=None,
            bucket_size=64,
            ff_chunks=1,
            ff_glu=False,
            ff_dropout=0.0,
            attn_layer_dropout=0.0,
            attn_dropout=0.0,
            reversible=False,
            blindspot_size=1,
            n_local_attn_heads=0,
            local_attn_window_size=128,
            receives_context=False,
            attend_axially=False,
            pkm_layers=tuple(),
            pkm_num_keys=128,
            linformer_settings=None,
            context_linformer_settings=None,
            shift_tokens=False,
        )
        # TODO: Should we have a final linear layer as part of this module?
        # The output dimension is the same as the embedding dimension.
        self.output_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class GraphEdgeActionGNN(nn.Module):
    """Implements a GNN for graph edge action prediction."""

    def __init__(
        self,
        n_nodes: int,
        directed: bool,
        num_edge_classes: int,
        num_conv_layers: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ) -> None:
        super().__init__()

        assert n_nodes > 0, "n_nodes must be greater than 0"
        assert embedding_dim > 0, "embedding_dim must be greater than 0"
        assert num_conv_layers > 0, "num_conv_layers must be greater than 0"
        assert isinstance(n_nodes, int), "n_nodes must be an integer"
        assert isinstance(embedding_dim, int), "embedding_dim must be an integer"
        assert isinstance(num_conv_layers, int), "num_conv_layers must be an integer"
        assert isinstance(directed, bool), "directed must be a boolean"
        assert isinstance(is_backward, bool), "is_backward must be a boolean"
        self._input_dim = 1  # Each node input is a single integer before embedding.
        self._n_nodes = n_nodes
        self.hidden_dim = self.embedding_dim = embedding_dim
        self.is_backward = is_backward
        self.is_directed = directed
        self.num_conv_layers = num_conv_layers
        self.num_edge_classes = num_edge_classes

        # Output dimension.
        edges_dim = self.n_nodes**2 - self.n_nodes
        if not self.is_directed:
            edges_dim = edges_dim // 2  # No double-counting.

        if not self.is_backward:
            out_dim = edges_dim + 1  # +1 for exit action.
        else:
            out_dim = edges_dim

        self._output_dim = int(out_dim)
        self._edges_dim = int(edges_dim)

        # Node embedding layer.
        self.embedding = nn.Embedding(n_nodes, self.embedding_dim)
        self.conv_blks = nn.ModuleList()
        self.exit_mlp = MLP(
            input_dim=self.hidden_dim,
            output_dim=1,
            hidden_dim=self.hidden_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

        if directed:
            for i in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        DirGNNConv(
                            GCNConv(
                                self.embedding_dim if i == 0 else self.hidden_dim,
                                self.hidden_dim,
                            ),
                            alpha=0.5,
                            root_weight=True,
                        ),
                        # Process in/out components separately
                        nn.ModuleList(
                            [
                                nn.Sequential(
                                    nn.Linear(
                                        self.hidden_dim // 2, self.hidden_dim // 2
                                    ),
                                    nn.ReLU(),
                                    nn.Linear(
                                        self.hidden_dim // 2, self.hidden_dim // 2
                                    ),
                                )
                                for _ in range(2)  # 1 for in & 1 for out-features.
                            ]
                        ),
                    ]
                )
        else:  # Undirected case.
            for i in range(num_conv_layers):
                self.conv_blks.extend(
                    [
                        GINConv(
                            MLP(
                                input_dim=(
                                    self.embedding_dim if i == 0 else self.hidden_dim
                                ),
                                output_dim=self.hidden_dim,
                                hidden_dim=self.hidden_dim,
                                n_hidden_layers=1,
                                add_layer_norm=True,
                            ),
                        ),
                        nn.Sequential(
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                            nn.ReLU(),
                            nn.Linear(self.hidden_dim, self.hidden_dim),
                        ),
                    ]
                )

        self.norm = nn.LayerNorm(self.hidden_dim)

    @property
    def input_dim(self):
        return self._input_dim

    @property
    def n_nodes(self):
        return self._n_nodes

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def edges_dim(self) -> int:
        return self._edges_dim

    @staticmethod
    def _group_mean(tensor: torch.Tensor, batch_ptr: torch.Tensor) -> torch.Tensor:
        cumsum = torch.zeros(
            (len(tensor) + 1, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        cumsum[1:] = torch.cumsum(tensor, dim=0)

        # Subtract the end val from each batch idx from the start val of each batch idx.
        size = batch_ptr[1:] - batch_ptr[:-1]
        return (cumsum[batch_ptr[1:]] - cumsum[batch_ptr[:-1]]) / size[:, None]

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        node_features, batch_ptr = (states_tensor.x, states_tensor.ptr)
        batch_size = int(math.prod(states_tensor.batch_shape))

        # Multiple action type convolutions with residual connections.
        x = self.embedding(node_features.squeeze().int())
        for i in range(0, len(self.conv_blks), 2):
            x_new = self.conv_blks[i](x, states_tensor.edge_index)  # GIN/GCN conv.
            if self.is_directed:
                assert isinstance(self.conv_blks[i + 1], nn.ModuleList)
                x_in, x_out = torch.chunk(x_new, 2, dim=-1)

                # Process each component separately through its own MLP.
                mlp_in, mlp_out = self.conv_blks[i + 1]
                x_in = mlp_in(x_in)
                x_out = mlp_out(x_out)
                x_new = torch.cat([x_in, x_out], dim=-1)
            else:
                x_new = self.conv_blks[i + 1](x_new)  # Linear -> ReLU -> Linear.

            x = x_new + x if i > 0 else x_new  # Residual connection.
            x = self.norm(x)  # Layernorm.

        node_feature_means = self._group_mean(x, batch_ptr)
        if not self.is_backward:
            exit_action = self.exit_mlp(node_feature_means).squeeze(-1)

        x = x.reshape(*states_tensor.batch_shape, self.n_nodes, self.hidden_dim)

        # Undirected.
        if self.is_directed:
            feature_dim = self.hidden_dim // 2
            source_features = x[..., :feature_dim]
            target_features = x[..., feature_dim:]

            # Dot product between source and target features (asymmetric).
            edgewise_dot_prod = torch.einsum(
                "bnf,bmf->bnm", source_features, target_features
            )
            edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(torch.tensor(feature_dim))

            i_up, j_up = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)
            i_lo, j_lo = torch.tril_indices(self.n_nodes, self.n_nodes, offset=-1)

            # Combine them.
            i0 = torch.cat([i_up, i_lo])
            i1 = torch.cat([j_up, j_lo])

        else:
            # Dot product between all node features (symmetric).
            edgewise_dot_prod = torch.einsum("bnf,bmf->bnm", x, x)
            edgewise_dot_prod = edgewise_dot_prod / torch.sqrt(
                torch.tensor(self.hidden_dim)
            )
            i0, i1 = torch.triu_indices(self.n_nodes, self.n_nodes, offset=1)

        # Grab the needed elements from the adjacency matrix and reshape.
        edge_actions = edgewise_dot_prod[torch.arange(batch_size)[:, None, None], i0, i1]
        edge_actions = edge_actions.reshape(
            *states_tensor["batch_shape"],
            self.edges_dim,
        )

        action_type = torch.zeros(*states_tensor["batch_shape"], 3, device=x.device)
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 1
        else:
            action_type[..., GraphActionType.ADD_EDGE] = 1 - exit_action
            action_type[..., GraphActionType.EXIT] = exit_action

        return TensorDict(
            {
                "action_type": action_type,
                "edge_class": torch.zeros(
                    *states_tensor["batch_shape"], self.num_edge_classes, device=x.device
                ),  # TODO: make it learnable.
                "node_class": torch.zeros(
                    *states_tensor["batch_shape"], 1, device=x.device
                ),
                "edge_index": edge_actions,
            },
            batch_size=states_tensor["batch_shape"],
        )


class GraphEdgeActionMLP(nn.Module):
    """Network that processes flattened adjacency matrices to predict graph actions.

    Unlike the GNN-based GraphEdgeActionGNN, this module uses standard MLPs to process
    the entire adjacency matrix as a flattened vector. This approach:

    1. Can directly process global graph structure without message passing.
    2. May be more effective for small graphs where global patterns are important.
    3. Does not require complex graph neural network operations.

    The module architecture consists of:
    - An MLP to process the flattened adjacency matrix into an embedding.
    - An edge MLP that predicts logits for each possible edge action.
    - An exit MLP that predicts a logit for the exit action.

    Args:
        n_nodes: Number of nodes in the graph.
        directed: Whether the graph is directed or undirected.
        n_hidden_layers: Number of hidden layers in the MLP for the edge actions.
        n_hidden_layers_exit: Number of hidden layers in the MLP for the exit action.
        embedding_dim: Dimension of internal embeddings.
        is_backward: Whether this is a backward policy.
    """

    def __init__(
        self,
        n_nodes: int,
        directed: bool,
        num_edge_classes: int,
        n_hidden_layers: int = 2,
        n_hidden_layers_exit: int = 1,
        embedding_dim: int = 128,
        is_backward: bool = False,
    ):
        super().__init__()
        assert n_nodes > 0, "n_nodes must be greater than 0"
        assert embedding_dim > 0, "embedding_dim must be greater than 0"
        assert n_hidden_layers > 0, "n_hidden_layers must be greater than 0"
        assert n_hidden_layers_exit > 0, "n_hidden_layers_exit must be greater than 0"
        assert isinstance(n_nodes, int), "n_nodes must be an integer"
        assert isinstance(embedding_dim, int), "embedding_dim must be an integer"
        assert isinstance(n_hidden_layers, int), "n_hidden_layers must be an integer"
        assert isinstance(
            n_hidden_layers_exit, int
        ), "n_hidden_layers_exit must be an integer"
        assert isinstance(directed, bool), "directed must be a boolean"
        assert isinstance(is_backward, bool), "is_backward must be a boolean"
        self._input_dim = n_nodes**2
        self.n_nodes = n_nodes
        self.is_directed = directed
        self.is_backward = is_backward
        self.hidden_dim = embedding_dim
        self.num_edge_classes = num_edge_classes

        # MLP for processing the flattened adjacency matrix
        self.mlp = MLP(
            input_dim=n_nodes**2,  # Flattened adjacency matrix
            output_dim=embedding_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=n_hidden_layers,
            add_layer_norm=True,
        )

        # Exit action MLP
        self.exit_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=1,
            hidden_dim=embedding_dim,
            n_hidden_layers=n_hidden_layers_exit,
            add_layer_norm=True,
        )

        # Edge prediction MLP
        # Output dimension.
        edges_dim = self.n_nodes**2 - self.n_nodes
        if not self.is_directed:
            edges_dim = edges_dim // 2  # No double-counting.

        if not self.is_backward:
            out_dim = edges_dim + 1  # +1 for exit action.
        else:
            out_dim = edges_dim

        self._output_dim = int(out_dim)
        self._edges_dim = int(edges_dim)

        self.edge_mlp = MLP(
            input_dim=embedding_dim,
            output_dim=self.edges_dim,
            hidden_dim=embedding_dim,
            n_hidden_layers=1,
            add_layer_norm=True,
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def edges_dim(self) -> int:
        return self._edges_dim

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        """Forward pass to compute action logits from graph states.

        Process:
        1. Convert the graph representation to adjacency matrices
        2. Process the flattened adjacency matrices through the main MLP
        3. Predict logits for edge actions and exit action

        Args:
            states_tensor: A GeometricBatch containing graph state information

        Returns:
            A tensor of logits for all possible actions
        """
        device = states_tensor.x.device
        # Convert the graph to adjacency matrix.
        batch_size = int(states_tensor.batch_size)
        adj_matrices = torch.zeros(
            (batch_size, self.n_nodes, self.n_nodes),
            device=device,
        )

        # Fill the adjacency matrices from edge indices
        if states_tensor.edge_index.numel() > 0:
            for i in range(batch_size):
                eis = states_tensor[i].edge_index
                adj_matrices[i, eis[0], eis[1]] = 1

        # Flatten the adjacency matrices for the MLP
        adj_matrices_flat = adj_matrices.view(batch_size, -1)

        # Process with MLP
        embedding = self.mlp(adj_matrices_flat)

        # Generate edge and exit actions
        edge_actions = self.edge_mlp(embedding)
        exit_action = self.exit_mlp(embedding).squeeze(-1)

        action_type = torch.zeros(*states_tensor["batch_shape"], 3, device=device)
        if self.is_backward:
            action_type[..., GraphActionType.ADD_EDGE] = 1
        else:
            action_type[..., GraphActionType.ADD_EDGE] = 1 - exit_action
            action_type[..., GraphActionType.EXIT] = exit_action

        return TensorDict(
            {
                "action_type": action_type,
                "edge_class": torch.zeros(
                    *states_tensor["batch_shape"], self.num_edge_classes, device=device
                ),  # TODO: make it learnable
                "node_class": torch.zeros(
                    *states_tensor["batch_shape"], 1, device=device
                ),
                "edge_index": edge_actions,
            },
            batch_size=states_tensor["batch_shape"],
        )


class GraphActionUniform(nn.Module):
    """Implements a uniform distribution over discrete actions given a graph state.

    It uses a zero function approximator (a function that always outputs 0) to be used as
    logits by a DiscretePBEstimator.

    Attributes:
        output_dim: The size of the output space.
    """

    def __init__(
        self,
        edges_dim: int,
        num_edge_classes: int,
        num_node_classes: int,
    ) -> None:
        """Initializes the uniform function approximiator.

        Args:
            edges_dim (int): The dimension of edge_index in GraphActions.
            num_edge_classes (int): Number of edge classes.
            num_node_classes (int): Number of node classes.
        """
        super().__init__()
        self.input_dim = 1  # has no effect
        self.edges_dim = edges_dim
        self.num_edge_classes = num_edge_classes
        self.num_node_classes = num_node_classes

    def forward(self, states_tensor: GeometricBatch) -> TensorDict:
        """Forward method for the uniform distribution.

        Args:
            states_tensor: a batch of states appropriately preprocessed for
                ingestion by the uniform distribution.

        Returns:
            A TensorDict containing logits for each action component, with all values set to 1 to represent a uniform distribution:
            - "action_type": Tensor of shape [*batch_shape, 3] for the 3 possible action types
            - "edge_class": Tensor of shape [*batch_shape, num_edge_classes] for edge class logits
            - "node_class": Tensor of shape [*batch_shape, num_node_classes] for node class logits
            - "edge_index": Tensor of shape [*batch_shape, edges_dim] for edge index logits
        """
        device = states_tensor.x.device
        return TensorDict(
            {
                "action_type": torch.ones(*states_tensor.batch_shape, 3, device=device),
                "edge_class": torch.ones(
                    *states_tensor.batch_shape, self.num_edge_classes, device=device
                ),
                "node_class": torch.ones(
                    *states_tensor.batch_shape, self.num_node_classes, device=device
                ),
                "edge_index": torch.ones(
                    *states_tensor.batch_shape, self.edges_dim, device=device
                ),
            },
            batch_size=states_tensor.batch_shape,
        )
