import torch
import torch.nn as nn
import einops

import torch.distributed.nn as distnn
from torch.nn.functional import scaled_dot_product_attention as sdpa
from torch_scatter import scatter_mean


# GeGLU activation
class GeGLU(nn.Module):
    """
    Gated GELU activation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.chunk(x, 2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SwiGLU(nn.Module):
    """
    SwiGLU activation.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = torch.chunk(x, 2, dim=-1)
        return x * torch.nn.functional.silu(gate)


def apply_rope(
    x: torch.Tensor,
    coords: torch.Tensor,
    max_wavelength: int = 100,
) -> torch.Tensor:
    n_dim = coords.shape[-1]
    feature_dim = x.shape[-1]

    # Each coordinate dimension gets an equal slice of the feature space.
    # Each of these slices must be divisible by 2 for the sin/cos rotation.
    # NOTE: we comment this out to allow for different feature dimensions
    # NOTE(CONTD.):but essentially, we're doing this in a loose sense just that we may
    # NOTE(CONTD.):have some feature dimensions that are not used for rotation
    # assert feature_dim % (2 * n_dim) == 0, (
    #     f"The feature dimension ({feature_dim}) must be divisible by "
    #     f"2 * n_dim ({2 * n_dim})."
    # )

    # Split the feature dimension into chunks, one for each coordinate dimension.
    per_dim_features = 2 * (feature_dim // (2 * n_dim))
    rotated_chunks = []
    for i in range(n_dim):
        # Select the feature chunk and coordinates for the current dimension.
        current_x_chunk = x[..., i * per_dim_features : (i + 1) * per_dim_features]
        current_coords = coords[..., i]

        # Calculate the frequencies (or inverse timescales).
        head_dim = per_dim_features
        half_head_dim = head_dim // 2
        fraction = 2 * torch.arange(half_head_dim, device=x.device) / head_dim
        timescale = max_wavelength**fraction

        # Calculate the angle `theta` for rotation.
        theta = current_coords.unsqueeze(-1) / timescale

        # Compute the sine and cosine of the angle.
        sin = torch.sin(theta)
        cos = torch.cos(theta)

        # Split the feature chunk into two halves for rotation.
        first_half, second_half = torch.chunk(current_x_chunk, 2, dim=-1)
        # Apply the 2D rotation matrix:
        sin = einops.repeat(sin, "b n c -> b h n c", h=first_half.shape[1])
        cos = einops.repeat(cos, "b n c -> b h n c", h=first_half.shape[1])

        rotated_first_half = first_half * cos - second_half * sin
        rotated_second_half = second_half * cos + first_half * sin

        # Concatenate the rotated halves back together.
        rotated_chunk = torch.cat([rotated_first_half, rotated_second_half], dim=-1)
        rotated_chunks.append(rotated_chunk)

    # Concatenate all the processed chunks back along the feature dimension.
    result = torch.cat(rotated_chunks, dim=-1)
    return result.to(x.dtype)


# GeGLU MLP without bias
class MlpBlock(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        use_bias: bool = False,
        activation: str = "swish",
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.activation = activation

        self.dense1 = nn.Linear(in_dim, hidden_dim * 2, bias=use_bias)
        if activation == "gelu":
            self.glu = GeGLU()
        elif activation == "swish":
            self.glu = SwiGLU()
        self.dense2 = nn.Linear(hidden_dim, out_dim, bias=use_bias)

        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        MLP Block that uses GeGLU.

        Args:
            inputs: Input tensor of shape (B, N, E)

        Returns:
            The output of the mlp
        """
        x = self.dense1(inputs)
        x = self.glu(x)
        x = self.dense2(x)
        return x


class ElementWiseAttention(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        num_elements: int,
        poly_order: int = 7,
        use_bias: bool = False,
        mlp_ratio: float = 1.0,
        activation: str = "swish",
        halo_swap_mode: str = "none",
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.num_elements = num_elements
        self.poly_order = poly_order
        self.use_bias = use_bias
        self.activation = activation
        self.mlp_ratio = mlp_ratio
        self.halo_swap_mode = halo_swap_mode

        # attention specific layers:
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.q_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)
        self.o_proj = nn.Linear(hidden_channels, hidden_channels, bias=use_bias)

        # mlp specific layers:
        self.norm2 = nn.LayerNorm(hidden_channels)
        self.ffn = MlpBlock(
            hidden_channels, int(hidden_channels * mlp_ratio), hidden_channels
        )

    def forward(
        self,
        x,
        pos,
        index,
        mask_send,
        mask_recv,
        buffer_send,
        buffer_recv,
        halo_info,
        idx_reduced2full,
        idx_full2reduced,
        neighboring_procs,
        SIZE,
    ):
        """
        Element-wise attention that performs attention only within nodes present within a single meshed element.
        This is specifically designed for spectral element methods, where attention is constrained to nodes
        within the same element to maintain local spatial relationships.

        Args:
            x: Input tensor of shape (B, E, N, C) where B is batch size, E is number of elements,
               N is nodes per element, and C is the feature dimension

        Returns:
            Tensor of the same shape as input with attention applied within each element
        """
        poly_order = self.poly_order
        nodes_per_element = (poly_order + 1) ** 3
        num_elements = idx_reduced2full.shape[0] // nodes_per_element

        res = x
        # our original data lives on a reduced graph
        # we'll expand it to the full graph
        x_full = x[idx_reduced2full]
        pos_full = pos[idx_reduced2full]

        # we'll reshape it so that we have (num_elements, nodes_per_element, feature_dim)
        # so we can now treat the number of elements as an effective batch size
        # and perform element-wise attention with the number of nodes per element as the sequence length
        x_full = x_full.reshape(num_elements, nodes_per_element, x.shape[-1])
        pos_full = pos_full.reshape(num_elements, nodes_per_element, pos.shape[-1])
        # NOTE: hardcoding the min and max positions for now
        min_pos = torch.tensor(
            [-10.0, -1.0, 0.0], device=pos_full.device, requires_grad=True
        )
        max_pos = torch.tensor(
            [25.0, 5.0, 2.0], device=pos_full.device, requires_grad=True
        )
        pos_full = (pos_full - min_pos) / (max_pos - min_pos)

        # we are performing a pre-norm transformer
        x_full = self.norm1(x_full)

        # Apply attention within each element (now with global context)
        q = self.q_proj(x_full)
        k = self.k_proj(x_full)
        v = self.v_proj(x_full)

        q = einops.rearrange(q, "ne np (h c) -> ne h np c", h=self.num_heads)
        k = einops.rearrange(k, "ne np (h c) -> ne h np c", h=self.num_heads)
        v = einops.rearrange(v, "ne np (h c) -> ne h np c", h=self.num_heads)

        # encoding position information through RoPE
        q = apply_rope(q, pos_full)
        k = apply_rope(k, pos_full)
        # using the scaled dot product attention from torch.nn.functional
        # this should be using flash attention if available
        attn_output = sdpa(q, k, v)
        attn_output = einops.rearrange(attn_output, "ne h np c -> ne np (h c)")
        attn_output = self.o_proj(attn_output)

        # note that the above will have different values across elements
        # even though they may correspond to the same physical location
        # we need to reconcile this by doing an average across points with shared IDs:
        ne, np, c = attn_output.shape
        # NOTE: we can possibly just compute this once and reuse it
        _, grouping_index = torch.unique(index[idx_reduced2full], return_inverse=True)
        attn_output = attn_output.reshape(ne * np, c)
        attn_output = scatter_mean(attn_output, grouping_index, dim=0)[grouping_index]
        # make attention output in reduced:
        attn_output = attn_output[idx_full2reduced]

        # add halo nodes
        num_halo_nodes = halo_info.shape[0]
        if SIZE > 1:
            # add attn_output to x except the last num_halo_nodes
            x[:-num_halo_nodes] = res[:-num_halo_nodes] + attn_output
            # this should have populated the halo nodes
            if self.halo_swap_mode == "all_to_all" or self.halo_swap_mode == "all_to_all_opt":
                x = self.halo_swap(
                    x,
                    mask_send,
                    mask_recv,
                    buffer_send,
                    buffer_recv,
                    neighboring_procs,
                    SIZE,
                )

            else:
                assert self.halo_swap_mode == "none", "Invalid halo swap mode"

            # now we'll aggregate the information:
            idx_recv = halo_info[:, 0]
            idx_send = halo_info[:, 1]
            x.index_add_(0, idx_recv, x.index_select(0, idx_send))
            # take the mean instead of the sum
            counts = torch.ones(x.size(0), device=x.device)  # self-contribution
            ones = torch.ones(idx_recv.size(0), device=x.device)
            counts.index_add_(0, idx_recv, ones)  # + incoming msgs
            # divide each node’s feature vector by its count
            x = x / counts.unsqueeze(-1)
        else:
            x = x + attn_output

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + y

        return x

    def halo_swap(
        self,
        input_tensor,
        mask_send,
        mask_recv,
        buff_send,
        buff_recv,
        neighboring_procs,
        SIZE,
    ):
        """
        Performs halo swap using send/receive buffers
        """
        if SIZE > 1:
            # Fill send buffer
            for i in neighboring_procs:
                n_send = len(mask_send[i])
                buff_send[i][:n_send, :] = input_tensor[mask_send[i]]

            # Perform all_to_all
            distnn.all_to_all(buff_recv, buff_send)

            # Fill halo nodes
            for i in neighboring_procs:
                n_recv = len(mask_recv[i])
                input_tensor[mask_recv[i]] = buff_recv[i][:n_recv, :]

        return input_tensor


class GraphTransformer(nn.Module):
    def __init__(
        self,
        input_node_channels,
        hidden_channels,
        output_node_channels,
        n_transformer_layers,
        num_heads,
        num_elements,
        halo_swap_mode,
        name,
    ):
        super().__init__()
        self.input_node_channels = input_node_channels
        self.hidden_channels = hidden_channels
        self.num_elements = num_elements
        self.output_node_channels = output_node_channels
        self.n_transformer_layers = n_transformer_layers
        self.num_heads = num_heads
        self.encoder = MlpBlock(input_node_channels, hidden_channels, hidden_channels)
        self.processor = nn.ModuleList()
        for i in range(n_transformer_layers):
            self.processor.append(
                ElementWiseAttention(
                    hidden_channels,
                    num_heads,
                    num_elements,
                    halo_swap_mode=halo_swap_mode,
                )
            )
        self.decoder = MlpBlock(hidden_channels, hidden_channels, output_node_channels)
        self.halo_swap_mode = halo_swap_mode
        self.name = name

    def forward(
        self,
        x,
        pos,
        index,
        mask_send,
        mask_recv,
        buffer_send,
        buffer_recv,
        halo_info,
        idx_reduced2full,
        idx_full2reduced,
        neighboring_procs,
        SIZE,
    ):
        # Encode nodes
        x = self.encoder(x)
        # Process nodes
        for i in range(self.n_transformer_layers):
            x = self.processor[i](
                x,
                pos,
                index,
                mask_send,
                mask_recv,
                buffer_send,
                buffer_recv,
                halo_info,
                idx_reduced2full,
                idx_full2reduced,
                neighboring_procs,
                SIZE,
            )

        # Decode nodes
        x = self.decoder(x)
        x = x.reshape(-1, x.shape[-1])
        return x

    def input_dict(self) -> dict:
        a = {
            "input_node_channels": self.input_node_channels,
            "hidden_channels": self.hidden_channels,
            "output_node_channels": self.output_node_channels,
            "n_transformer_layers": self.n_transformer_layers,
            "num_heads": self.num_heads,
            "name": self.name,
        }
        return a

    def get_save_header(self) -> str:
        a = self.input_dict()
        header = "graph_transformer"

        for key in a.keys():
            header += "_" + str(a[key])

        return header
