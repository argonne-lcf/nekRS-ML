import math
from typing import Optional, Union, Callable, List, Dict, Any
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
import torch.distributed as dist
import torch.distributed.nn as distnn

class DistributedDGN(torch.nn.Module):
    r"""Distributed Diffusion Graph Neural Network (DGN)
    Args:
        arch (Dict[str, Any]): Architecture configuration
    """
    def __init__(self, arch: Dict[str, Any]):
        super().__init__()

        self.parse_arch(arch)

        # ~~~~ Diffusion-step embedding
        self.diffusion_step_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(self.mlp_hidden_channels),
            nn.Linear(self.mlp_hidden_channels, self.emb_width),
            nn.SELU(),
        )

        # ~~~~ Diffusion-step encoder
        self.diffusion_step_encoder = nn.ModuleList([
            nn.Linear(self.emb_width, self.mlp_hidden_channels),
            nn.SELU(), 
            nn.Linear(self.mlp_hidden_channels * 2, self.mlp_hidden_channels),
        ])

        # ~~~~ node encoder MLP
        self.node_encoder = MLP(
                input_features = self.input_node_features + self.cond_node_features,
                hidden_channels = [self.mlp_hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.mlp_hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.mlp_hidden_channels) if self.layer_norm else None,
                dropout_rate = self.dropout_rate
        )

        # ~~~~ edge encoder MLP
        self.edge_encoder = MLP(
                input_features = self.input_edge_features,
                hidden_channels = [self.mlp_hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.mlp_hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.mlp_hidden_channels) if self.layer_norm else None,
                dropout_rate = self.dropout_rate
        )

        # ~~~~ node decoder MLP
        self.node_decoder = MLP(
                input_features = self.mlp_hidden_channels,
                hidden_channels = [self.mlp_hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_features,
                activation_layer = torch.nn.ELU(),
                norm_layer = None,
                dropout_rate = self.dropout_rate
        )

        # ~~~~ Processor
        self.processor = torch.nn.ModuleList()
        for _ in range(self.n_messagePassing_layers):
            self.processor.append(
                        DistributedMessagePassingLayer(
                                     channels = self.mlp_hidden_channels,
                                     emb_features = self.emb_width,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers,
                                     halo_swap_mode = self.halo_swap_mode, 
                                     layer_norm = self.layer_norm,
                                     dropout_rate = self.dropout_rate
                        )
            )

        self.reset_parameters()

    def parse_arch(self, arch: Dict[str, Any]):
        self.arch = arch
        self.input_node_features = arch['input_node_features']
        self.cond_node_features = arch['cond_node_features']
        self.input_edge_features = arch['input_edge_features']
        self.mlp_hidden_channels = arch['mlp_hidden_channels']
        self.n_mlp_hidden_layers = arch['n_mlp_hidden_layers']
        self.n_messagePassing_layers = arch['n_messagePassing_layers']
        self.halo_swap_mode = arch['halo_swap_mode']
        self.layer_norm = arch['layer_norm']
        self.dropout_rate = arch['dropout_rate']
        self.emb_width = arch.get('emb_width', self.mlp_hidden_channels * 4)
        self.learnable_variance = arch.get('learnable_variance', False)
        self.output_node_features = self.input_node_features * 2 if self.learnable_variance else self.input_node_features
        self.name = arch['name']

    def forward(
            self,
            field_r: Tensor,
            r: Tensor,
            edge_index: torch.LongTensor,
            edge_attr: Tensor,
            edge_weight: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: List[Tensor],
            buffer_recv: List[Tensor],
            neighboring_procs: Tensor, 
            SIZE: Tensor,
            cond_node_features: Optional[Tensor] = None,
            batch: Optional[torch.LongTensor] = None
    ) -> Tensor:

        if batch is None:
            batch = torch.zeros(field_r.size(0), device=field_r.device, dtype=torch.long) # Shape (num_nodes,)

        # ~~~~ Embed the diffusion step
        emb = self.diffusion_step_embedding(r) # Shape (batch_size, emb_width)

        # ~~~~ Node encoder
        x = torch.cat([field_r, cond_node_features], dim=1) if cond_node_features is not None else field_r
        x = self.node_encoder(x) # Shape (batch_size, num_nodes, mlp_hidden_channels)

        # ~~~~ Encode the diffusion step embedding into the node features
        emb_proj = self.diffusion_step_encoder[0](emb) # Shape (batch_size, mlp_hidden_channels)
        x = torch.cat([x, emb_proj[batch]], dim=1) # Shape (batch_size, num_nodes, 2*mlp_hidden_channels)
        for layer in self.diffusion_step_encoder[1:]:
            x = layer(x)

        # ~~~~ Edge encoder
        e = self.edge_encoder(edge_attr) # Shape (num_edges, mlp_hidden_channels)

        # ~~~~ Processor
        for i in range(self.n_messagePassing_layers):
            x , _ = self.processor[i](x,
                                      e,
                                      emb, # TODO: need to figure out what to do with this
                                      edge_index,
                                      edge_weight,
                                      halo_info,
                                      mask_send,
                                      mask_recv,
                                      buffer_send,
                                      buffer_recv,
                                      neighboring_procs,
                                      SIZE,
                                      batch
            )

        # ~~~~ Node decoder
        x = self.node_decoder(x)

        # Return the output
        if self.learnable_variance:
            return torch.chunk(x, 2, dim=1)
        else:
            return x, torch.zeros_like(x)


    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.processor:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        return self.arch

    def get_save_header(self) -> str:
        header = self.name
        for key in self.arch.keys():
            if key != 'name':
                header += '_' + str(self.arch[key])
        return header


class MLP(torch.nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_channels: List[int],
                 output_channels: Optional[int] = None,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
                 dropout_rate: Optional[float] = 0.0,
                 bias: bool = True):
        super().__init__()

        self.input_features = input_features
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels if output_channels is not None else hidden_channels[-1]
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.dropout_rate = dropout_rate
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        
        self.ic = [input_features] + hidden_channels # input channel dimensions for each layer
        self.oc = hidden_channels + [output_channels] # output channel dimensions for each layer 

        self.mlp = torch.nn.ModuleList()
        for i in range(len(self.ic)):
            self.mlp.append( torch.nn.Linear(self.ic[i], self.oc[i], bias=bias) )

        self.reset_parameters()

        return

    def forward(self, x: Tensor) -> Tensor:
        for i in range(len(self.ic)):
            x = self.mlp[i](x)
            if i < (len(self.ic) - 1):
                x = self.activation_layer(x)
                x = self.norm_layer(x) if self.norm_layer else x
                x = self.dropout_layer(x)
        return x

    def reset_parameters(self):
        for module in self.mlp:
            module.reset_parameters()
        if self.norm_layer:
            self.norm_layer.reset_parameters()
        return

class DistributedMessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 emb_features: int,
                 n_mlp_hidden_layers: int,
                 halo_swap_mode: str,
                 layer_norm: Optional[bool] = False,
                 dropout_rate: Optional[float] = 0.0
    ) -> None:
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 
        self.halo_swap_mode = halo_swap_mode
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate

        # Projection of the diffusion-step embedding
        self.emb_features = emb_features
        if self.emb_features > 0: 
            self.node_emb_linear = nn.Linear(emb_features, channels)

        # Edge update MLP 
        self.edge_updater = MLP(
                input_features = self.channels*3,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.channels) if self.layer_norm else None,
                dropout_rate = self.dropout_rate
        )

        # Node update MLP
        self.node_updater = MLP(
                input_features = self.channels*2,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.channels) if self.layer_norm else None,
                dropout_rate = self.dropout_rate
        )

        self.reset_parameters()

        return 

    def forward(self,
            x: Tensor,
            e: Tensor,
            emb: Tensor,
            edge_index: torch.LongTensor,
            edge_weight: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: list,
            buffer_recv: list,
            neighboring_procs: Tensor,
            SIZE: Tensor,
            batch: Optional[torch.LongTensor] = None) -> Tensor:

        if batch is None:
            batch = torch.zeros(x.size(0), device=x.device, dtype=torch.long) # Shape (num_nodes,)

        # ~~~~ Project the diffusion-step embedding to the node embedding space
        if self.emb_features > 0:
            x += self.node_emb_linear(emb)[batch] # Shape (num_nodes, in_node_features)
        
        # ~~~~ Edge update 
        x_send = x[edge_index[0,:],:]
        x_recv = x[edge_index[1,:],:]
        e += self.edge_updater(
                torch.cat((x_send, x_recv, e), dim=1)
                )
        
        # ~~~~ Edge aggregation
        edge_weight = edge_weight.unsqueeze(1)
        e = e * edge_weight
        edge_agg = self.edge_aggregator(x, edge_index, e)

        if SIZE > 1 and self.halo_swap_mode != 'none':
            # ~~~~ Halo exchange: swap the edge aggregates. This populates the halo nodes  
            edge_agg = self.halo_swap(edge_agg, 
                                      mask_send,
                                      mask_recv,
                                      buffer_send, 
                                      buffer_recv, 
                                      neighboring_procs, 
                                      SIZE)

            # ~~~~ Local scatter using halo nodes (use halo_info) 
            idx_recv = halo_info[:,0]
            idx_send = halo_info[:,1]
            edge_agg.index_add_(0, idx_recv, edge_agg.index_select(0, idx_send))

        # ~~~~ Node update 
        x += self.node_updater(
                torch.cat((x, edge_agg), dim=1)
                )

        return x,e  

    def halo_swap(self,
                  input_tensor,
                  mask_send,
                  mask_recv,
                  buff_send,
                  buff_recv,
                  neighboring_procs,
                  SIZE):
        """
        Performs halo swap using send/receive buffers
        """
        if SIZE > 1:
            if self.halo_swap_mode == 'all_to_all' \
               or self.halo_swap_mode == 'all_to_all_opt' \
               or self.halo_swap_mode == 'all_to_all_opt_intel':
                # Fill send buffer
                for i in neighboring_procs:
                    n_send = len(mask_send[i])
                    buff_send[i][:n_send,:] = input_tensor[mask_send[i]]

                # # Perform all_to_all
                distnn.all_to_all(buff_recv, buff_send)

                # Fill halo nodes
                for i in neighboring_procs:
                    n_recv = len(mask_recv[i])
                    input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]

            elif self.halo_swap_mode == 'send_recv':
                # Fill send buffer
                for i in neighboring_procs:
                    n_send = len(mask_send[i])
                    buff_send[i][:n_send,:] = input_tensor[mask_send[i]] 

                # Perform sendrecv 
                distnn.send_recv(buff_recv, buff_send, neighboring_procs)

                # send_req = []
                # for dst in neighboring_procs:
                #     tmp = dist.isend(buff_send[dst], dst)
                #     send_req.append(tmp)
                # recv_req = []
                # for src in neighboring_procs:
                #     tmp = dist.irecv(buff_recv[src], src)
                #     recv_req.append(tmp)

                # for req in send_req:
                #     req.wait()
                # for req in recv_req:
                #     req.wait()
                # dist.barrier()

                # Fill halo nodes
                for i in neighboring_procs:
                    n_recv = len(mask_recv[i])
                    input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]


            elif self.halo_swap_mode == 'none':
                pass
            else:
                raise ValueError("halo_swap_mode %s not valid. Valid options: all_to_all, all_to_all_opt, all_to_all_opt_intel, send_recv, none" %(self.halo_swap_mode))
        return input_tensor


    def halo_swap_alloc(self,
                  input_tensor,
                  mask_send,
                  mask_recv,
                  buff_send,
                  buff_recv,
                  neighboring_procs,
                  SIZE):
        """
        Performs halo swap using send/receive buffers
        uses all_to_all implementation
        """
        if SIZE > 1:
            if self.halo_swap_mode == 'all_to_all':

                # Re-alloc send buffer 
                for i in range(SIZE):
                    #buff_send[i] = torch.empty([n_buffer_rows, n_features], dtype=input_tensor.dtype, device=input_tensor.device)
                    buff_send[i] = torch.empty_like(buff_send[i])

                # Fill send buffer
                for i in neighboring_procs:
                    n_send = len(mask_send[i])
                    buff_send[i][:n_send,:] = input_tensor[mask_send[i]]

                # # Perform all_to_all
                distnn.all_to_all(buff_recv, buff_send)

                # Fill halo nodes
                for i in neighboring_procs:
                    n_recv = len(mask_recv[i])
                    input_tensor[mask_recv[i]] = buff_recv[i][:n_recv,:]

            elif self.halo_swap_mode == 'none':
                pass
            else:
                raise ValueError("halo_swap_mode %s not valid. Valid options: all_to_all, sendrecv" %(self.halo_swap_mode))
        return input_tensor


    def reset_parameters(self):
        self.edge_updater.reset_parameters()
        self.node_updater.reset_parameters()
        return


class EdgeAggregation(MessagePassing):
    r"""This is a custom class that returns node quantities that represent the neighborhood-averaged edge features.
    Args:
        edge_dim (int, optional): Edge feature dimensionality. If set to
            :obj:`None`, node and edge feature dimensionality is expected to
            match. Other-wise, edge features are linearly transformed to match
            node feature dimensionality. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V_s}|, F_{s}), (|\mathcal{V_t}|, F_{t}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """

    propagate_type = {'x': Tensor, 'edge_attr': Tensor}

    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        return out

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        x_j = edge_attr
        return x_j

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}'


class SinusoidalPositionEmbedding(nn.Module):
    r"""Defines a sinusoidal embedding like in the paper "Attention is All You Need" (https://arxiv.org/abs/1706.03762).

    Args:
        dim (int): The dimension of the embedding.
        theta (float, optional): The theta parameter of the sinusoidal embedding. Defaults to 10000.
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.,
        ) -> None:
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even."
        self.dim = dim
        self.theta = theta

    def forward(
        self,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the embedding of position `r`."""    
        device = r.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb) # Dimensions: [dim/2]
        emb = r.unsqueeze(-1) * emb.unsqueeze(0) # Dimensions: [batch_size, dim/2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # Dimensions: [batch_size, dim]
        return emb
