from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x
import torch.distributed as dist
import torch.distributed.nn_mod as distnn

class DistributedGNN(torch.nn.Module):
    def __init__(self,
                 input_node_channels: int,
                 input_edge_channels: int,
                 hidden_channels: int,
                 output_node_channels: int,
                 n_mlp_hidden_layers: int,
                 n_messagePassing_layers: int,
                 halo_swap_mode: Optional[str] = 'all_to_all',
                 name: Optional[str] = 'gnn'):
        super().__init__()

        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.halo_swap_mode = halo_swap_mode
        self.name = name

        # ~~~~ node encoder MLP
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ edge encoder MLP
        self.edge_encoder = MLP(
                input_channels = self.input_edge_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ node decoder MLP
        self.node_decoder = MLP(
                input_channels = self.hidden_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.output_node_channels,
                activation_layer = torch.nn.ELU(),
                )

        # ~~~~ Processor
        self.processor = torch.nn.ModuleList()
        for i in range(self.n_messagePassing_layers):
            self.processor.append(
                          DistributedMessagePassingLayer(
                                     channels = self.hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers,
                                     halo_swap_mode = self.halo_swap_mode, 
                                     )
                                  )

        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            edge_weight: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: List[Tensor],
            buffer_recv: List[Tensor],
            neighboring_procs: Tensor, 
            SIZE: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))

        # ~~~~ Compute edge features
        x_send = x[edge_index[0,:],:]
        x_recv = x[edge_index[1,:],:]
        pos_send = pos[edge_index[0,:],:]
        pos_recv = pos[edge_index[1,:],:]
        e_1 = pos_send - pos_recv
        e_2 = torch.norm(e_1, dim=1, p=2, keepdim=True)
        e_3 = x_send - x_recv
        e = torch.cat((e_1, e_2, e_3), dim=1)

        # ~~~~ Node encoder
        x = self.node_encoder(x)

        # ~~~~ Edge encoder
        e = self.edge_encoder(e)
        
        # ~~~~ Processor
        for i in range(self.n_messagePassing_layers):
            x = self.processor[i](x,
                                    e,
                                    edge_index,
                                    edge_weight,
                                    halo_info,
                                    mask_send,
                                    mask_recv,
                                    buffer_send,
                                    buffer_recv,
                                    neighboring_procs,
                                    SIZE,
                                    batch)

        # ~~~~ Node decoder
        x = self.node_decoder(x)

        return x

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.processor:
            module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'input_edge_channels': self.input_edge_channels,
             'hidden_channels': self.hidden_channels,
             'output_node_channels': self.output_node_channels,
             'n_mlp_hidden_layers': self.n_mlp_hidden_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'halo_swap_mode': self.halo_swap_mode,
             'name': self.name}
        return a

    def get_save_header(self) -> str:
        a = self.input_dict()
        header = a['name']

        for key in a.keys():
            if key != 'name':
                header += '_' + str(a[key])

        #for item in self.input_dict():
        return header


class MLP(torch.nn.Module):
    def __init__(self,
                 input_channels: int,
                 hidden_channels: List[int],
                 output_channels: int,
                 norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
                 activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU(),
                 bias: bool = True):
        super().__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        self.ic = [input_channels] + hidden_channels # input channel dimensions for each layer
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
                 n_mlp_hidden_layers: int,
                 halo_swap_mode: str):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 
        self.halo_swap_mode = halo_swap_mode

        # Edge update MLP 
        self.edge_updater = MLP(
                input_channels = self.channels*3,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        # Node update MLP
        self.node_updater = MLP(
                input_channels = self.channels*2,
                hidden_channels = [self.channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.channels)
                )

        self.reset_parameters()

        return 

    def forward(self,
            x: Tensor,
            e: Tensor,
            edge_index: LongTensor,
            edge_weight: Tensor,
            halo_info: Tensor,
            mask_send: list,
            mask_recv: list,
            buffer_send: list,
            buffer_recv: list,
            neighboring_procs: Tensor,
            SIZE: Tensor,
            batch: Optional[LongTensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
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

        if SIZE > 1:
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

        return x

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
        uses all_to_all implementation
        """
        if SIZE > 1:
            if self.halo_swap_mode == 'all_to_all':
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
                raise ValueError("halo_swap_mode %s not valid. Valid options: all_to_all, sendrecv" %(self.halo_swap_mode))
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












