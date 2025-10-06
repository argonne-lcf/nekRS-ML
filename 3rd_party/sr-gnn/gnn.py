from __future__ import absolute_import, division, print_function, annotations
from typing import Optional, Union, Callable, List
import torch
from torch import Tensor
import torch_geometric.nn as tgnn
from torch_scatter import scatter_mean
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor, PairTensor
from pooling import TopKPooling_Mod, avg_pool_mod, avg_pool_mod_no_x

class GNN(torch.nn.Module):
    def __init__(self, 
                 input_node_channels: int, 
                 input_edge_channels: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_messagePassing_layers: int,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
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
                          MessagePassingLayer(
                                     channels = hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                     ) 
                                  )
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
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
            x,e = self.processor[i](x,e,edge_index,batch)

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

class GNN_Element_Neighbor(torch.nn.Module):
    def __init__(self, 
                 input_node_channels: int, 
                 input_edge_channels: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_messagePassing_layers: int,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.input_edge_channels = input_edge_channels
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
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
                          MessagePassingLayer(
                                     channels = hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                     ) 
                                  )
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            edge_index: LongTensor,
            pos: Tensor,
            batch: Optional[LongTensor] = None,
            edge_index_coin: Optional[LongTensor] = None,
            degree: Optional[Tensor] = None) -> Tensor:

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
            x,e = self.processor[i](x,e,edge_index,batch,edge_index_coin,degree)

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

class GNN_Element_Neighbor_Lo_Hi(torch.nn.Module):
    def __init__(self, 
                 input_node_channels: int, 
                 input_edge_channels_coarse: int, 
                 input_edge_channels_fine: int, 
                 hidden_channels: int, 
                 output_node_channels: int, 
                 n_mlp_hidden_layers: int, 
                 n_messagePassing_layers: int,
                 use_fine_messagePassing: bool,
                 name: Optional[str] = 'gnn'):
        super().__init__()
        
        self.input_node_channels = input_node_channels
        self.input_edge_channels_coarse = input_edge_channels_coarse
        self.input_edge_channels_fine = input_edge_channels_fine
        self.hidden_channels = hidden_channels
        self.output_node_channels = output_node_channels 
        self.n_mlp_hidden_layers = n_mlp_hidden_layers
        self.n_messagePassing_layers = n_messagePassing_layers
        self.use_fine_messagePassing = use_fine_messagePassing
        self.name = name 

        # ~~~~ node encoder MLP  
        self.node_encoder = MLP(
                input_channels = self.input_node_channels,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

        # ~~~~ edge encoder MLP coarse 
        self.edge_encoder_coarse = MLP(
                input_channels = self.input_edge_channels_coarse,
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
        
        # ~~~~ Processor coarse 
        self.processor_coarse = torch.nn.ModuleList()
        for i in range(self.n_messagePassing_layers):
            self.processor_coarse.append( 
                          MessagePassingLayer(
                                     channels = hidden_channels,
                                     n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                     ) 
                                  )

        # ~~~~ Processor fine  
        if self.use_fine_messagePassing: 
            self.edge_encoder_fine = MLP(
                input_channels = self.input_edge_channels_fine,
                hidden_channels = [self.hidden_channels]*(self.n_mlp_hidden_layers+1),
                output_channels = self.hidden_channels,
                activation_layer = torch.nn.ELU(),
                norm_layer = torch.nn.LayerNorm(self.hidden_channels)
                )

            self.processor_fine = torch.nn.ModuleList()
            for i in range(self.n_messagePassing_layers):
                self.processor_fine.append( 
                              MessagePassingLayer(
                                         channels = hidden_channels,
                                         n_mlp_hidden_layers = self.n_mlp_hidden_layers, 
                                         ) 
                                      )
        
        self.reset_parameters()

    def forward(
            self,
            x: Tensor,
            mask: Tensor,
            edge_index_lo: LongTensor,
            edge_index_hi: LongTensor,
            pos_lo: Tensor,
            pos_hi: Tensor,
            batch_lo: Optional[LongTensor] = None,
            batch_hi: Optional[LongTensor] = None,
            edge_index_coin: Optional[LongTensor] = None,
            degree: Optional[Tensor] = None) -> Tensor:

        if batch_lo is None:
            batch_lo = edge_index_lo.new_zeros(pos_lo.size(0))
        if batch_hi is None:
            batch_hi = edge_index_hi.new_zeros(pos_hi.size(0))

        # ~~~~ Compute edge features 
        x_send = x[edge_index_lo[0,:],:]
        x_recv = x[edge_index_lo[1,:],:]
        pos_send = pos_lo[edge_index_lo[0,:],:]
        pos_recv = pos_lo[edge_index_lo[1,:],:]
        e_1 = pos_send - pos_recv
        e_2 = torch.norm(e_1, dim=1, p=2, keepdim=True)
        e_3 = x_send - x_recv
        e = torch.cat((e_1, e_2, e_3), dim=1)

        # ~~~~ Node encoder 
        x = self.node_encoder(x) 

        # ~~~~ Edge encoder 
        e = self.edge_encoder_coarse(e) 

        # ~~~~ Coarse processor
        for i in range(self.n_messagePassing_layers):
            x,e = self.processor_coarse[i](x,e,edge_index_lo,batch_lo,edge_index_coin,degree)

        # ~~~~ Interpolate 
        x = tgnn.unpool.knn_interpolate(
                x = x[mask,:], 
                pos_x = pos_lo[mask,:],
                pos_y = pos_hi,
                batch_x = batch_lo[mask],
                batch_y = batch_hi,
                k = 8)

        if self.use_fine_messagePassing: 
            # ~~~~ Fine edge features
            x_send = x[edge_index_hi[0,:],:]
            x_recv = x[edge_index_hi[1,:],:]
            pos_send = pos_hi[edge_index_hi[0,:],:]
            pos_recv = pos_hi[edge_index_hi[1,:],:]
            e_1 = pos_send - pos_recv
            e_2 = torch.norm(e_1, dim=1, p=2, keepdim=True)
            e_3 = x_send - x_recv
            e = torch.cat((e_1, e_2, e_3), dim=1)
            e = self.edge_encoder_fine(e)

            # ~~~~ Fine mp layers
            for i in range(self.n_messagePassing_layers):
                x,e = self.processor_fine[i](x,e,edge_index_hi,batch_hi)

        # ~~~~ Node decoder 
        x = self.node_decoder(x)
        
        return x 

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        self.edge_encoder_coarse.reset_parameters()
        self.node_decoder.reset_parameters()
        for module in self.processor_coarse:
            module.reset_parameters()
        if self.use_fine_messagePassing:
            self.edge_encoder_fine.reset_parameters()
            for module in self.processor_fine:
                module.reset_parameters()
        return

    def input_dict(self) -> dict:
        a = {'input_node_channels': self.input_node_channels,
             'input_edge_channels_coarse': self.input_edge_channels_coarse,
             'input_edge_channels_fine': self.input_edge_channels_fine,
             'hidden_channels': self.hidden_channels, 
             'output_node_channels': self.output_node_channels,
             'n_mlp_hidden_layers': self.n_mlp_hidden_layers,
             'n_messagePassing_layers': self.n_messagePassing_layers,
             'use_fine_messagePassing': self.use_fine_messagePassing,
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

    def freeze_parameters(self):
        if self.norm_layer:
            self.norm_layer.weight.requires_grad=False
            self.norm_layer.bias.requires_grad=False
        for k in range(len(self.mlp)):
            self.mlp[k].weight.requires_grad = False
            self.mlp[k].bias.requires_grad = False


class MessagePassingLayer(torch.nn.Module):
    def __init__(self, 
                 channels: int, 
                 n_mlp_hidden_layers: int):
        super().__init__()

        self.edge_aggregator = EdgeAggregation(aggr='add')
        self.channels = channels
        self.n_mlp_hidden_layers = n_mlp_hidden_layers 

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

    def forward(
            self,
            x: Tensor,
            e: Tensor,
            edge_index: LongTensor,
            batch: Optional[LongTensor] = None, 
            edge_index_coin: Optional[LongTensor] = None,
            degree: Optional[Tensor] = None) -> Tensor:

        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        
        # ~~~~ Edge update 
        e += self.edge_updater(
                torch.cat((x[edge_index[0,:],:], x[edge_index[1,:],:], e), dim=1)
                )
        
        # ~~~~ Edge aggregation
        edge_agg = self.edge_aggregator(x, edge_index, e)

        # ~~~~ Synchronize neighboring elements 
        if edge_index_coin is not None: 
            edge_agg = edge_agg + self.edge_aggregator(x, edge_index_coin, edge_agg[edge_index_coin[0,:],:])
            edge_agg = edge_agg/degree.unsqueeze(-1)
            #edge_agg_2 = self.edge_aggregator(x, edge_index_coin, edge_agg_send)

            #print(f"edge_agg shape: {edge_agg.shape}")
            #print(f"edge_agg_2 shape: {edge_agg_2.shape}")

            ## What are the values in edge agg, for receiver nodes ? 
            #node_id = set(torch.arange(x.shape[0]).tolist())
            #set_1 = set(torch.unique(edge_index_coin[1,:]).tolist())
            #set_2 = node_id - set_1
            #set_1 = torch.tensor(list(set_1))
            #set_2 = torch.tensor(list(set_2))

            #print(f"receiver node values:     {edge_agg_2[set_1,0]}")
            #print(f"receiver node values [sum, max, min]:     [{edge_agg_2[set_1].sum()}, {edge_agg_2[set_1].max()} {edge_agg_2[set_1].min()}]")
            #print(f"\nNOT receiver node values: {edge_agg_2[set_2,0]}")
            #print(f"NOT receiver node values [sum, max, min]: [{edge_agg_2[set_2].sum()}, {edge_agg_2[set_2].max()}, {edge_agg_2[set_2].min()}]")

        # ~~~~ Node update 
        x += self.node_updater(
                torch.cat((x, edge_agg), dim=1)
                )

        return x,e  

    def reset_parameters(self):
        self.edge_updater.reset_parameters()
        self.node_updater.reset_parameters()
        return

    def freeze_parameters(self):
        self.edge_updater.freeze_parameters()
        self.node_updater.freeze_parameters()
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
