import torch
# Now let's create a model
from torch_geometric.nn import GCNConv, GraphConv
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn

class MO_GNN_large(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels,num_layers,dropout=0.2,seed=12345):
        super(MO_GNN_large, self).__init__()
        # seed
        torch.manual_seed(seed)
        # Create the layers
        self.MLP = torch.nn.Linear(in_channels,hidden_channels)
        self.MLP2 = torch.nn.Linear(hidden_channels,hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.init_conv = GCNConv(in_channels, hidden_channels)
        self.bn_extra = torch.nn.BatchNorm1d(hidden_channels)
        self.init_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn_extra_2 = torch.nn.BatchNorm1d(hidden_channels)
        cached = False
        add_self_loops = True
        save_mem  = False
        self.bn = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))
        # Final layer
        #self.fc1 = Linear((num_layers + 2)*hidden_channels, out_channels)
        self.fc1 = Linear(hidden_channels, out_channels)
        # Attention mechanism
        self.att = nn.Parameter(torch.ones(num_layers + 2))
        self.sm = nn.Softmax(dim=0)
        # Dropout
        self.dropout = dropout
    def forward(self, x, edge_indexes, edge_attrs):
        mask = self.sm(self.att)
        # GCNConv over the original graph
        extra_conv = self.init_conv(x, edge_indexes[-1]).relu()
        extra_conv = F.dropout(extra_conv, p=0.5, training=self.training)
        extra_conv = self.init_conv2(extra_conv, edge_indexes[-1]).relu() * mask[-1]
        # GCNConv over the n hops of graph
        embeddings = list()
        for i, conv in enumerate(self.convs):
            tmp_embedding = conv(x, edge_indexes[i]).relu() * mask[i]
            embeddings.append(tmp_embedding.unsqueeze(0))
        # MLP over the features of the graph
        x = self.MLP(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.MLP2(x).relu() * mask[-2]
        # Sum all the embeddings
        final_embedding = torch.cat(embeddings,dim=0)
        final_embedding = torch.cat([final_embedding, x.unsqueeze(0)], dim=0)
        final_embedding = torch.cat([final_embedding, extra_conv.unsqueeze(0)], dim=0)
        # Sum all the embeddings
        final_embedding = final_embedding.sum(dim=0)
        z = F.dropout(final_embedding, p=self.dropout, training=self.training)
        z = self.fc1(z).log_softmax(dim=-1)
        return z
class MO_GNN_large_xl(torch.nn.Module):
    def __init__(self, in_channels,hidden_channels, out_channels,num_layers,dropout=0.2,seed=12345):
        super(MO_GNN_large_xl, self).__init__()
        # seed
        torch.manual_seed(seed)
        # Create the layers
        self.MLP = torch.nn.Linear(in_channels,hidden_channels)
        self.MLP2 = torch.nn.Linear(hidden_channels,hidden_channels)
        self.convs = torch.nn.ModuleList()
        self.init_conv = GCNConv(in_channels, hidden_channels)
        self.bn_extra = torch.nn.BatchNorm1d(hidden_channels)
        self.init_conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn_extra_2 = torch.nn.BatchNorm1d(hidden_channels)
        cached = False
        add_self_loops = True
        save_mem  = False
        self.bn = torch.nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bn.append(torch.nn.BatchNorm1d(hidden_channels))
        # Final layer
        #self.fc1 = Linear((num_layers + 2)*hidden_channels, out_channels)
        self.fc1 = Linear(hidden_channels, out_channels)
        # Attention mechanism
        self.att = nn.Parameter(torch.ones(num_layers + 2))
        self.sm = nn.Softmax(dim=0)
        # Dropout
        self.dropout = dropout
    def forward(self, x, edge_indexes, edge_attrs):
        mask = self.sm(self.att)
        # GCNConv over the original graph
        extra_conv = self.init_conv(x, edge_indexes[-1])
        extra_conv = self.bn_extra(extra_conv).relu()
        extra_conv = F.dropout(extra_conv, p=0.5, training=self.training)
        extra_conv = self.init_conv2(extra_conv, edge_indexes[-1])
        extra_conv = self.bn_extra_2(extra_conv).relu() * mask[-1]
        # GCNConv over the n hops of graph
        embeddings = list()
        for i, conv in enumerate(self.convs):
            #tmp_embedding = conv(x, edge_indexes[i]).relu() * mask[i]
            tmp_embedding = conv(x, edge_indexes[i])
            tmp_embedding = self.bn[i](tmp_embedding).relu() * mask[i]
            embeddings.append(tmp_embedding.unsqueeze(0))
        # MLP over the features of the graph
        x = self.MLP(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.MLP2(x).relu() * mask[-2]
        # Sum all the embeddings
        final_embedding = torch.cat(embeddings,dim=0)
        final_embedding = torch.cat([final_embedding, x.unsqueeze(0)], dim=0)
        final_embedding = torch.cat([final_embedding, extra_conv.unsqueeze(0)], dim=0)
        
        # Sum all the embeddings
        final_embedding = final_embedding.sum(dim=0)
        z = F.dropout(final_embedding, p=self.dropout, training=self.training)
        z = self.fc1(z).log_softmax(dim=-1)
        return z