#passing message
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool
import torch.nn as nn
import torch

def global_std_pool(x, batch):
    mean=global_mean_pool(x, batch)
    diff_mean=x-mean[batch]
    var=global_mean_pool(diff_mean*diff_mean, batch)
    return torch.sqrt(torch.clamp(var, min=1e-12))

#message passing - nodes: hits, edges: next z layer
#Message Passing Neural Network (MPNN)
class MessagePassModel(MessagePassing):
    def __init__(self, hit_chr, neurons): #hit chr - hit characteristics (7) #neurons - neutrons ex. 24
        super().__init__(aggr="mean") #avarage of characteristics (of hit_chr) bc we have 2 neibourgs or 1 depending on position of node
        self.mlp=nn.Sequential(nn.Linear(hit_chr*2, neurons),
                               nn.ReLU(), nn.Linear(neurons, hit_chr),)
    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x) #pass message

    def message(self, x_j): #message from neigbour node
        return x_j

    def update(self, aggr_nodes, x):
        return self.mlp(torch.cat([x, aggr_nodes], dim=-1)) #we add node characteristics (7) with 7 mean chcarkeristic from neigbour nodes
    #aggr_nodes: mean of neigbour nodes characteristics, x: old characteristics of node, as a result we get new characteristics of this node
    #we connect neigbours

#model of track
class TrackMessPassMod(nn.Module):
    def __init__(self, hit_chr, neurons):
        super().__init__()
        self.mp1=MessagePassModel(hit_chr, neurons) #first tour trough net
        self.mp2=MessagePassModel(hit_chr, neurons) #second tour through net UT -> SciFy -> UT

        self.head=nn.Sequential(nn.Linear(2*hit_chr, neurons), #+1 for max_ut_y_errror
                                nn.ReLU(), nn.Linear(neurons, 1),) #we classify entire track (one vector of characteristics by one track) as output we get is_real_track

#data. - nodes char, data.edge_index- edges, data.batch - info about batch, data.graph_ft -global graph chr - add momentum !!
    def forward(self, data):
        x, edge_index, batch=data.x, data.edge_index, data.batch
        #first message passing
        x=self.mp1(x, edge_index)
        x=torch.relu(x)
        #second message passing
        x=self.mp2(x, edge_index)
        x=torch.relu(x)
        mean=global_mean_pool(x, batch)
        std=global_std_pool(x, batch)
        # g = torch.cat([mean, std,data.graph_ft], dim=1) #if added global features 
        g = torch.cat([mean, std], dim=1)
        is_real_track=self.head(g).view(-1)
        return is_real_track