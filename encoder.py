import torch_geometric
import torch

from modules import GNNModel, ResidualAttentionBlock
from base_model import BaseModel

class IPEncoder(BaseModel):
    
    """
    Convert IP problem into IP embedding
    input: features from instance
    output: IP embeddings
    """
    
    def __init__(self, emb_size = 128, cons_nfeats = 4, edge_nfeats = 1, var_nfeats = 13, gcn_mlp_layer_num = 2):
        super().__init__()
        self.gnn_model = GNNModel(emb_size, cons_nfeats, edge_nfeats, var_nfeats, gcn_mlp_layer_num=gcn_mlp_layer_num)
        self.mean_aggr = torch_geometric.nn.aggr.MeanAggregation()


    def forward(self, constraint_features, edge_indices, edge_features, variable_features):
        variable_features = self.gnn_model(constraint_features, edge_indices, edge_features, variable_features)

        return variable_features
    
class SolutionEncoder(torch.nn.Module):
    """
    Convert solved IP solutions into solution embedding
    input: solutions
    output: solution embedding
    """
    def __init__(self, emb_num: int, attn_dim: int, num_heads:int, layers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.embedding = torch.nn.Embedding(emb_num, attn_dim)
        self.width = attn_dim
        self.layers = layers
        self.resblocks = torch.nn.ModuleList([ResidualAttentionBlock(attn_dim, num_heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        x = self.embedding(x)
        for module in self.resblocks:
            x = x + module(x, key_padding_mask)
        return x