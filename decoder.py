import os.path
from collections import OrderedDict

import torch
from modules import ResidualAttentionBlock


class SolutionDecoder(torch.nn.Module):
    """
    Convert solution embedding to solution
    input: IP features as condition, solution embedding and key padding mask
    output: IP solution
    """

    def __init__(self, attn_dim: int, n_heads: int, n_layers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn_dim = attn_dim
        self.resblocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(attn_dim, n_heads, attn_mask) for _ in range(n_layers)])

        self.linear_proj = torch.nn.Sequential(OrderedDict([
            ("linear_projection", torch.nn.Linear(attn_dim, 1))
        ]))

    def forward(self, mip_features: torch.Tensor, x_features: torch.Tensor, key_padding_mask: torch.Tensor = None):
        y = torch.concat([mip_features, x_features], dim=1)

        concat_key_padding_mask = key_padding_mask if key_padding_mask is None else torch.concat(
            [key_padding_mask, key_padding_mask], dim=1)
        for module in self.resblocks:
            y = module(y, concat_key_padding_mask)
        y = y[:, -mip_features.shape[1]:, :]

        z = self.linear_proj(y).squeeze(dim=-1)
        z.masked_fill_(key_padding_mask, -torch.inf)

        z = torch.sigmoid(z)
        sols = torch.masked_select(z, ~key_padding_mask)

        return sols

    def apply_model(self, mip_features: torch.Tensor, x_features: torch.Tensor, key_padding_mask: torch.Tensor = None):
        """
        Use the trained model to predict
        """
        with torch.no_grad():
            self.eval()
            output = self(mip_features, x_features, key_padding_mask)
        return output

    def save_model(self, path):
        PATH = os.path.join(path, 'decoder.pth')
        torch.save(self.state_dict(), PATH)
