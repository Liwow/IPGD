import os.path

import torch
import numpy as np

from encoder import IPEncoder, SolutionEncoder
from utils import prenorm, get_padding


class CISP(torch.nn.Module):
    """
    Contrast IP-Solution Pretrain
    """

    def __init__(self, emb_num=3, emb_dim=128, n_heads=2, n_layers=2, padding_len=2000, position_emb=False):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        if position_emb:
            var_nfeats = 25
        else:
            var_nfeats = 13
        self.mip_encoder = IPEncoder(emb_size=emb_dim, var_nfeats=var_nfeats)
        self.sol_encoder = SolutionEncoder(emb_num, emb_dim, n_heads, n_layers)
        self.padding_len = padding_len
        self.softmax = torch.nn.Softmax()

    def mip_prenorm(self, pre_train_loader):
        return prenorm(self.mip_encoder, pre_train_loader)

    def constrast_learning(self, mip_features, x_features):

        mip_features = mip_features / mip_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_mip = logit_scale * mip_features @ x_features.t()  # S(j,k)
        logits_per_x = logits_per_mip.t()
        return logits_per_mip, logits_per_x

    def get_features(self, mip, x=None):
        with torch.no_grad():
            n_int_vars = mip.n_int_vars
            mip_features, key_padding_mask = self.encode_mip(mip, n_int_vars)
            x_features = None
            if x is not None:
                x_features, key_padding_mask = self.encode_solution(x, n_int_vars)
        return mip_features, x_features, key_padding_mask

    def encode_mip(self, mip, n_int_vars):
        mip_features = self.mip_encoder(
            mip.constraint_features,
            mip.edge_index,
            mip.edge_attr,
            mip.variable_features
        )[mip.int_indices.long()]
        mip_features, key_padding_mask = get_padding(mip_features, n_int_vars, self.padding_len, "mip")
        return mip_features, key_padding_mask

    def encode_solution(self, x, n_int_vars):
        x, key_padding_mask = get_padding(x, n_int_vars, self.padding_len, "solution")
        x = self.sol_encoder(x, key_padding_mask)
        return x, key_padding_mask

    def forward(self, mip, x):
        n_batches = mip.n_vars.shape[0]
        n_int_vars = mip.n_int_vars

        mip_features, _ = self.encode_mip(mip, n_int_vars)
        x_features, key_padding_mask = self.encode_solution(x, n_int_vars)

        mip_features = mip_features.view(n_batches, -1)
        x_features = x_features.view(n_batches, -1)

        logits_per_mip, logits_per_x = self.constrast_learning(mip_features, x_features)
        return logits_per_mip, logits_per_x, key_padding_mask

    def save_best(self, path):
        PATH = os.path.join(path, 'cisp_best.pth')
        torch.save(self.state_dict(), PATH)
