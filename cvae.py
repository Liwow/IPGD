import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

from encoder import SolutionEncoder, IPEncoder
from decoder import SolutionDecoder
from modules import ResidualAttentionBlock, QuickGELU
from utils import get_padding


class vae_encoder(nn.Module):
    def __init__(self, attn_dim: int, n_heads: int, n_layers: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn_dim = attn_dim
        self.gelu = QuickGELU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(attn_dim, n_heads, attn_mask) for _ in range(n_layers)])

    def forward(self, mip_features: torch.Tensor, x_features: torch.Tensor, key_padding_mask: torch.Tensor = None):
        y = torch.concat([mip_features, x_features], dim=1)

        concat_key_padding_mask = key_padding_mask if key_padding_mask is None else torch.concat(
            [key_padding_mask, key_padding_mask], dim=1)
        for module in self.resblocks:
            y = module(y, concat_key_padding_mask)
        y = y[:, -mip_features.shape[1]:, :]

        return y


class CVAE(nn.Module):
    def __init__(self, embedding=False, latent_size=128, emb_num=3, padding_len=2000):
        super(CVAE, self).__init__()
        self.embedding = embedding
        self.latent_size = latent_size
        self.padding_len = padding_len
        self.emb_num = emb_num
        self.mip_encoder = IPEncoder(emb_size=latent_size)
        self.sol_encoder = SolutionEncoder(emb_num=self.emb_num, attn_dim=latent_size, num_heads=2, layers=2)
        self.embedding = torch.nn.Embedding(self.emb_num, self.latent_size) \
            if self.embedding is True else None
        self.encoder = vae_encoder(attn_dim=latent_size, n_heads=4, n_layers=1, )
        self.mu = nn.Linear(self.latent_size, self.latent_size)
        self.logvar = nn.Linear(self.latent_size, self.latent_size)
        self.decoder = SolutionDecoder(attn_dim=latent_size, n_heads=4, n_layers=2, attn_mask=None)

    def reparameterize(self, mu, log_var):
        # Compute the standard deviation from the log variance
        std = torch.exp(0.5 * log_var)
        # Generate random noise using the same shape as std
        eps = torch.randn_like(std)
        # Return the reparameterized sample
        return mu + eps * std

    def encode_mip(self, mip, n_int_vars):
        mip_features = self.mip_encoder(
            mip.constraint_features,
            mip.edge_index,
            mip.edge_attr,
            mip.variable_features
        )
        # [mip.int_indices.long()]
        mip_features, key_padding_mask = get_padding(mip_features, n_int_vars, self.padding_len, "mip")
        return mip_features, key_padding_mask

    def encode_solution(self, x, n_int_vars):
        x, key_padding_mask = get_padding(x, n_int_vars, self.padding_len, "solution")
        x = self.sol_encoder(x, key_padding_mask)
        return x, key_padding_mask

    def embedding_solution(self, x, n_int_vars):
        x, key_padding_mask = get_padding(x, n_int_vars, self.padding_len, "solution")
        x = self.embedding(x)
        return x, key_padding_mask

    def forward(self, x, mip):
        n_int_vars = mip.n_int_vars
        mip_feature, _ = self.encode_mip(mip, n_int_vars)

        if self.embedding:
            x_feature, key = self.embedding_solution(x, n_int_vars)
        else:
            x_feature, key = self.encode_solution(x, n_int_vars)

        h1 = self.encoder(mip_feature, x_feature, key)

        mu = self.mu(h1)
        logvar = self.logvar(h1)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(mip_feature, z, key)
        BCE, KLD = self.loss_function(recon_x, x, mu, logvar)

        return recon_x, mu, logvar, BCE, KLD

    def sample(self, num_samples, y):
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(num_samples, self.latent_size)
            # Pass the noise through the decoder to generate samples
            mip_feature, key = self.encode_mip(y, y.n_int_vars)
            samples = self.decoder(mip_feature, z, key)
            # Return the generated samples
        return samples

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.float(), reduction='sum')
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE, KLD
