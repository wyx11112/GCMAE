from typing import Optional
from itertools import chain
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gin import GIN
from .gat import GAT
from .gcn import GCN
from .sage import SAGE
from .dot_gat import DotGAT
from .loss_func import sce_loss
from graphmae.utils import create_norm, drop_edge, drop_node, degree_drop_weights, pr_drop_weights, evc_drop_weights, \
    feature_drop_weights, compute_pr, eigenvector_centrality, drop_edge_weighted, drop_feature_weighted

import dgl


def setup_module(m_type, enc_dec, in_dim, num_hidden, out_dim, num_layers, dropout, activation, residual, norm, nhead,
                 nhead_out, attn_drop, aggr, negative_slope=0.2, concat_out=True) -> nn.Module:
    if m_type == "gat":
        mod = GAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "sage":
        mod = SAGE(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
            norm=create_norm(norm),
            aggr=aggr,
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "dotgat":
        mod = DotGAT(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            concat_out=concat_out,
            activation=activation,
            feat_drop=dropout,
            attn_drop=attn_drop,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gin":
        mod = GIN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=norm,
            encoding=(enc_dec == "encoding"),
        )
    elif m_type == "gcn":
        mod = GCN(
            in_dim=in_dim,
            num_hidden=num_hidden,
            out_dim=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
            residual=residual,
            norm=create_norm(norm),
            encoding=(enc_dec == "encoding")
        )
    elif m_type == "mlp":
        # * just for decoder
        mod = nn.Sequential(
            nn.Linear(in_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_hidden, out_dim)
        )
    elif m_type == "linear":
        mod = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError

    return mod


class PreModel(nn.Module):
    def __init__(
            self,
            in_dim: int,
            num_hidden: int,
            num_projector_hidden: int,
            num_projector: int,
            num_layers: int,
            nhead: int,
            nhead_out: int,
            activation: str,
            feat_drop: float,
            attn_drop: float,
            negative_slope: float,
            residual: bool,
            norm: Optional[str],
            mask_rate: float = 0.3,
            temperature: float = 0.4,
            encoder_type: str = "gat",
            decoder_type: str = "gat",
            loss_fn: str = "sce",
            loss_weight: float = 0.5,
            mu: float = 0.5,
            nu: float = 0.5,
            augmentation: str = "drop_node",
            drop_node_rate: float = 0.0,
            drop_edge_rate: float = 0.0,
            drop_feature_rate: float = 0.0,
            replace_rate: float = 0.1,
            alpha_l: float = 2,
            concat_hidden: bool = False,
            aggr: str = None,
         ):
        super(PreModel, self).__init__()
        self._mask_rate = mask_rate
        self._temperature = temperature

        self._encoder_type = encoder_type
        self._decoder_type = decoder_type
        self._augmentation = augmentation
        self._drop_node_rate = drop_node_rate
        self._drop_edge_rate = drop_edge_rate
        self._drop_feature_rate = drop_feature_rate
        self._output_hidden_size = num_hidden
        self._num_projector = num_projector
        self._concat_hidden = concat_hidden

        self._replace_rate = replace_rate
        self._mask_token_rate = 1 - self._replace_rate
        self._loss_weight = loss_weight
        self._mu = mu
        self._nu = nu

        assert num_hidden % nhead == 0
        assert num_hidden % nhead_out == 0
        if encoder_type in ("gat", "dotgat"):
            enc_num_hidden = num_hidden // nhead
            enc_nhead = nhead
        else:
            enc_num_hidden = num_hidden
            enc_nhead = 1

        dec_in_dim = num_hidden
        dec_num_hidden = num_hidden // nhead_out if decoder_type in ("gat", "dotgat") else num_hidden

        self._dec_in_dim = dec_in_dim

        # build encoder
        self.encoder = setup_module(
            m_type=encoder_type,
            enc_dec="encoding",
            in_dim=in_dim,
            num_hidden=enc_num_hidden,
            out_dim=enc_num_hidden,
            num_layers=num_layers,
            nhead=enc_nhead,
            nhead_out=enc_nhead,
            concat_out=True,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            aggr=aggr,
        )

        # build decoder for attribute prediction
        self.decoder = setup_module(
            m_type=decoder_type,
            enc_dec="decoding",
            in_dim=dec_in_dim,
            num_hidden=dec_num_hidden,
            out_dim=in_dim,
            num_layers=1,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            dropout=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=True,
            aggr=aggr,
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.std_expander = nn.Sequential(nn.Linear(num_hidden, num_hidden),
                                          nn.PReLU())
        if concat_hidden:
            self.encoder_to_decoder = nn.Linear(dec_in_dim * num_layers, dec_in_dim, bias=False)
        else:
            self.encoder_to_decoder = nn.Linear(dec_in_dim, dec_in_dim, bias=False)

        self.projector_fc1 = nn.Sequential(nn.Linear(num_hidden, num_projector_hidden, bias=True),
                                           nn.PReLU(),
                                           nn.Linear(num_projector_hidden, num_projector, bias=True)
                                           )
        self.projector_fc2 = nn.Sequential(nn.Linear(num_hidden, num_projector_hidden, bias=True),
                                           nn.PReLU(),
                                           nn.Linear(num_projector_hidden, num_projector, bias=True)
                                           )
        # * setup loss function
        self.criterion = self.setup_loss_fn(loss_fn, alpha_l)

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self._temperature)
        inter_sim = f(self.sim(z1, z1))
        intra_sim = f(self.sim(z1, z2))
        return -torch.log(intra_sim.diag() / inter_sim.sum(1) + intra_sim.sum(1) - inter_sim.diag())

    def contrastive_loss(self, emb):
        h1 = self.projector_fc1(emb)
        h2 = self.projector_fc2(emb)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        loss = (l1 + l2) * 0.5
        return loss.mean()

    @property
    def output_hidden_dim(self):
        return self._output_hidden_size

    def setup_loss_fn(self, loss_fn, alpha_l):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion

    def encoding_mask_noise(self, g, x, mask_rate=0.3):
        num_nodes = g.num_nodes()
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_g = g.clone()

        return use_g, out_x, (mask_nodes, keep_nodes)

    def forward(self, g, x):
        # ---- attribute reconstruction ----
        loss = self.mask_attr_prediction(g, x)
        loss_item = {"loss": loss.item()}
        return loss, loss_item

    def mask_attr_prediction(self, g, x):

        pre_use_g, use_x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(g, x, self._mask_rate)

        if self._augmentation == 'drop_node':
            aug_g, drop_idx = drop_node(g, self._drop_node_rate, return_mask_nodes=True)
            aug_x = x[drop_idx]
        elif self._augmentation == 'drop_edge':
            aug_g, masked_edges = drop_edge(g, self._drop_edge_rate, return_edges=True)
            aug_x = x
        elif self._augmentation in ['degree', 'pr', 'evc']:
            num_nodes = g.num_nodes()
            device = g.device
            g = g.to('cpu')
            x = g.ndata['feat']
            # x = g.ndata['attr']
            if self._augmentation == 'degree':
                g_ = g
                node_c = g_.in_degrees()
                drop_weights = degree_drop_weights(g)
                feature_weights = feature_drop_weights(x, node_c)
            elif self._augmentation == 'pr':
                node_c = compute_pr(g)
                drop_weights = pr_drop_weights(g, aggr='source', k=200)
                feature_weights = feature_drop_weights(x, node_c)
            elif self._augmentation == 'evc':
                node_c = eigenvector_centrality(g)
                drop_weights = evc_drop_weights(g)
                feature_weights = feature_drop_weights(x, node_c)
            else:
                drop_weights = None
                feature_weights = None
            edge_ = drop_edge_weighted(g.edges(), drop_weights, self._drop_edge_rate, threshold=0.7)
            x_ = drop_feature_weighted(x, feature_weights, self._drop_feature_rate)
            aug_g = dgl.graph((edge_[0], edge_[1]), num_nodes=num_nodes)
            aug_g = aug_g.add_self_loop()
            aug_g.ndata['feat'] = x_.to('cpu')
            aug_g = aug_g.to(device)
            aug_x = x_.to(device)
        else:
            raise NotImplementedError

        # shared encoder
        enc_rep, all_hidden = self.encoder(pre_use_g, use_x, return_hidden=True)
        enc_edge, all_edge_hidden = self.encoder(aug_g, aug_x, return_hidden=True)

        if self._concat_hidden:
            enc_rep = torch.cat(all_hidden, dim=1)

        # ---- attribute reconstruction ----
        rep = self.encoder_to_decoder(enc_rep)

        if self._decoder_type not in ("mlp", "linear"):
            # * remask, re-mask
            rep[mask_nodes] = 0

        if self._decoder_type in ("mlp", "liear"):
            recon = self.decoder(rep)
        else:
            recon = self.decoder(pre_use_g, rep)

        x_init = x[mask_nodes].to(g.device)
        x_rec = recon[mask_nodes].to(g.device)

        loss_rec = self.criterion(x_rec, x_init)
        loss = loss_rec + self._loss_weight * self.contrastive_loss(enc_edge) \
            + self._mu * self.reconstruct_adj_mse(g, enc_rep) + self._nu * self.std_loss(enc_rep)
        return loss

    def sage_embed(self, g, device, batch_size):
        return self.encoder.inference(g, device, batch_size)

    def embed(self, g, x):
        rep = self.encoder(g, x)
        return rep

    def reconstruct_adj(self, g, x):
        emb = self.encoder(g, x)
        adj = torch.sigmoid((emb @ emb.t()))

        return adj

    def reconstruct_adj_mse(self, g, emb):
        adj = g.adj().to_dense()
        adj = adj.to(emb.device)
        res_adj = (emb @ emb.t())
        res_adj = F.sigmoid(res_adj)
        relative_distance = (adj * res_adj).sum() / (res_adj * (1 - adj)).sum()
        cri = torch.nn.MSELoss()
        res_loss = cri(adj, res_adj) + F.binary_cross_entropy_with_logits(adj, res_adj)
        loss = res_loss + relative_distance

        return loss

    def std_loss(self, z):
        z = self.std_expander(z)
        z = F.normalize(z, dim=1)
        std_z = torch.sqrt(z.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z))
        return std_loss

