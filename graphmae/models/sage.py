import tqdm
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader
from dgl.nn.pytorch import SAGEConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from graphmae.utils import create_activation


class SAGE(nn.Module):
    def __init__(self,
                 in_dim,
                 num_hidden,
                 out_dim,
                 num_layers,
                 dropout,
                 activation,
                 norm,
                 aggr,
                 encoding=False
                 ):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.active = create_activation(activation)
        last_activation = create_activation(activation) if encoding else None
        last_norm = norm(out_dim) if encoding else None

        self.layers = nn.ModuleList()
        if self.num_layers == 1:
            self.layers.append(SAGEConv(in_dim, out_dim, aggr))
        else:
            self.layers.append(SAGEConv(in_dim, num_hidden, aggr))
            for l in range(1, num_layers - 1):
                self.layers.append(SAGEConv(num_hidden, num_hidden, aggr))
            self.layers.append(SAGEConv(num_hidden, out_dim, aggr, norm=last_norm, activation=last_activation))

        self.head = nn.Identity()

    def forward(self, sub, x, return_hidden=False):
        h = x
        hidden_list = []
        for l, layer in enumerate(self.layers):
            h = layer(sub, h)
            if l != self.num_layers - 1:
                h = self.active(h)
                h = self.dropout(h)
                hidden_list.append(h)
        hidden_list.append(h)
        if return_hidden:
            return self.head(h), hidden_list
        else:
            return self.head(h)

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != self.num_layers - 1 else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y



