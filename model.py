__auther__ = 'dovkess@gmail.com'

import numpy as np
import torch
from torch import nn
from torch.nn import LSTM
from torch_geometric.nn.models import MLP

'''
The GCPNet calculates GCN layers with input:

For vertices embedding:
    * M_vc  --> Vertices to color matrix, size of v*c (where v is # of vertices, c is number of colors)
    * M_vv  --> Adjacency matrix, size of v*v (where v is # of vertices)
    * C     --> Color embedding, size of #of_colors.
    * V     --> Vertices embedding, size of #of_vertices.

    We then calculate LSTMAgg(M_vv*V AND M_vc*MLP(C, num_layers=3, num_channels=64))

For Color embedding:
    * M_vc^T    --> Vertices to color transposed (color to vertices)
    * V         --> Vertices embedding

    We then calculate LSTMAgg(M_vc^T*MLP(V, num_layers=3, num_channels=64))


The final output of the net would be sigmoid(V_logits <-- V_vote(V))

'''


class GCPNet(nn.Module):
    def __init__(self, embedding_size=64, tmax=32, device=None):
        super(GCPNet, self).__init__()
        self.tmax = tmax
        self.device = device or 'cuda' if torch.cuda.is_available() else 'cpu'
        self.one_tensor = torch.ones(1).to(device)
        # self.one_tensor.requires_grad = False
        self.embedding_size = embedding_size

        self.v_emb = nn.Linear(1, self.embedding_size)
        self.mlpC = MLP(in_channels=self.embedding_size, hidden_channels=self.embedding_size, out_channels=self.embedding_size, num_layers=3)
        self.mlpV = MLP(in_channels=self.embedding_size, hidden_channels=self.embedding_size, out_channels=self.embedding_size, num_layers=3)
        self.LSTM_v = LSTM(input_size=self.embedding_size * 2, hidden_size=self.embedding_size)
        self.LSTM_c = LSTM(input_size=self.embedding_size, hidden_size=self.embedding_size)
        self.V_vote_mlp = MLP(in_channels=self.embedding_size, hidden_channels=self.embedding_size, out_channels=1, num_layers=3)
        self.default_V_h = torch.rand((1, embedding_size)).to(device)
        self.default_C_h = torch.rand((1, embedding_size)).to(device)

    def optimizer_and_loss(self, lr=2e-5, weight_decay=1e-10):
        opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        loss = torch.nn.BCELoss()
        return opt, loss

    def forward(self, M_vv, M_vc, slice_idx):
        V = (torch.normal(mean=0., std=1., size=(M_vv.shape[0], self.embedding_size)) / 8.).to(self.device)
        C = torch.rand(M_vc.shape[1], self.embedding_size).to(self.device)

        inits_shape = (1, self.embedding_size)
        V_h = (self.default_V_h, torch.zeros(inits_shape).to(self.device))
        C_h = (self.default_C_h, torch.zeros(inits_shape).to(self.device))

        for i in range(self.tmax):
            mlp_V = self.mlpV(V)

            # Calculate the new Vertex embedding.
            V, V_h = self.LSTM_v(torch.cat([torch.matmul(M_vv, V), torch.matmul(M_vc, self.mlpC(C))], dim=1),  V_h)
            # Calculate the new Color embedding.
            C, C_h = self.LSTM_c(torch.matmul(M_vc.T, mlp_V), C_h)

        # Calculate the logit probability for each vertex.
        v_vote = self.V_vote_mlp(V)
        v_vote = v_vote.split(slice_idx)
        ret = [torch.sigmoid(v.mean()) for v in v_vote]
        return torch.vstack(ret).squeeze(0)