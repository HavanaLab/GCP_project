
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
    def __init__(self, embedding_size, tmax=32, device='cpu'):
        super(GCPNet, self).__init__()
        self.tmax = tmax
        self.device = device
        self.one_tensor = torch.ones(1).to(device)
        self.one_tensor.requires_grad = False
        self.v_emb = nn.Linear(1, embedding_size)
        self.mlpC = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=3)
        self.mlpV = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=3)
        self.LSTM_v = LSTM(input_size=embedding_size*2, hidden_size=embedding_size)
        self.LSTM_c = LSTM(input_size=embedding_size, hidden_size=embedding_size)
        self.V_vote_mlp = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=1, num_layers=3)

    def forward(self, M_vv, M_vc, C, slice_idx):
        # run for tmax times the message passing process
        V = self.v_emb(self.one_tensor)
        V = V.repeat(1, M_vv.shape[0], 1)
        V_h = (V, torch.zeros(V.shape).to(self.device))
        C_h = (C.unsqueeze(0), torch.zeros(C.unsqueeze(0).shape).to(self.device))
        # V = V.squeeze()

        for i in range(self.tmax):
            V_ = V.clone().squeeze()
            # Calculate the new Vertex embedding.
            _, V_h = self.LSTM_v(torch.cat([torch.matmul(M_vv, V), torch.matmul(M_vc, self.mlpC(C).unsqueeze(0))], dim=2), V_h)
            # Calculate the new Color embedding.
            _, C_h = self.LSTM_c(torch.matmul(M_vc.T, self.mlpV(V_).unsqueeze(0)), C_h)

        # Calculate the logit probability for each vertex.
        v_vote = self.V_vote_mlp(V_h[0].squeeze())
        v_vote = v_vote.split(slice_idx)
        ret = [torch.sigmoid(v.mean()) for v in v_vote]
        return torch.vstack(ret).squeeze(), V, C
