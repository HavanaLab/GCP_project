
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
        self.embedding_size = embedding_size
        # self.v_emb = nn.Linear(1, embedding_size) # TODO: try w/ and w/o
        self.mlpC = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=3)
        self.mlpV = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=3)
        self.LSTM_v = LSTM(input_size=embedding_size*2, hidden_size=embedding_size)
        self.LSTM_c = LSTM(input_size=embedding_size, hidden_size=embedding_size)
        self.V_vote_mlp = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=1, num_layers=3)
        self.V_h_orig = (torch.rand((1, embedding_size)).to(device), torch.zeros((1, embedding_size)).to(device))
        self.C_h_orig = (torch.rand((1, embedding_size)).to(device), torch.zeros((1, embedding_size)).to(device))

        self.c_rand = torch.ones(1, self.embedding_size) / self.embedding_size
        self.c_init = MLP(in_channels=1, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=1)
        self.c_one = torch.ones(1).to(self.device)
        self.v_normal = torch.ones(1, self.embedding_size) / self.embedding_size
        self.v_init = MLP(in_channels=1, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=1)
        self.v_one = torch.ones(1).to(self.device)

    def get_vh(self):
        return (self.V_h[0].detach(), self.V_h[1].detach())

    def set_vh(self, V_h):
        self.V_h = V_h

    def get_ch(self):
        return (self.C_h[0].detach(), self.C_h[1].detach())

    def set_ch(self, C_h):
        self.C_h = C_h

    def dov_style(self, M_vv, M_vc, slice_idx, cn=[0]):
        mult_coef = 1
        slice_idx = slice_idx * mult_coef
        M_vv = torch.block_diag(*([M_vv] * mult_coef))
        M_vc = torch.block_diag(*([M_vc] * mult_coef))
        c_rand = torch.ones(1, self.embedding_size) / self.embedding_size
        C = c_rand.expand(int(cn.sum() * mult_coef), self.embedding_size).to(self.device)
        v_normal = torch.ones(1, self.embedding_size) / self.embedding_size
        v_size = M_vv.shape[0]
        V = v_normal.expand(v_size, self.embedding_size).to(self.device)

        self.V_h = (self.V_h_orig[0].detach().clone(), torch.zeros_like(self.V_h_orig[1]).to(self.device))
        self.C_h = (self.C_h_orig[0].detach().clone(), torch.zeros_like(self.C_h_orig[1]).to(self.device))

        for i in range(self.tmax):
            V_ = V.clone().squeeze()
            V, self.V_h = self.LSTM_v(torch.cat([torch.matmul(M_vv, V), torch.matmul(M_vc, self.mlpC(C))], dim=1), self.V_h)
            # Calculate the new Color embedding.
            C, self.C_h = self.LSTM_c(torch.matmul(M_vc.T, self.mlpV(V_)), self.C_h)

        # Calculate the logit probability for each vertex.
        v_vote = self.V_vote_mlp(V.squeeze())
        v_vote = v_vote.split(slice_idx)
        ret = [torch.sigmoid(v.mean()) for v in v_vote]
        stacked_ret = torch.vstack(ret).squeeze()
        return stacked_ret[:len(stacked_ret) // mult_coef], V, C

    def elad_style(self, M_vv, M_vc, slice_idx, cn=0):
        # C = self.c_rand.expand(int(cn.sum()), self.embedding_size).to(self.device)
        C = self.c_init(self.c_one).expand(cn.sum(), self.embedding_size)
        # V = self.v_normal.expand(M_vv.shape[0], self.embedding_size).to(self.device)
        V = self.v_init(self.v_one).expand(M_vv.shape[0], self.embedding_size)
        V_h = (self.V_h_orig[0].detach().clone().repeat(V.shape[0],1).unsqueeze(0), torch.zeros_like(self.V_h_orig[1]).repeat(V.shape[0],1).to(self.device).unsqueeze(0))
        C_h = (self.C_h_orig[0].detach().clone().repeat(C.shape[0],1).unsqueeze(0), torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0],1).to(self.device).unsqueeze(0))

        for i in range(self.tmax):
            V_ = self.mlpV(V)
            # Calculate the new Vertex embedding.
            V, V_h = self.LSTM_v(torch.cat([torch.matmul(M_vv, V), torch.matmul(M_vc, self.mlpC(C))], dim=1).unsqueeze(0), V_h)
            V = V.squeeze(0)
            # Calculate the new Color embedding.
            C, C_h = self.LSTM_c(torch.matmul(M_vc.T, V_).unsqueeze(0), C_h)
            C = C.squeeze(0)

        # Calculate the logit probability for each vertex.
        v_vote = self.V_vote_mlp(V.squeeze())
        v_vote = v_vote.split(slice_idx)
        ret = [torch.sigmoid(v.mean()) for v in v_vote]
        stacked_ret = torch.vstack(ret).squeeze()
        return stacked_ret, V, C

    def forward(self, M_vv, M_vc, slice_idx, cn=0, dov=False):
        if dov:
            return self.dov_style(M_vv, M_vc, slice_idx, cn)
        return self.elad_style(M_vv, M_vc, slice_idx, cn)
