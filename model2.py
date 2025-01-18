

import numpy as np
import torch
from torch import nn
from torch.nn import LSTM

from lstm import LayerNormLSTM, SIMPLE_LayerNorm_LSTM, LayerNormLSTMCell
# from torch_geometric.nn.models import MLP
from mlp import MLP

class GCPNet(nn.Module):
    def __init__(self, embedding_size, tmax=32, device='cpu', our_way=True):
        super(GCPNet, self).__init__()
        self.tmax = tmax
        self.device = device
        self.one_tensor = torch.ones(1).to(device)
        self.one_tensor.requires_grad = False
        self.embedding_size = embedding_size
        # self.v_emb = nn.Linear(1, embedding_size) # TODO: try w/ and w/o
        # self.mlpC = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=3)
        self.mlpC = MLP(in_dim=embedding_size, hidden_dim=embedding_size, out_dim=embedding_size).to(device)
        # self.mlpV = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=3)
        self.mlpV = MLP(in_dim=embedding_size, hidden_dim=embedding_size, out_dim=embedding_size).to(device)
        # self.LSTM_v = LSTM(input_size=embedding_size*2, hidden_size=embedding_size)
        self.LSTM_v = LayerNormLSTM(input_size=embedding_size*2, hidden_size=embedding_size).to(device) if our_way else LayerNormLSTMCell(2*embedding_size, embedding_size, activation=torch.relu, device=self.device)#SIMPLE_LayerNorm_LSTM(input_size=embedding_size*2, hidden_size=embedding_size).to(device)
        # self.LSTM_c = LSTM(input_size=embedding_size, hidden_size=embedding_size)
        self.LSTM_c = LayerNormLSTM(input_size=embedding_size, hidden_size=embedding_size).to(device) if our_way else LayerNormLSTMCell(embedding_size, embedding_size, activation=torch.relu, device=self.device)#SIMPLE_LayerNorm_LSTM(input_size=embedding_size, hidden_size=embedding_size).to(device)
        # self.V_vote_mlp = MLP(in_channels=embedding_size, hidden_channels=embedding_size, out_channels=1, num_layers=3)
        self.V_vote_mlp = MLP(in_dim=embedding_size, hidden_dim=embedding_size, out_dim=1).to(device)
        self.V_h_orig = (torch.rand((1, embedding_size)).to(device), torch.zeros((1, embedding_size)).to(device))
        self.C_h_orig = (torch.rand((1, embedding_size)).to(device), torch.zeros((1, embedding_size)).to(device))

        # self.c_rand = (torch.ones(1, self.embedding_size) / self.embedding_size**0.5).to(device)
        self.c_rand = torch.rand(size=(1,self.embedding_size)).to(device)
        # self.c_init = MLP(in_channels=1, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=1)
        self.c_init = MLP(in_dim=1, hidden_dim=embedding_size, out_dim=embedding_size, with_act=False).to(device)
        self.c_one = torch.ones(1).to(self.device)
        # self.v_normal = torch.ones(1, self.embedding_size).to(device) #/ self.embedding_size
        self.v_normal = torch.empty(1, self.embedding_size).normal_(mean=0,std=1).to(device)  # / self.embedding_size
        # self.v_init = MLP(in_channels=1, hidden_channels=embedding_size, out_channels=embedding_size, num_layers=1)
        self.v_init = MLP(in_dim=1, hidden_dim=embedding_size, out_dim=embedding_size, with_act=False).to(device)
        self.v_one = torch.ones(1).to(self.device)
        self.history=torch.ones(1)
        self.simple_lstm = nn.RNN(input_size=embedding_size, hidden_size=embedding_size, ).to(device)

    def forward(self, M_vv, M_vc, slice_idx, cn=0):
        return self.elad_simple(M_vv, slice_idx)

    def load_all_attributes(model, state_dict):
        natural_state_dict = model.state_dict()
        # Iterate over all attributes of the model
        for name, value in model.__dict__.items():
            # If the attribute is in the state dict and it's a tensor, load it into the model
            if name in state_dict and name not in natural_state_dict and isinstance(value, torch.Tensor):
                setattr(model, name, state_dict[name].to(model.device))
                del state_dict[name]

        # Load the parameters of the model
        model.load_state_dict(state_dict)

    def save_all_attributes(model):
        # Get the state dict of the model
        state_dict = model.state_dict()

        # Iterate over all attributes of the model
        for name, value in model.__dict__.items():
            # If the attribute is not in the state dict and it's a tensor, add it to the state dict
            if name not in state_dict and isinstance(value, torch.Tensor):
                state_dict[name] = value

        return state_dict

    def elad_simple(self, M_vv, slice_idx):
        V = self.v_init(self.v_one).expand(M_vv.shape[0], self.embedding_size).unsqueeze(0)
        V_h = torch.zeros_like(V).to(self.device)

        for i in range(self.tmax):
            # TODO try lstm with norm(understand how)
            # TODO to lstm with relu activation
            # TODO try to have tmax different lstm
            V_ = self.mlpV(torch.matmul(M_vv, V))
            V, V_h = self.simple_lstm(V_, V_h)

        V = V.squeeze()
        self.history = V
        v_vote = self.V_vote_mlp(V)
        v_vote = v_vote.squeeze().split(slice_idx)
        means = [v.mean() for v in v_vote]
        stacked_means = torch.vstack(means).squeeze()
        pred = [torch.sigmoid(v) for v in means]
        stacked_pred = torch.vstack(pred).squeeze()
        return stacked_pred, stacked_means, V, None

