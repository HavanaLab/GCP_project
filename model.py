
__auther__ = 'dovkess@gmail.com'

import numpy as np
import torch
from torch import nn
from torch.nn import LSTM

from lstm import LayerNormLSTM, SIMPLE_LayerNorm_LSTM, LayerNormLSTMCell
# from torch_geometric.nn.models import MLP
from mlp import MLP
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
        self.histories=[]
        self.clauses_hist=[]
        self.simple_lstm = nn.LSTM(input_size=embedding_size, hidden_size=embedding_size).to(device)




    def elad_style2_bad_delete(self, M_vv, M_vc, slice_idx, cn=0, history = False):
        if history: self.histories = []
        if history: self.clauses_hist = []
        for i in range(1):
            C = self.c_init(self.c_one).expand(cn.sum(), self.embedding_size)
            # C = torch.rand(self.c_rand.size()).to(self.device).expand(cn.sum(), self.embedding_size).clone() #self.c_rand2.expand(cn.sum(), self.embedding_size)
            # C = self.c_rand.expand(cn.sum(), self.embedding_size).clone() #self.c_rand2.expand(cn.sum(), self.embedding_size)
            # V = self.v_init(self.v_one).expand(M_vv.shape[0], self.embedding_size)
            V = self.v_normal.expand(M_vv.shape[0], self.embedding_size).clone()
            # V.normal_(mean=0, std=1)

            V_h = (
                V.unsqueeze(0).clone(), # self.V_h_orig[0].detach().clone().repeat(V.shape[0], 1).unsqueeze(0),
                torch.zeros_like(self.V_h_orig[1]).repeat(V.shape[0], 1).to(self.device).unsqueeze(0)
            )
            C_h = (
                C.unsqueeze(0).clone(), # self.C_h_orig[0].detach().clone().repeat(C.shape[0], 1).unsqueeze(0),
                torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0], 1).to(self.device).unsqueeze(0)#torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0], 1).to(self.device).unsqueeze(0)
            )
            C_h = (C_h[0][:, :1, :], C_h[1][:, :1, :])
            for i in range(self.tmax):
                V = V_h[0].squeeze()
                C = C_h[0].squeeze()
                V_ = self.mlpV(V)

                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), torch.zeros_like(torch.matmul(M_vc, self.mlpC(C)))], dim=1).unsqueeze(0), V_h)
                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), M_vc.shape[1]*self.mlpC(C).repeat(M_vv.size(1),1)], dim=1).unsqueeze(0), V_h)
                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), M_vc.shape[1]*V.sum(0).repeat(M_vv.size(1),1)], dim=1).unsqueeze(0), V_h)
                V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), torch.matmul(M_vc, C)], dim=1).unsqueeze(0), V_h)

                # V, V_h = self.LSTM_v(torch.concat(
                #     [torch.matmul(M_vv, V), (M_vc.shape[1]*self.mlpC(C)).repeat(M_vv.size(1),1) ], dim=1).unsqueeze(0),
                #     # [torch.matmul(M_vv, V), torch.zeros_like((M_vc.shape[1]*self.mlpC(C)).repeat(M_vv.size(1),1)) ], dim=1).unsqueeze(0),
                #                      V_h,
                #                      # (V_h[0], torch.zeros_like(V_h[1])),
                #                      )
                V = V.squeeze(0,1)
                if history: self.histories.append(V)

                # C, C_h = self.LSTM_c(
                #     self.mlpV(V).sum(0).unsqueeze(0).unsqueeze(0),
                #     (torch.zeros_like(C_h[0]), torch.zeros_like(C_h[1])),
                #     # C_h,
                #     # (C_h[0][:, :1, :], C_h[1][:, :1, :]),
                # )
                # C = C.repeat(1, 1, 3, 1)
                # C_h = (C_h[0].repeat(1, 3, 1), C_h[1].repeat(1, 3, 1))

                C, C_h = self.LSTM_c(
                    V_.unsqueeze(0),
                    (torch.zeros_like(C_h[0]), torch.zeros_like(C_h[1])),
                )
                C = C.squeeze(0,1)
                C = self.mlpC(C)
                if history: self.clauses_hist.append(C)
                # print(torch.all(C[0, :] == C[1, :]) and torch.all(C[0, :] == C[2, :]))
        self.history_c = C
        self.history = V
        stacked_pred, stacked_mean, V = self.vote(V, slice_idx)[:3]
        return stacked_pred, stacked_mean, V, C

    def elad_style2(self, M_vv, M_vc, slice_idx, cn=0, history = False, attempts=1):
        if history: self.histories = []
        if history: self.clauses_hist = []
        for i in range(attempts):
            # C = self.c_init(self.c_one).expand(cn.sum(), self.embedding_size)
            C = torch.rand(self.c_rand.size()).to(self.device).expand(cn.sum(), self.embedding_size).clone() #self.c_rand2.expand(cn.sum(), self.embedding_size)
            C = torch.tensor([[0.5045243501663208, 0.5550212264060974, 0.49969482421875, 0.4653627276420593, 0.6495580673217773, 0.5141299962997437, 0.7843188047409058, 0.9166250824928284, 0.046895623207092285, 0.7952654361724854, 0.6014167070388794, 0.24314039945602417, 0.15122956037521362, 0.4336457848548889, 0.9023979306221008, 0.7501546144485474, 0.5971448421478271, 0.06768172979354858, 0.6949077248573303, 0.6882343292236328, 0.1929575800895691, 0.7575528621673584, 0.6478279829025269, 0.6173565983772278, 0.19411027431488037, 0.24367332458496094, 0.7350466251373291, 0.2546481490135193, 0.8495948910713196, 0.17235970497131348, 0.6612893342971802, 0.4750482439994812, 0.08803242444992065, 0.0060392022132873535, 0.22852694988250732, 0.924595832824707, 0.11460268497467041, 0.8744701743125916, 0.6785927414894104, 0.3160938620567322, 0.6726816296577454, 0.7783023715019226, 0.49296504259109497, 0.428031861782074, 0.771365225315094, 0.7714883685112, 0.4873005747795105, 0.6010115742683411, 0.9785568118095398, 0.7439051270484924, 0.9949114918708801, 0.19849419593811035, 0.7321798801422119, 0.030841350555419922, 0.20249396562576294, 0.6240171790122986, 0.9782233238220215, 0.47201651334762573, 0.9281638860702515, 0.4573204517364502, 0.3578141927719116, 0.2193591594696045, 0.7556163668632507, 0.45111334323883057], [0.5045243501663208, 0.5550212264060974, 0.49969482421875, 0.4653627276420593, 0.6495580673217773, 0.5141299962997437, 0.7843188047409058, 0.9166250824928284, 0.046895623207092285, 0.7952654361724854, 0.6014167070388794, 0.24314039945602417, 0.15122956037521362, 0.4336457848548889, 0.9023979306221008, 0.7501546144485474, 0.5971448421478271, 0.06768172979354858, 0.6949077248573303, 0.6882343292236328, 0.1929575800895691, 0.7575528621673584, 0.6478279829025269, 0.6173565983772278, 0.19411027431488037, 0.24367332458496094, 0.7350466251373291, 0.2546481490135193, 0.8495948910713196, 0.17235970497131348, 0.6612893342971802, 0.4750482439994812, 0.08803242444992065, 0.0060392022132873535, 0.22852694988250732, 0.924595832824707, 0.11460268497467041, 0.8744701743125916, 0.6785927414894104, 0.3160938620567322, 0.6726816296577454, 0.7783023715019226, 0.49296504259109497, 0.428031861782074, 0.771365225315094, 0.7714883685112, 0.4873005747795105, 0.6010115742683411, 0.9785568118095398, 0.7439051270484924, 0.9949114918708801, 0.19849419593811035, 0.7321798801422119, 0.030841350555419922, 0.20249396562576294, 0.6240171790122986, 0.9782233238220215, 0.47201651334762573, 0.9281638860702515, 0.4573204517364502, 0.3578141927719116, 0.2193591594696045, 0.7556163668632507, 0.45111334323883057], [0.5045243501663208, 0.5550212264060974, 0.49969482421875, 0.4653627276420593, 0.6495580673217773, 0.5141299962997437, 0.7843188047409058, 0.9166250824928284, 0.046895623207092285, 0.7952654361724854, 0.6014167070388794, 0.24314039945602417, 0.15122956037521362, 0.4336457848548889, 0.9023979306221008, 0.7501546144485474, 0.5971448421478271, 0.06768172979354858, 0.6949077248573303, 0.6882343292236328, 0.1929575800895691, 0.7575528621673584, 0.6478279829025269, 0.6173565983772278, 0.19411027431488037, 0.24367332458496094, 0.7350466251373291, 0.2546481490135193, 0.8495948910713196, 0.17235970497131348, 0.6612893342971802, 0.4750482439994812, 0.08803242444992065, 0.0060392022132873535, 0.22852694988250732, 0.924595832824707, 0.11460268497467041, 0.8744701743125916, 0.6785927414894104, 0.3160938620567322, 0.6726816296577454, 0.7783023715019226, 0.49296504259109497, 0.428031861782074, 0.771365225315094, 0.7714883685112, 0.4873005747795105, 0.6010115742683411, 0.9785568118095398, 0.7439051270484924, 0.9949114918708801, 0.19849419593811035, 0.7321798801422119, 0.030841350555419922, 0.20249396562576294, 0.6240171790122986, 0.9782233238220215, 0.47201651334762573, 0.9281638860702515, 0.4573204517364502, 0.3578141927719116, 0.2193591594696045, 0.7556163668632507, 0.45111334323883057]],
                             device=self.device).expand(cn.sum(), self.embedding_size).clone()
            if history: self.clauses_hist.append(C)

            # C = self.c_rand.expand(cn.sum(), self.embedding_size).clone() #self.c_rand2.expand(cn.sum(), self.embedding_size)
            # V = self.v_init(self.v_one).expand(M_vv.shape[0], self.embedding_size)
            V = self.v_normal.expand(M_vv.shape[0], self.embedding_size).clone()/ ((len(M_vv)-1)//50 +1)
            # V = torch.randn(M_vv.shape[0], self.embedding_size, device=self.device)
            # V.normal_(mean=0, std=1)

            V_h = (
                V.unsqueeze(0).clone(), # self.V_h_orig[0].detach().clone().repeat(V.shape[0], 1).unsqueeze(0),
                torch.zeros_like(self.V_h_orig[1]).repeat(V.shape[0], 1).to(self.device).unsqueeze(0)
            )
            C_h = (
                C.unsqueeze(0).clone(), # self.C_h_orig[0].detach().clone().repeat(C.shape[0], 1).unsqueeze(0),
                torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0], 1).to(self.device).unsqueeze(0)#torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0], 1).to(self.device).unsqueeze(0)
            )
            C_h = (C_h[0][:, :1, :], C_h[1][:, :1, :])
            for i in range(self.tmax):
                V = V_h[0].squeeze()
                C = C_h[0].squeeze()
                V_ = self.mlpV(V)

                C = self.mlpC(C)
                if history: self.clauses_hist.append(C)
                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), torch.zeros_like(torch.matmul(M_vc, self.mlpC(C)))], dim=1).unsqueeze(0), V_h)
                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), M_vc.shape[1]*self.mlpC(C).repeat(M_vv.size(1),1)], dim=1).unsqueeze(0), V_h)
                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), M_vc.shape[1]*V.sum(0).repeat(M_vv.size(1),1)], dim=1).unsqueeze(0), V_h)
                # V, V_h = self.LSTM_v(torch.concat([torch.matmul(M_vv, V), torch.matmul(M_vc, self.mlpC(C))], dim=1).unsqueeze(0), V_h)

                V, V_h = self.LSTM_v(torch.concat(
                    [torch.matmul(M_vv, V), (M_vc.shape[1]*C).repeat(M_vv.size(1),1) ], dim=1).unsqueeze(0),
                    # [torch.matmul(M_vv, V), torch.zeros_like((M_vc.shape[1]*self.mlpC(C)).repeat(M_vv.size(1),1)) ], dim=1).unsqueeze(0),
                                     V_h,
                                     # (V_h[0], torch.zeros_like(V_h[1])),
                                     )
                V = V.squeeze(0,1)
                if history: self.histories.append(V)

                C, C_h = self.LSTM_c(
                    self.mlpV(V).sum(0).unsqueeze(0).unsqueeze(0),
                    # V_.sum(0).unsqueeze(0).unsqueeze(0),
                    # (torch.zeros_like(C_h[0]), torch.zeros_like(C_h[1])),
                    C_h,
                    # (C_h[0][:, :1, :], C_h[1][:, :1, :]),
                )
                # C = C.repeat(1, 1, 3, 1)
                # C_h = (C_h[0].repeat(1, 3, 1), C_h[1].repeat(1, 3, 1))

                # C, C_h = self.LSTM_c(
                #     torch.matmul(
                #         M_vc.T,
                #         V_,
                #     ).unsqueeze(0),
                #     C_h
                # )
                C = C.squeeze(0,1)
                # print(torch.all(C[0, :] == C[1, :]) and torch.all(C[0, :] == C[2, :]))
        self.history_c = C
        self.history = V
        stacked_pred, stacked_mean, V = self.vote(V, slice_idx)[:3]
        return stacked_pred, stacked_mean, V, C

    def vote(self, V, slice_idx):
        v_vote = self.V_vote_mlp(V.squeeze())
        v_vote = v_vote.squeeze().split(slice_idx)
        means = [v.mean() for v in v_vote]
        stacked_means = torch.vstack(means).squeeze()
        pred = [torch.sigmoid(v) for v in means]
        stacked_pred = torch.vstack(pred).squeeze()
        return stacked_pred, stacked_means, V, v_vote


    def forward(self, M_vv, M_vc, slice_idx, cn=0, history = True, attempts=1):
        return self.elad_style2(M_vv, M_vc, slice_idx, cn, history = history, attempts=attempts)
        # return self.elad_simple(M_vv, slice_idx)

    def load_all_attributes(model, state_dict):
        natural_state_dict = model.state_dict()
        # Iterate over all attributes of the model
        for name, value in model.__dict__.items():
            # If the attribute is in the state dict and it's a tensor, load it into the model
            if name in state_dict and name not in natural_state_dict and isinstance(value, torch.Tensor):
                setattr(model, name, state_dict[name].to(model.device))
                del state_dict[name]

        if "history_c" in state_dict: del state_dict["history_c"]
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


    def elad_style3(self, M_vv, M_vc, slice_idx, cn=0, history = False):
        if history: self.histories = []
        # C = self.c_init(self.c_one).expand(cn.sum(), self.embedding_size)
        C = torch.rand(self.c_rand.size()).to(self.device).expand(cn.sum(), self.embedding_size).clone() #self.c_rand2.expand(cn.sum(), self.embedding_size)
        # C = self.c_rand.expand(cn.sum(), self.embedding_size).clone() #self.c_rand2.expand(cn.sum(), self.embedding_size)
        # V = self.v_init(self.v_one).expand(M_vv.shape[0], self.embedding_size)
        V = self.v_normal.expand(M_vv.shape[0], self.embedding_size).clone()
        # V.normal_(mean=0, std=1)
        V_h = (
            V.unsqueeze(0).clone(), # self.V_h_orig[0].detach().clone().repeat(V.shape[0], 1).unsqueeze(0),
            torch.zeros_like(self.V_h_orig[1]).repeat(V.shape[0], 1).to(self.device).unsqueeze(0)
        )
        C_h = (
            C.unsqueeze(0).clone(), # self.C_h_orig[0].detach().clone().repeat(C.shape[0], 1).unsqueeze(0),
            torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0], 1).to(self.device).unsqueeze(0)#torch.zeros_like(self.C_h_orig[1]).repeat(C.shape[0], 1).to(self.device).unsqueeze(0)
        )
        C_h = (C_h[0][:, :1, :], C_h[1][:, :1, :])
        for i in range(self.tmax):
            V = V_h[0].squeeze()

            C, C_h = self.LSTM_c(
                V.sum(0).unsqueeze(0).unsqueeze(0),
                (torch.zeros_like(C_h[0]), torch.zeros_like(C_h[1])),
            )
            C = C.squeeze(0,1)

            V, V_h = self.LSTM_v(torch.concat(
                [torch.matmul(M_vv, V), (M_vc.shape[1]*self.mlpC(C)).repeat(M_vv.size(1),1) ], dim=1).unsqueeze(0),                                 V_h,
                                 )
            V = V.squeeze(0,1)
            if history: self.histories.append(V)


        self.history_c = C
        if history: self.histories.append(V)
        self.history = V
        stacked_pred, stacked_mean, V = self.vote(V, slice_idx)[:3]
        return stacked_pred, stacked_mean, V, C



    def elad_simple(self, M_vv, slice_idx):
        V = self.v_init(self.v_one).expand(M_vv.shape[0], self.embedding_size)
        V_h = (
            V.unsqueeze(0).clone(),
            torch.zeros_like(self.V_h_orig[1]).repeat(V.shape[0], 1).to(self.device).unsqueeze(0)
        )

        V = V_h[0]
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

