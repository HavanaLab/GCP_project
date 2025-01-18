import torch
import torch.nn as nn
import torch.nn.functional as F

class NormLSTM(nn.Module):
    def __init__(self, dims, clipping=False):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dims))
        self.beta = nn.Parameter(torch.empty(dims))
        self.clipping = clipping

    def forward(self, inputs):
        variance, mean = torch.var_mean(inputs, dim=-1, keepdim=True, correction=0)
        shifted = inputs-mean
        if self.clipping:
            shifted = torch.clamp(shifted, min=-1, max=1)
            variance = ((shifted**2).sum(-1)/(inputs.shape[-1])).unsqueeze(-1)
        res = shifted/torch.sqrt(variance + 1e-12)*self.gamma + self.beta
        return res


class LayerNormLSTMCell(nn.LSTMCell):

    def __init__(self, input_size, hidden_size, activation=torch.tanh, device="cpu", bias=True):
        super().__init__(input_size, hidden_size, bias)
        self.activation = activation
        self.hidden_size = hidden_size
        self.ln_ih = nn.LayerNorm(4 * hidden_size).to(device)
        self.ln_hh = nn.LayerNorm(4 * hidden_size).to(device)
        self.ln_ho = nn.LayerNorm(hidden_size).to(device)

    def forward(self, input, hidden=None):
        if hidden is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        gates = self.ln_ih(F.linear(input, self.weight_ih, self.bias_ih)) + self.ln_hh(F.linear(hx, self.weight_hh, self.bias_hh))
        i, f, o, g = gates.chunk(4, -1)
        i = i.sigmoid()
        f = f.sigmoid()
        o = o.sigmoid()
        g = g.tanh()

        cy = (f * cx) + (i * g)
        hy = o * self.activation(self.ln_ho(cy))
        return hy, (hy, cy)

class LayerNormLSTM(nn.LSTMCell):
    def __init__(self, input_size, hidden_size, num_layers=1, activation=torch.relu, my_layer_norm=True):
        super().__init__(input_size, hidden_size, bias=True)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ln_ih = NormLSTM(hidden_size) if my_layer_norm else nn.LayerNorm(hidden_size)
        self.ln_hf = NormLSTM(hidden_size) if my_layer_norm else nn.LayerNorm(hidden_size)
        self.ln_ho = NormLSTM(hidden_size) if my_layer_norm else nn.LayerNorm(hidden_size)
        self.ln_hc = NormLSTM(hidden_size) if my_layer_norm else nn.LayerNorm(hidden_size)
        self.ln_hcy = NormLSTM(hidden_size) if my_layer_norm else nn.LayerNorm(hidden_size)

        self.fc = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)
        self.activation = activation
        # self.activation = lambda x: x

    def forward(self, input, states):
        seq_len, batch_size, _ = input.size()
        if states is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            cx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            hx, cx = states

        outs = []
        for i in range(seq_len):
            hx, cx = self.lstm_cell(input[i:i+1], (hx, cx))
            outs.append(
                torch.relu(hx)
            )

        out = torch.stack(outs)
        return out, (hx, cx)

    def lstm_cell(self, input, states):
        hx, cx = states
        inp_hx = torch.cat([input, hx], -1)
        gates = self.fc(inp_hx) #kernel
        gates = gates.chunk(4, -1)

        i_gate, f_gate, c_gate, o_gate = gates
        i_gate = self.ln_ih(i_gate)
        f_gate = self.ln_hf(f_gate)
        c_gate = self.ln_hc(c_gate)
        o_gate = self.ln_ho(o_gate)
        f_gate = self.activation(f_gate)

        new_c = ((cx * torch.sigmoid(c_gate+1)) + torch.sigmoid(i_gate) * f_gate)
        new_c = self.ln_hcy(new_c)
        hy = torch.sigmoid(o_gate) * self.activation(new_c)

        return hy, new_c


class SIMPLE_LayerNorm_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, activation=torch.relu, my_layer_norm=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ln_c = nn.LayerNorm(hidden_size)
        self.ln_h = nn.LayerNorm(hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.activation = activation

    def forward(self, input, states):
        seq_len, batch_size, _ = input.size()
        if states is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
            cx = input.new_zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            hx, cx = states

        outs = []
        for i in range(seq_len):
            hx, cx = self.ln_h(hx), self.ln_c(cx)
            o, (hx, cx) = self.lstm(input[i:i+1], (hx, cx))
            outs.append(
                self.activation(o)
            )

        out = torch.stack(outs)
        return out, (hx, cx)

