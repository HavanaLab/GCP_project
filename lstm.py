import torch
import torch.nn as nn


class NormLSTM(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dims))
        self.beta = nn.Parameter(torch.zeros(dims))

    def moments_to_torch(self, inputs):
        # Calculate mean and variance along the last dimension (-1)
        mean = torch.mean(inputs, dim=-1, keepdim=True)
        variance = torch.var(inputs, dim=-1, keepdim=True, unbiased=False)
        return mean, variance

    def forward(self, inputs):
        mean, variance = self.moments_to_torch(inputs)
        inv = torch.rsqrt(variance + 1e-12)
        invg = inv * self.gamma
        return inputs*invg + (self.beta - mean*invg)

class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, activation=torch.relu):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.ln_ih = NormLSTM(hidden_size)
        self.ln_hf = NormLSTM(hidden_size)
        self.ln_ho = NormLSTM(hidden_size)
        self.ln_hc = NormLSTM(hidden_size)
        self.ln_hcy = NormLSTM(hidden_size)

        self.fc_ih = nn.Linear(input_size, 4 * hidden_size)
        self.fc_hh = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc = nn.Linear(input_size + hidden_size, 4 * hidden_size, bias=False)
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
            hx, cx = self.lstm_cell2(input[i:i+1], (hx, cx))
            outs.append(
                torch.relu(hx)
            )

        out = torch.stack(outs)
        return out, (hx, cx)

    # def lstm_cell(self, input, states):
    #     hx, cx = states
    #     gates = self.fc_ih(input) + self.fc_hh(hx)
    #     gates = gates.chunk(4, -1)
    #
    #     i_gate, f_gate, c_gate, o_gate = gates
    #     i_gate = torch.sigmoid(self.ln_ih(i_gate))
    #     f_gate = torch.sigmoid(self.ln_ih(f_gate))
    #     c_gate = torch.tanh(self.ln_ih(c_gate))
    #     o_gate = torch.sigmoid(self.ln_ih(o_gate))
    #
    #     cy = (f_gate * cx) + (i_gate * c_gate)
    #     hy = o_gate * torch.tanh(self.ln_ho(cy))
    #
    #     return hy, cy

    def lstm_cell2(self, input, states):
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
