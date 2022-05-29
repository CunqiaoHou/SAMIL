import torch
import torch.nn as nn
import torch.nn.functional as F

class att_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, batch_first, output_size):
        super(att_lstm, self).__init__()

        self.attention = AttentionModule()

        self.H = 3
        self.D = 2048
        self.W = torch.nn.Parameter(torch.randn(self.H * self.D, self.D))

        self.LSTM = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layer,
                           batch_first=True
                           )
        self.out = nn.Sequential(nn.Linear(hidden_size, output_size),
                                 nn.Softmax())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, V):
        V = V.squeeze(0)
        A = self.attention(V)
        B = self.attention(V)
        C = self.attention(V)
        V1 = torch.cat([A, B, C], dim=1)
        V1 =torch.mm(V1, self.W)
        V1= x + V1
        V1 = V1.unsqueeze(0)

        r_out, (h_n, h_c) = self.LSTM(V1, None)
        out = self.out(h_n[-1, :, :])
        return out


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()

        self.D_prime = 512
        self.D = 2048

        self.U_t = torch.nn.Parameter(torch.randn(self.D_prime, self.D))
        self.U_t.required_grad = True

        self.W_a = torch.nn.Parameter(torch.randn(1, self.D_prime))
        self.W_a.requires_grad = True

        self.tanh = nn.Tanh()

    def forward(self, V):
        tanh_Ut_V = self.tanh(torch.mm(self.U_t, torch.transpose(V, 0, 1)))
        a = F.softmax(torch.mm(self.W_a, tanh_Ut_V), dim=1)
        a_transposed = torch.transpose(a, 0, 1)

        A = a_transposed.repeat(1, self.D)
        f = torch.mul(V, A)

        return f

