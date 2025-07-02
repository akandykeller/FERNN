import torch
import torch.nn as nn
import random
import torch.nn.functional as F

class FERNN_Cell(nn.Module):
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3, v_range=0):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.v_list = [(x, y) for x in range(-v_range, v_range + 1) for y in range(-v_range, v_range + 1)]
        self.num_v = len(self.v_list)

        # circular convs without bias
        u_pad = u_kernel_size // 2
        h_pad = h_kernel_size // 2
        self.conv_u = nn.Conv2d(input_channels, hidden_channels, u_kernel_size,
                                 padding=u_pad, padding_mode='circular', bias=False)
        self.conv_h = nn.Conv2d(hidden_channels, hidden_channels, h_kernel_size,
                                 padding=h_pad, padding_mode='circular', bias=False)
        self.activation = nn.ReLU()

    def forward(self, u_t, h_prev):
        # u_t: (batch, C, H, W)
        # h_prev: (batch, num_v, hidden, H, W)
        batch, C, H, W = u_t.size()
        # conv_u then expand
        u_conv = self.conv_u(u_t)  # (batch, hidden, H, W)
        u_conv = u_conv.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

        # shift hidden via torch.roll per velocity
        h_shift = []
        for i, (vx, vy) in enumerate(self.v_list):
            h_shift.append(torch.roll(h_prev[:, i], shifts=(vy, vx), dims=(2, 3)))
        h_shift = torch.stack(h_shift, dim=1)  # (batch, num_v, hidden, H, W)

        # conv_h on flattened v dimension
        h_flat = h_shift.view(batch * self.num_v, self.hidden_channels, H, W)
        h_conv = self.conv_h(h_flat)
        h_conv = h_conv.view(batch, self.num_v, self.hidden_channels, H, W)

        # combine and activate
        h_next = self.activation(u_conv + h_conv)
        return h_next


class Seq2SeqFERNN(nn.Module):
    def __init__(self, input_channels, hidden_channels, height, width,
                 output_channels=None, h_kernel_size=3, u_kernel_size=3,
                 v_range=0, pool_type='max', decoder_conv_layers=1):
        super().__init__()
        self.height = height
        self.width = width
        self.pool_type = pool_type
        self.output_channels = output_channels or input_channels

        self.cell = FERNN_Cell(
            input_channels, hidden_channels,
            h_kernel_size, u_kernel_size, v_range)
        self.hidden_channels = hidden_channels
        self.num_v = self.cell.num_v

        decoder = []
        for _ in range(decoder_conv_layers):
            decoder += [nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, padding_mode='circular', bias=False), nn.ReLU()]
        decoder += [nn.Conv2d(hidden_channels, self.output_channels, 3, padding=1, padding_mode='circular', bias=False)]
        self.decoder_conv = nn.Sequential(*decoder)

    def forward(self, input_seq, pred_len, teacher_forcing_ratio=0.0, target_seq=None, return_hidden=False):
        batch, T_in, C, H, W = input_seq.size()
        device = input_seq.device

        if return_hidden:
            input_seq_hiddens = torch.zeros(batch, T_in, self.num_v, self.hidden_channels, H, W, device=device)
            out_seq_hiddens = torch.zeros(batch, pred_len, self.num_v, self.hidden_channels, H, W, device=device)

        # Initialize hidden state
        h = torch.zeros(batch, self.num_v, self.hidden_channels, H, W, device=device)

        # Encoder pass through cell
        for t in range(T_in):
            u_t = input_seq[:, t]
            h = self.cell(u_t, h)

            if return_hidden:
                input_seq_hiddens[:, t] += h.detach()

        prev = input_seq[:, -1]
        outputs = []

        # Decoder
        for t in range(pred_len):
            if self.training and target_seq is not None and random.random() < teacher_forcing_ratio:
                frame = target_seq[:, t]
            else:
                frame = prev.detach()
            h = self.cell(frame, h)

            if return_hidden:
                out_seq_hiddens[:, t] += h.detach()

            # pool over velocities
            if self.pool_type == 'max':
                feat = h.max(1)[0]
            elif self.pool_type == 'mean':
                feat = h.mean(1)
            elif self.pool_type == 'sum':
                feat = h.sum(1)
            else:
                feat = h.max(1)[0]

            out = self.decoder_conv(feat)
            outputs.append(out)
            prev = out

        if return_hidden:
            return torch.stack(outputs, dim=1), input_seq_hiddens, out_seq_hiddens # _, (B, T_in, num_v, C, H, W), (B, T_out, num_v, C, H, W)
        else:
            return torch.stack(outputs, dim=1)
