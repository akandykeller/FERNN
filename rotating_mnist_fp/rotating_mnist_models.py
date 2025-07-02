import torch
import torch.nn as nn
import random
import torch.nn.functional as F
from escnn import gspaces, nn as e2nn              # `escnn` ≥ 0.1
import math

def build_rot_equivariant_conv(in_type, out_channels, kernel_size):
    """Return an R2Conv that is equivariant to the discrete rotation group."""
    gspace = in_type.gspace                       # same g-space
    out_type = e2nn.FieldType(
        gspace, out_channels * [gspace.regular_repr]
    )
    padding = kernel_size // 2                    # keep spatial size
    conv = e2nn.R2Conv(in_type, out_type,
                       kernel_size=kernel_size,
                       padding=padding, bias=False)
    return conv, out_type

class RotationFlowRNNCell(nn.Module):
    """
    One step of a rotation-flow equivariant FERNN.
    The hidden state has shape (B, num_ω, hidden, H, W).
    """
    def __init__(self, input_channels, hidden_channels,
                 h_kernel_size=3, u_kernel_size=3,
                 v_list=[-40, -20, 0, 20, 40], N=8):
        super().__init__()
        # --- group & field types ------------------------------------------------
        self.N = N
        self.gspace      = gspaces.rot2dOnR2(N=N)        # C_N group
        in_type_u        = e2nn.FieldType(self.gspace,
                                          input_channels * [self.gspace.trivial_repr])
        self.in_type_u   = in_type_u
        self.v_list      = v_list  # list of angular velocities
        self.num_v       = len(self.v_list)

        # input to hidden ---------------------------------------------------------
        self.conv_u, hid_type = build_rot_equivariant_conv(in_type_u,
                                                           hidden_channels,
                                                           u_kernel_size)
        self.hid_type = hid_type

        # hidden to hidden --------------------------------------------------------
        self.conv_h, _ = build_rot_equivariant_conv(hid_type,
                                                    hidden_channels,
                                                    h_kernel_size)
        self.hidden_channels = hidden_channels
        self.activation      = nn.ReLU()

    # -------------------------------------------------------------------------
    def act_on_hidden(self, tensor, k):
        """
        tensor : (B, C, H, W) with C = hidden_channels * N
        k      : angle to rotate in degrees
        N      : group order
        """
        # 1) cyclically permute orientation channels (regular repr.)
        B, C, H, W = tensor.shape
        C0 = C // self.N
        tensor = tensor.view(B, C0, self.N, H, W)          # split orientation axis
        g_idx = int(k * self.N / 360) 
        tensor = torch.roll(tensor, shifts=g_idx, dims=2) # shift d_{g_i} → d_{g_{i+k}}
        tensor = tensor.view(B, C, H, W)

        # 2) rotate the spatial grid by the same angle
        if k % 90 == 0:                                # multiples of 90deg — cheap path
            return torch.rot90(tensor, k // 2, dims=(2, 3))

        # Convert degrees to radians
        angle_rad = math.radians(k)
        rot_mat = tensor.new_tensor([[ math.cos(angle_rad), -math.sin(angle_rad), 0],
                                    [ math.sin(angle_rad),  math.cos(angle_rad), 0]])
        # Expand rotation matrix to match batch size
        rot_mat = rot_mat.unsqueeze(0).expand(B, -1, -1)
        grid = F.affine_grid(rot_mat, tensor.size(),
                            align_corners=False)
        return F.grid_sample(tensor, grid, align_corners=False,
                            padding_mode='zeros')


    # -------------------------------------------------------------------------
    def forward(self, u_t, h_prev):
        """
        u_t     : (B, C, H, W)
        h_prev  : (B, num_ω, hidden, H, W)
        returns : h_next with same shape as h_prev
        """
        B, _, H, W = u_t.shape
        # lift input frame to g-space ------------------------------------------
        u_feat = e2nn.GeometricTensor(u_t, self.in_type_u)
        u_conv = self.conv_u(u_feat).tensor                # (B, hidden, H, W)
        u_conv = u_conv.unsqueeze(1).expand(-1, self.num_v, -1, -1, -1)

        # rotate hidden slices by their w --------------------------------------
        h_rot = []
        for i, w in enumerate(self.v_list):
            h_slice = h_prev[:, i]                         # (B, hidden, H, W)
            h_rot.append(self.act_on_hidden(h_slice, w))
        h_rot = torch.stack(h_rot, dim=1)                  # (B, num_ω, hidden, H, W)

        # group-equivariant convolution on each slice --------------------------
        h_flat = h_rot.reshape(B * self.num_v, self.hidden_channels * self.N, H, W)
        h_conv = self.conv_h(e2nn.GeometricTensor(h_flat,
                                                  self.conv_h.in_type)).tensor
        h_conv = h_conv.view(B, self.num_v, self.hidden_channels * self.N, H, W)

        # recurrence -----------------------------------------------------------
        h_next = self.activation(u_conv + h_conv)
        return h_next


class Seq2SeqRotationRNN(nn.Module):
    """
    Rotation-flow equivariant sequence-to-sequence model.
    Matches the I/O signature of your original Galilean model.
    """
    def __init__(self, input_channels, hidden_channels,
                 height, width, output_channels=None,
                 h_kernel_size=3, u_kernel_size=3,
                 v_list=[-40, -20, 0, 20, 40], N=8, pool_type='max',
                 decoder_conv_layers=1):
        super().__init__()
        self.height, self.width = height, width
        self.pool_type          = pool_type
        self.output_channels    = output_channels or input_channels

        self.cell = RotationFlowRNNCell(input_channels, hidden_channels,
                                         h_kernel_size, u_kernel_size,
                                         v_list=v_list, N=N)
        self.hidden_channels = hidden_channels
        self.total_hidden_channels = hidden_channels * N
        self.num_v = self.cell.num_v
        self.N = N

        # decoder (ordinary channelwise convs – rotation invariance achieved by pooling) –
        layers = []
        for _ in range(decoder_conv_layers):
            layers += [nn.Conv2d(self.total_hidden_channels,
                                 self.total_hidden_channels,
                                 kernel_size=3, padding=1, bias=False),
                        nn.ReLU()]
        layers += [nn.Conv2d(self.total_hidden_channels,
                             self.output_channels,
                             kernel_size=3, padding=1, bias=False)]
        self.decoder_conv = nn.Sequential(*layers)

    # -------------------------------------------------------------------------
    def forward(self, input_seq, pred_len, teacher_forcing_ratio=1.0,
                target_seq=None, return_hidden=False):
        """
        input_seq : (B, T_in, C, H, W)
        returns   : (B, pred_len, C_out, H, W)
        """
        B, T_in, C, H, W = input_seq.shape
        device = input_seq.device

        h = torch.zeros(B, self.num_v, self.total_hidden_channels,
                        H, W, device=device)

        # encoder -------------------------------------------------------------
        for t in range(T_in):
            h = self.cell(input_seq[:, t], h)

        prev = input_seq[:, -1]
        outputs = []
        hiddens = []

        # decoder -------------------------------------------------------------
        for t in range(pred_len):
            frame = (target_seq[:, t] if self.training and
                     target_seq is not None and
                     random.random() < teacher_forcing_ratio else prev.detach())
            h = self.cell(frame, h)

            if return_hidden:
                hiddens.append(h.clone())

            # pool over w (velocity) dimension – yields rotation-invariant code
            if   self.pool_type == 'mean': feat = h.mean(1)
            elif self.pool_type == 'sum':  feat = h.sum(1)
            else:                          feat = h.max(1)[0]

            out = self.decoder_conv(feat)
            outputs.append(out)
            prev = out

        if return_hidden:
            return torch.stack(outputs, dim=1), hiddens
        else:
            return torch.stack(outputs, dim=1)