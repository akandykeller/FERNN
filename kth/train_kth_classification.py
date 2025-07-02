#!/usr/bin/env python
"""train_kth.py — Action recognition on the **KTH Action** dataset
-------------------------------------------------------------------
Run example
-----------
python kth/train_kth_classification.py --epochs 500 --vx_range 2 --vy_range 2 --lr 3e-4 \
        --vx_data_range 2 --vy_data_range 2 --batch_size 32 --height 32 --width 32 --wandb \
        --model galilean --run_name FERNN-V2_V2-train
"""

import argparse, random, re, urllib.request, zipfile
from pathlib import Path
from typing import Dict, List, Tuple
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import h5py
import matplotlib.pyplot as plt
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import wandb

_BASE = "http://www.csc.kth.se/cvap/actions/"
_ACTIONS = [
    ("walking",      242),
    ("jogging",      168),
    ("running",      149),
    ("boxing",       194),
    ("handwaving",   218),
    ("handclapping", 176),
]
_ACTION2IDX = {name: idx for idx, (name, _) in enumerate(_ACTIONS)}
_SEQ_TXT = "00sequences.txt"


# ===========================================================================
# Utils
# ===========================================================================

def _human(mib: int) -> str:
    return f"{mib}\u00a0MB" if mib < 1024 else f"{mib/1024:.1f}\u00a0GB"


def _safe_dl(url: str, dst: Path) -> Path:
    """Download *url* to *dst* unless it already exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    with urllib.request.urlopen(url) as resp, open(dst, "wb") as fh:
        total = int(resp.getheader("Content-Length", "0"))
        pbar = tqdm(total=total, unit="B", unit_scale=True,
                    desc=f"⬇ {dst.name}")
        while buf := resp.read(16 << 10):
            fh.write(buf); pbar.update(len(buf))
        pbar.close()
    return dst


def _download_kth(root: Path) -> Path:
    """Ensure the KTH archives + annotation file are present; return video dir."""
    root = root.expanduser()
    video_dir = root / "kth_actions"
    video_dir.mkdir(parents=True, exist_ok=True)

    # annotation list
    _safe_dl(_BASE + _SEQ_TXT, root / _SEQ_TXT)

    # each action archive
    for act, mib in _ACTIONS:
        zip_path = root / f"{act}.zip"
        if not zip_path.exists():
            print(f"Downloading {act} ({_human(mib)}) …")
            _safe_dl(_BASE + f"{act}.zip", zip_path)
        marker = video_dir / f".{act}_unzipped"
        if not marker.exists():
            print(f"🗜️  Extracting {zip_path.name} …")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(video_dir)
            marker.touch()
    return video_dir


def preprocess_videos(video_dir: Path, hdf5_path: Path, height: int = 64, width: int = 64) -> None:
    """Pre-extract frames from all videos into a single HDF5 file."""
    if hdf5_path.exists():
        return
        
    def process_video(video_path: Path) -> Tuple[str, np.ndarray]:
        video_name = video_path.stem
        if video_name.endswith("_uncomp"):
            video_name = video_name[:-7]
            
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to grayscale and resize
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        cap.release()
        return video_name, np.stack(frames)
    
    # Process all videos in parallel
    video_paths = list(video_dir.glob("*.avi"))
    print(f"Pre-extracting frames from {len(video_paths)} videos...")
    
    # Create HDF5 file
    with h5py.File(hdf5_path, 'w') as f:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_video, vp) for vp in video_paths]
            for future in tqdm(futures, total=len(video_paths)):
                video_name, frames = future.result()
                f.create_dataset(video_name, data=frames, compression='gzip', compression_opts=1)


# ===========================================================================
# Dataset
# ===========================================================================

class KTHVideoClips(Dataset):
    """Return `(clip, label)` where *clip* has shape `(T,1,H,W)` and label∈[0,5]."""

    _LINE_RE = re.compile(r"^(?P<fname>\S+)\s+frames\s+(?P<ranges>[\d,\s\-]+)$")

    def __init__(self,
                 root: Path,
                 seq_len: int = 16,
                 height: int = 64,
                 width: int | None = None,
                 step: int = 1,
                 split: str = 'train',  # 'train', 'val', or 'test'
                 vx_data_range: int = 0,
                 vy_data_range: int = 0,
                 split_method: str = 'person'):  # 'person' or 'random'
        self.video_root = _download_kth(root)
        self.hdf5_path = root / "kth_frames.h5"
        self.seq_len, self.step = seq_len, step
        self.h, self.w = height, width or height
        self.split = split
        self.vx_data_range = vx_data_range
        self.vy_data_range = vy_data_range
        self.split_method = split_method
        
        # Pre-extract frames if needed
        if not self.hdf5_path.exists():
            print("Pre-extracting frames for faster loading...")
            preprocess_videos(self.video_root, self.hdf5_path, height, width)
        
        # Load entire dataset into memory
        print("Loading dataset into memory...")
        self.frames_dict = {}
        with h5py.File(self.hdf5_path, 'r') as f:
            for video_name in tqdm(f.keys(), desc="Loading videos"):
                self.frames_dict[video_name] = f[video_name][:]  # Load entire video into memory
        
        # Parse annotation file and split sequences
        self.sequences: List[Tuple[str,int,int]] = []
        self._parse_sequences(root / _SEQ_TXT)
        
        # Generate deterministic velocities for each sequence
        if vx_data_range > 0 or vy_data_range > 0:
            print("Generating deterministic velocities for data augmentation...")
            # Use a fixed seed for reproducibility
            rng = np.random.RandomState(42)
            self.velocities = []
            for _ in range(len(self.sequences)):
                vx = rng.randint(-vx_data_range, vx_data_range + 1)
                vy = rng.randint(-vy_data_range, vy_data_range + 1)
                self.velocities.append((vy, vx))  # Note: (vy, vx) to match kernel dimensions
            print(f"Generated {len(self.velocities)} velocity vectors")
        else:
            self.velocities = None
            
        print(f"Dataset loaded into memory. Total size: {sum(frames.nbytes for frames in self.frames_dict.values()) / (1024**3):.2f} GB")

    def _parse_sequences(self, txt_path: Path):
        # First collect all sequences
        all_sequences = []
        with open(txt_path) as fh:
            for line in fh:
                m = self._LINE_RE.match(line.strip())
                if not m: continue
                vname = m.group("fname")
                rng_blob = m.group("ranges")
                vpath = self.video_root / f"{vname}_uncomp.avi"
                if not vpath.exists(): continue
                
                # Get frame count from memory
                if vname not in self.frames_dict: continue
                total_frames = len(self.frames_dict[vname])
                
                for rng in rng_blob.split(','):
                    rng = rng.strip()
                    if not rng: continue
                    if '-' in rng:
                        s, e = map(int, rng.split('-'))
                    else:
                        s = e = int(rng)
                    e = min(e, total_frames - 1)
                    if e - s + 1 >= self.seq_len * self.step:
                        all_sequences.append((vname, s, e))

        if self.split_method == 'person':
            # Split sequences by person ID
            train_sequences = []
            val_sequences = []
            test_sequences = []
            
            for vname, s, e in all_sequences:
                # Extract person ID from video name (e.g., "person01_boxing_d1" -> "01")
                person_id = int(vname.split('_')[0][6:8])
                
                if person_id <= 16:
                    train_sequences.append((vname, s, e))
                elif person_id <= 20:
                    val_sequences.append((vname, s, e))
                else:  # person_id 21-25
                    test_sequences.append((vname, s, e))
        else:  # random split
            # Use a fixed random seed for reproducibility
            rng = np.random.RandomState(42)
            # Shuffle sequences
            rng.shuffle(all_sequences)
            # Split into train (70%), val (15%), test (15%)
            n = len(all_sequences)
            train_size = int(0.7 * n)
            val_size = int(0.15 * n)
            
            train_sequences = all_sequences[:train_size]
            val_sequences = all_sequences[train_size:train_size + val_size]
            test_sequences = all_sequences[train_size + val_size:]

        # Assign sequences based on split
        if self.split == 'train':
            self.sequences = train_sequences
        elif self.split == 'val':
            self.sequences = val_sequences
        else:  # test
            self.sequences = test_sequences

        print(f"Loaded {len(self.sequences):,} valid subsequences for {self.split} split using {self.split_method} method.")

    def __len__(self):
        return len(self.sequences)

    def _apply_velocity_shift(self, frame, vy, vx):
        """Apply velocity shift to a single frame using circular padding,
        but reduce shift & pad modulo the frame dimensions so you never
        wrap more than once."""
        _, H, W = frame.shape

        # 1) reduce the shift itself modulo the spatial dims
        #    (so a shift of -1 becomes H-1, equivalent to one left wrap)
        vy = vy % H
        vx = vx % W

        if vy == 0 and vx == 0:
            return frame

        # 2) build a single‐pixel “shift” kernel
        max_shift = max(vy, vx)
        kernel_size = max_shift * 2 + 1
        shift_kernel = torch.zeros(1, 1, kernel_size, kernel_size,
                                    device=frame.device, dtype=frame.dtype)
        center = kernel_size // 2
        shift_kernel[0, 0, center + vy, center + vx] = 1.0

        # 3) effective pad is ≤ H and ≤ W
        pad_y = max_shift % H or H    # if max_shift % H == 0, we want a full H wrap
        pad_x = max_shift % W or W

        # F.pad takes (pad_left, pad_right, pad_top, pad_bottom)
        padded = F.pad(frame,
                        (pad_x, pad_x, pad_y, pad_y),
                        mode='circular')

        # 4) convolve the one‐pixel kernel over each channel
        shifted = F.conv2d(padded,
                            shift_kernel,
                            padding=0)
        return shifted

    def __getitem__(self, idx):
        vname, s, e = self.sequences[idx]
        max_start = e - self.seq_len * self.step + 1
        start = random.randint(s, max_start)
        
        # Load frames from memory
        frames = []
        for i in range(self.seq_len):
            frame_idx = start + i * self.step
            frame = self.frames_dict[vname][frame_idx]
            frame = torch.from_numpy(frame).float().unsqueeze(0) / 255.0
            frames.append(frame)
        
        clip = torch.stack(frames, dim=0)  # (T,1,H,W)

        # Apply velocity shift if enabled
        if self.velocities is not None:
            vy, vx = self.velocities[idx]
            shifted_frames = []
            for t, frame in enumerate(clip):
                # Apply accumulated shift: velocity * time
                shift_y = vy * t
                shift_x = vx * t
                shifted = self._apply_velocity_shift(frame, shift_y, shift_x)
                shifted_frames.append(shifted)
            clip = torch.stack(shifted_frames, dim=0)

        # augmentations
        if self.split == 'train':
            if random.random() < 0.5:
                clip = torch.flip(clip, dims=[3])      # horizontal flip

        # label from video name
        action = vname.split('_')[1]
        label = _ACTION2IDX[action]
        return clip, label


# ──────────────────────────────────────────────────────────────
#  3-D CNN baseline
# ──────────────────────────────────────────────────────────────
class Video3DCNNClassifier(nn.Module):
    def __init__(
        self,
        input_ch: int     = 1,     # grayscale frames
        base: int         = 32,    # starting #channels; doubles each stage
        num_classes: int  = 6,
        T_stride: int     = 2,     # time stride for the *first* conv layer
    ):
        super().__init__()

        def block(in_ch, out_ch, stride_t=1, stride_xy=1):
            return nn.Sequential(
                nn.Conv3d(
                    in_ch, out_ch, kernel_size=(3, 3, 3),
                    stride=(stride_t, stride_xy, stride_xy),
                    padding=1, bias=False
                ),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        # stage-1:  T/2, H,W intact  (B, base, T/2, 64, 64)
        self.st1 = block(input_ch, base, stride_t=T_stride)

        # stage-2:  T/2, 32×32
        self.st2 = nn.Sequential(
            block(base, base * 2, stride_xy=2),
            block(base * 2, base * 2)
        )

        # stage-3:  T/2, 16×16
        self.st3 = nn.Sequential(
            block(base * 2, base * 4, stride_xy=2),
            block(base * 4, base * 4)
        )

        # global space-time pool  → 1×1×1
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = nn.Linear(base * 4, num_classes)

        # handy parameter counter
        self._n_params = sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ #
    def forward(self, clip):             # clip: (B, T, C, H, W)
        clip = clip.permute(0, 2, 1, 3, 4) # (B, C, T, H, W)
        x = self.st1(clip)
        x = self.st2(x)
        x = self.st3(x)                  # (B, C3, T/2, 16, 16)
        x = self.global_pool(x)          # (B, C3, 1,1,1)
        x = x.flatten(1)                 # (B, C3)
        return self.head(x)              # logits

    # ------------------------------------------------------------------ #
    def count_parameters(self, verbose=True):
        if verbose:
            print(f"Total trainable parameters: {self._n_params/1e6:.2f} M")
        return self._n_params


# ──────────────────────────────────────────────────────────────
#  FERNN
# ──────────────────────────────────────────────────────────────
class FERNN(nn.Module):
    """
    Flow Equivariant Recurrent Neural Network

    v_list: list of (vy, vx) pairs for which the network is equivariant. 
            Setting to [(0,0)] means the network is not equivariant (G-RNN).
    """
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1, v_list=[(0,0)]):
        super(FERNN, self).__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.v_list = v_list
        self.num_v = len(v_list)
        self.num_layers = num_layers

        # Create shift kernels for each velocity
        self.register_buffer('shift_kernels', self._create_shift_kernels(kernel_size))

        # First layer processes input channels
        self.conv_u_first = nn.Sequential(
            nn.Conv2d(input_channels, 32, 5, 1, 2, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True), 
            nn.Conv2d(64, hidden_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True), 
        )

        # Subsequent layers process hidden_channels * num_v channels
        self.conv_u_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(num_layers - 1)
        ])

        # Hidden state processing layers
        self.conv_h_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding=padding, bias=False, padding_mode='circular')
            ) for _ in range(num_layers)
        ])

        self.activation = nn.Tanh()

    def _create_shift_kernels(self, kernel_size):
        # Create a kernel for each velocity that will perform the shift operation
        max_shift = max(max(abs(vx) for _, vx in self.v_list), max(abs(vy) for vy, _ in self.v_list))
        kernel_size = max_shift * 2 + 1
        shift_kernels = torch.zeros(self.num_v, 1, kernel_size, kernel_size)
        
        for i, (vy, vx) in enumerate(self.v_list):
            # Place 1 at the position that will create the desired shift
            shift_kernels[i, 0, kernel_size//2 + vy, kernel_size//2 + vx] = 1.0
            
        return shift_kernels

    def _process_layer(self, h, u_t, layer_idx):
        """Process a single layer of the RNN."""
        batch_size = h.size(0)
        height, width = h.size(-2), h.size(-1)
        
        # Reshape h to separate velocity groups
        h_reshaped = h.view(batch_size, self.num_v, self.hidden_channels, height, width)
        
        # Reshape for group convolution
        h_reshaped = h_reshaped.transpose(1, 2)
        h_reshaped = h_reshaped.reshape(-1, self.num_v, height, width)
        
        # Add circular padding
        pad_size = self.shift_kernels.size(-1)//2
        h_padded = F.pad(h_reshaped, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        
        # Apply velocity shifts using convolution
        h_shifted = F.conv2d(
            h_padded,
            self.shift_kernels,
            padding=0,
            groups=self.num_v
        )
        
        # Reshape back to original dimensions
        h_shifted = h_shifted.view(batch_size, self.hidden_channels, self.num_v, height, width)
        h_shifted = h_shifted.transpose(1, 2)
        h_shifted = h_shifted.reshape(batch_size * self.num_v, self.hidden_channels, height, width)
        
        # Process input for this layer
        if layer_idx == 0:
            u_processed = self.conv_u_first(u_t).view(batch_size, 1, self.hidden_channels, height, width).repeat(1, self.num_v, 1, 1, 1) 
        else:
            # For subsequent layers, process the previous layer's output
            u_t_reshaped = u_t.view(batch_size * self.num_v, self.hidden_channels, height, width)
            u_processed = self.conv_u_layers[layer_idx-1](u_t_reshaped)
            u_processed = u_processed.view(batch_size, self.num_v, self.hidden_channels, height, width)
        
        # Update hidden state
        h_new = self.activation(
            u_processed
            + self.conv_h_layers[layer_idx](h_shifted).view(batch_size, self.num_v, self.hidden_channels, height, width)
        )
        
        return h_new

    def forward(self, u):
        batch_size, time_steps, channels, in_height, in_width = u.size()
        height, width = in_height, in_width
        
        # Initialize hidden states for all layers
        h_layers = [torch.zeros(batch_size, self.num_v, self.hidden_channels, height, width, device=u.device) 
                   for _ in range(self.num_layers)]
        
        outputs = []

        for t in range(time_steps):
            u_t = u[:, t]
            
            # Process each layer
            for layer_idx in range(self.num_layers):
                h_layers[layer_idx] = self._process_layer(h_layers[layer_idx], 
                                                        u_t if layer_idx == 0 else h_layers[layer_idx-1],
                                                        layer_idx)
            
            # Use output from last layer
            out = h_layers[-1].view(batch_size, self.num_v, self.hidden_channels, height, width)
            out = out.permute(1, 0, 2, 3, 4)
            outputs.append(out.unsqueeze(2))

        outputs = torch.cat(outputs, dim=2)
        return outputs


class GRNN_Plus(FERNN):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers=1, v_list=[(0,0)]):
        super(GRNN_Plus, self).__init__(input_channels, hidden_channels, kernel_size, num_layers, v_list)
        # Replace shift kernels with learned kernels
        max_shift = max(max(abs(vx) for _, vx in self.v_list), max(abs(vy) for vy, _ in self.v_list))
        kernel_size = max_shift * 2 + 1
        
        # Initialize learned kernels with Kaiming initialization (same as nn.Conv2d default)
        self.learned_kernels = nn.Parameter(torch.zeros(self.num_v, 1, kernel_size, kernel_size))
        # Initialize each kernel separately since they are used in group convolution
        for i in range(self.num_v):
            nn.init.kaiming_normal_(self.learned_kernels[i:i+1], mode='fan_out', nonlinearity='linear')
        
        # Remove the shift_kernels buffer since we're using learned_kernels
        delattr(self, 'shift_kernels')

    def _process_layer(self, h, u_t, layer_idx):
        """Process a single layer of the RNN."""
        batch_size = h.size(0)
        height, width = h.size(-2), h.size(-1)
        
        # Reshape h to separate velocity groups
        h_reshaped = h.view(batch_size, self.num_v, self.hidden_channels, height, width)
        
        # Reshape for group convolution
        h_reshaped = h_reshaped.transpose(1, 2)
        h_reshaped = h_reshaped.reshape(-1, self.num_v, height, width)
        
        # Add circular padding
        pad_size = self.learned_kernels.size(-1)//2
        h_padded = F.pad(h_reshaped, (pad_size, pad_size, pad_size, pad_size), mode='circular')
        
        # Apply learned kernels using convolution
        h_shifted = F.conv2d(
            h_padded,
            self.learned_kernels,
            padding=0,
            groups=self.num_v
        )
        
        # Reshape back to original dimensions
        h_shifted = h_shifted.view(batch_size, self.hidden_channels, self.num_v, height, width)
        h_shifted = h_shifted.transpose(1, 2)
        h_shifted = h_shifted.reshape(batch_size * self.num_v, self.hidden_channels, height, width)
        
        # Process input for this layer
        if layer_idx == 0:
            u_processed = self.conv_u_first(u_t).view(batch_size, 1, self.hidden_channels, height, width).repeat(1, self.num_v, 1, 1, 1) 
        else:
            # For subsequent layers, process the previous layer's output
            u_t_reshaped = u_t.view(batch_size * self.num_v, self.hidden_channels, height, width)
            u_processed = self.conv_u_layers[layer_idx-1](u_t_reshaped)
            u_processed = u_processed.view(batch_size, self.num_v, self.hidden_channels, height, width)
        
        # Update hidden state
        h_new = self.activation(
            u_processed
            + self.conv_h_layers[layer_idx](h_shifted).view(batch_size, self.num_v, self.hidden_channels, height, width)
        )
        
        return h_new


class FERNN_Classifier(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_channels, num_patterns, v_list=[(0,0)], num_layers=1, use_fake_rnn=False):
        super(FERNN_Classifier, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_channels = hidden_channels
        rnn_class = GRNN_Plus if use_fake_rnn else FERNN
        self.lstm = rnn_class(input_channels=1, hidden_channels=self.hidden_channels, kernel_size=3, v_list=v_list, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.hidden_channels, num_patterns)
       # convenient parameter counter
        self._n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        rnn_out = self.lstm(x)
        out = torch.clone(rnn_out)
        out = out.max(0, keepdim=False)[0]
        out = out[:, -1]
        out = self.pool(out)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out

    def count_parameters(self, verbose=True):
        if verbose:
            print(f"Total trainable parameters: {self._n_params/1e6:.2f} M")
        return self._n_params


# ===========================================================================
# Training / Eval
# ===========================================================================

def _train_epoch(model, loader, opt, criterion, device):
    model.train()
    loss_sum, correct = 0.0, 0
    for clips, labels in tqdm(loader, desc="Training", total=len(loader)):
        clips, labels = clips.to(device), labels.to(device)
        opt.zero_grad()
        logits = model(clips)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_sum += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


@torch.no_grad()
def _eval(model, loader, criterion, device):
    model.eval()
    loss_sum, correct = 0.0, 0
    for clips, labels in tqdm(loader, desc="Validation", total=len(loader)):
        clips, labels = clips.to(device), labels.to(device)
        logits = model(clips)
        loss = criterion(logits, labels)
        loss_sum += loss.item() * clips.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
    return loss_sum / len(loader.dataset), correct / len(loader.dataset)


def log_example_sequences(train_dataset, test_dataset, num_examples=3, frames_per_sequence=4):
    """Log example sequences to wandb for visualization."""
    # Create a figure for training examples
    train_fig = plt.figure(figsize=(15, 5*num_examples))
    for i in range(num_examples):
        # Get a random training example
        idx = random.randint(0, len(train_dataset)-1)
        clip, label = train_dataset[idx]
        
        # Select evenly spaced frames
        frame_indices = np.linspace(0, len(clip)-1, frames_per_sequence, dtype=int)
        
        # Plot frames
        for j, frame_idx in enumerate(frame_indices):
            plt.subplot(num_examples, frames_per_sequence, i*frames_per_sequence + j + 1)
            plt.imshow(clip[frame_idx].squeeze().cpu().numpy(), cmap='gray')
            if j == 0:  # Only show label on first frame
                plt.title(f'Train: {list(_ACTION2IDX.keys())[label]}')
            plt.axis('off')
    
    plt.tight_layout()
    wandb.log({"train_examples": wandb.Image(train_fig)})
    plt.close()
    
    # Create a figure for test examples
    test_fig = plt.figure(figsize=(15, 5*num_examples))
    for i in range(num_examples):
        # Get a random test example
        idx = random.randint(0, len(test_dataset)-1)
        clip, label = test_dataset[idx]
        
        # Select evenly spaced frames
        frame_indices = np.linspace(0, len(clip)-1, frames_per_sequence, dtype=int)
        
        # Plot frames
        for j, frame_idx in enumerate(frame_indices):
            plt.subplot(num_examples, frames_per_sequence, i*frames_per_sequence + j + 1)
            plt.imshow(clip[frame_idx].squeeze().cpu().numpy(), cmap='gray')
            if j == 0:  # Only show label on first frame
                plt.title(f'Test: {list(_ACTION2IDX.keys())[label]}')
            plt.axis('off')
    
    plt.tight_layout()
    wandb.log({"test_examples": wandb.Image(test_fig)})
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="./data/kth/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--vx_range", type=int, default=0, help="Range of x-velocities to make netowrk equivariant w.r.t. (-vx_range to vx_range)")
    parser.add_argument("--vy_range", type=int, default=0, help="Range of y-velocities to make netowrk equivariant w.r.t. (-vy_range to vy_range)")
    parser.add_argument("--vx_data_range", type=int, default=0, help="Range of x-velocities in dataset (-vx_data_range to vx_data_range)")
    parser.add_argument("--vy_data_range", type=int, default=0, help="Range of y-velocities in dataset (-vy_data_range to vy_data_range)")
    parser.add_argument("--height", type=int, default=64, help="Height of the video frames")
    parser.add_argument("--width", type=int, default=64, help="Width of the video frames")
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--step", type=int, default=1, help="Step size")
    parser.add_argument("--split_method", type=str, default="person", choices=["person", "random"], 
                       help="Method to split the dataset: 'person' (split by person ID) or 'random' (random split by clip)")
    parser.add_argument("--use_fake_rnn", action="store_true", help="Use learned kernels instead of shift kernels (G-RNN+ Architecture)")
    parser.add_argument("--channel_multiplier", type=int, default=1, help="Channel multiplier for increasing the number of parameters for small models")
    parser.add_argument("--num_rnn_layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="kth-action-recognition", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="ENTITY", help="Wandb entity name")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--wandb_dir", type=str, default="./tmp/", help="Directory for wandb files")
    parser.add_argument("--model", type=str, default="fernn", choices=["fernn", "grnn", "grnn_plus", "3dcnn"], help="Model to use")
    parser.add_argument("--v_gen_test", action="store_true", help="Run velocity generalization test sets")
    parser.add_argument("--vx_gen_test", action="store_true", help="Run velocity generalization test sets for x translation")


    args = parser.parse_args()

    # Generate v_list based on vx_range and vy_range
    v_list = [(j, i) for i in range(-args.vx_range, args.vx_range + 1) 
                     for j in range(-args.vy_range, args.vy_range + 1)]
    if not v_list:  # If both ranges are 0, use default [(0,0)]
        v_list = [(0,0)]
    print(f"Using velocity list: {v_list}")
    print(f"Total number of velocities: {len(v_list)}")
    print(f"Using channel multiplier: {args.channel_multiplier}")
    print("Note: Each velocity is (vy,vx) where:")
    print("  - vy is velocity in height dimension")
    print("  - vx is velocity in width dimension")

    # ───────────────────────── Data
    print(f"Loading KTH dataset from {args.data_root} …")
    
    # Create datasets for each split
    train_dataset = KTHVideoClips(
        Path(args.data_root), 
        seq_len=16, 
        height=args.height, 
        width=args.width, 
        step=2, 
        split='train',
        vx_data_range=args.vx_data_range,
        vy_data_range=args.vy_data_range,
        split_method=args.split_method
    )
    
    val_dataset = KTHVideoClips(
        Path(args.data_root), 
        seq_len=16, 
        height=args.height, 
        width=args.width, 
        step=2, 
        split='val',
        vx_data_range=args.vx_data_range,
        vy_data_range=args.vy_data_range,
        split_method=args.split_method
    )
    
    test_dataset = KTHVideoClips(
        Path(args.data_root), 
        seq_len=16, 
        height=args.height, 
        width=args.width, 
        step=2, 
        split='test',
        vx_data_range=args.vx_data_range,
        vy_data_range=args.vy_data_range,
        split_method=args.split_method
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=True
    )

    if args.v_gen_test:
        v_gen_0_test_dataset = KTHVideoClips(
            Path(args.data_root), 
            seq_len=16, 
            height=args.height, 
            width=args.width, 
            step=2, 
            split='test',
            vx_data_range=0,
            vy_data_range=0,
            split_method=args.split_method
        )

        v_gen_1_test_dataset = KTHVideoClips(
            Path(args.data_root), 
            seq_len=16, 
            height=args.height, 
            width=args.width, 
            step=2, 
            split='test',
            vx_data_range=1,
            vy_data_range=1,
            split_method=args.split_method
        )

        v_gen_2_test_dataset = KTHVideoClips(
            Path(args.data_root), 
            seq_len=16, 
            height=args.height, 
            width=args.width, 
            step=2, 
            split='test',
            vx_data_range=2,
            vy_data_range=2,
            split_method=args.split_method
        )

        v_gen_0_test_loader = DataLoader(
            v_gen_0_test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )

        v_gen_1_test_loader = DataLoader(
            v_gen_1_test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=True
        )

        v_gen_2_test_loader = DataLoader(
            v_gen_2_test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=args.prefetch_factor,   
            persistent_workers=True
        )


    # ───────────────────────── Model
    print("Building model …")
    assert args.height == args.width, "Height and width must be the same"

    if args.model == "fernn":
        model = FERNN_Classifier(
            input_size=args.height, 
            hidden_size=args.height*args.width, 
            hidden_channels=128*args.channel_multiplier, 
            num_patterns=6, 
            v_list=v_list, 
            num_layers=args.num_rnn_layers,
            use_fake_rnn=args.use_fake_rnn
        )
    elif args.model == "grnn":
        assert args.vx_range == 0 and args.vy_range == 0, "vx_range and vy_range must be 0 for grnn"
        model = FERNN_Classifier(
            input_size=args.height, 
            hidden_size=args.height*args.width, 
            hidden_channels=128*args.channel_multiplier, 
            num_patterns=6, 
            v_list=[(0,0)], 
            num_layers=args.num_rnn_layers,
            use_fake_rnn=args.use_fake_rnn
        )
    elif args.model == "grnn_plus":
        model = FERNN_Classifier(
            input_size=args.height, 
            hidden_size=args.height*args.width, 
            hidden_channels=128*args.channel_multiplier, 
            num_patterns=6, 
            v_list=v_list, 
            num_layers=args.num_rnn_layers,
            use_fake_rnn=True
        )
    elif args.model == "3dcnn":
        model = Video3DCNNClassifier(
            input_ch=1,
            base=16*args.channel_multiplier,
            num_classes=6,
            T_stride=2
        )
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    model = model.to(args.device)
    n_params = model.count_parameters(verbose=True)

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            dir=args.wandb_dir,
            config=vars(args)
        )
        # Log model architecture
        wandb.watch(model, log="all")
        
        # Log example sequences
        print("Logging example sequences to wandb...")
        log_example_sequences(train_dataset, test_dataset, frames_per_sequence=16)

        wandb.log({"n_params": n_params})

    # ───────────────────────── Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ───────────────────────── Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = _eval(model, val_loader, criterion, args.device)
        test_loss, test_acc = _eval(model, test_loader, criterion, args.device)

        if args.v_gen_test:
            v_gen_0_test_loss, v_gen_0_test_acc = _eval(model, v_gen_0_test_loader, criterion, args.device)
            v_gen_1_test_loss, v_gen_1_test_acc = _eval(model, v_gen_1_test_loader, criterion, args.device)
            v_gen_2_test_loss, v_gen_2_test_acc = _eval(model, v_gen_2_test_loader, criterion, args.device)

        # ───────────────────────── Logging
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} - "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        if args.v_gen_test:
            print(f"V-Gen 0 Test Loss: {v_gen_0_test_loss:.4f}, V-Gen 0 Test Acc: {v_gen_0_test_acc:.4f} - "
                  f"V-Gen 1 Test Loss: {v_gen_1_test_loss:.4f}, V-Gen 1 Test Acc: {v_gen_1_test_acc:.4f} - "
                  f"V-Gen 2 Test Loss: {v_gen_2_test_loss:.4f}, V-Gen 2 Test Acc: {v_gen_2_test_acc:.4f}")

        # Log metrics to wandb
        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            if args.v_gen_test:
                wandb.log({
                    "v_gen_0_test_loss": v_gen_0_test_loss,
                    "v_gen_0_test_acc": v_gen_0_test_acc,
                    "v_gen_1_test_loss": v_gen_1_test_loss,
                    "v_gen_1_test_acc": v_gen_1_test_acc,
                    "v_gen_2_test_loss": v_gen_2_test_loss,
                    "v_gen_2_test_acc": v_gen_2_test_acc
                })

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), Path(args.data_root) / "best_model.pth")
            print(f"Saved new best model with validation accuracy: {best_val_acc:.4f}.")
            print(f"Corresponding test accuracy: {best_test_acc:.4f}.")
            
            if args.v_gen_test:
                best_v_gen_0_test_acc = v_gen_0_test_acc
                best_v_gen_1_test_acc = v_gen_1_test_acc
                best_v_gen_2_test_acc = v_gen_2_test_acc

            # Log best model to wandb
            if args.wandb:
                wandb.save(str(Path(args.data_root) / "best_model.pth"))

    print(f"Best validation accuracy: {best_val_acc:.4f}.")
    print(f"Corresponding test accuracy: {best_test_acc:.4f}.")
    

    # Log final best metrics to wandb
    if args.wandb:
        wandb.log({
            "best_val_acc": best_val_acc,
            "best_test_acc": best_test_acc
        })

        if args.v_gen_test:
            wandb.log({
                "best_v_gen_0_test_acc": best_v_gen_0_test_acc,
                "best_v_gen_1_test_acc": best_v_gen_1_test_acc,
                "best_v_gen_2_test_acc": best_v_gen_2_test_acc
            })

        wandb.finish()

if __name__ == "__main__":
    main()



