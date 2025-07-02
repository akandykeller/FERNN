import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class MovingMNISTDataset(Dataset):
    """
    PyTorch Dataset for generating sequences of moving MNIST digits with wrap-around boundary conditions.

    Args:
        root (str): path to download/load the MNIST data.
        train (bool): if True, load training split; otherwise, test split.
        seq_len (int): number of frames in each sequence.
        image_size (int): height and width of the square output frames.
        velocity_range_x (tuple of int): (min_vx, max_vx) inclusive range for horizontal velocity.
        velocity_range_y (tuple of int): (min_vy, max_vy) inclusive range for vertical velocity.
        num_digits (int): number of MNIST digits to overlay in each sequence.
        transform (callable, optional): transform applied to the full sequence tensor.
        download (bool): whether to download MNIST if not present.
        random (bool): if True, use random selection; if False, use deterministic random selection.
        seed (int): random seed for deterministic selection when random=False.
    """
    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=28,
        velocity_range_x=(-3, 3),
        velocity_range_y=(-3, 3),
        num_digits=2,
        transform=None,
        download=True,
        random=True,
        seed=42
    ):
        super().__init__()
        self.mnist = MNIST(root=root, train=train, download=download)
        self.seq_len = seq_len
        self.image_size = image_size
        self.vx_min, self.vx_max = velocity_range_x
        self.vy_min, self.vy_max = velocity_range_y
        self.num_digits = num_digits
        self.transform = transform
        self.to_tensor = ToTensor()
        self.random = random
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        # Sample digits and their labels
        imgs = []
        labels = []
        for n in range(self.num_digits):
            if self.random:
                idx = np.random.randint(0, len(self.mnist))
            else:
                # Use deterministic random selection
                idx = self.rng.randint(0, len(self.mnist))
            img_pil, lbl = self.mnist[idx]
            img = self.to_tensor(img_pil)  # shape: (1, 28, 28)
            imgs.append(img)
            labels.append(lbl)

        # Sample integer velocities and initial positions for each digit
        velocities_x = []
        velocities_y = []
        positions_x = []
        positions_y = []
        
        # Prepare output sequence
        seq = torch.zeros(
            self.seq_len, 1, self.image_size, self.image_size, dtype=imgs[0].dtype
        )

        for _ in range(self.num_digits):
            # Sample integer velocities for this digit
            if self.random:
                vx = np.random.randint(self.vx_min, self.vx_max + 1)
                vy = np.random.randint(self.vy_min, self.vy_max + 1)
            else:
                vx = self.rng.randint(self.vx_min, self.vx_max + 1)
                vy = self.rng.randint(self.vy_min, self.vy_max + 1)
            velocities_x.append(vx)
            velocities_y.append(vy)
            
            # Sample integer initial positions (top-left corner of digit)
            if self.random:
                x0 = np.random.randint(0, self.image_size)
                y0 = np.random.randint(0, self.image_size)
            else:
                x0 = self.rng.randint(0, self.image_size)
                y0 = self.rng.randint(0, self.image_size)
            positions_x.append(x0)
            positions_y.append(y0)

        # Generate each frame by rolling each digit and summing
        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)
            
            for i, img in enumerate(imgs):
                # Pad the MNIST image (28x28) to match the frame size if needed (image_size x image_size)
                pad_size = (self.image_size - img.shape[1]) // 2
                padded_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
                
                # If padding is uneven, add extra padding to the right/bottom
                if padded_img.shape[1] < self.image_size:
                    padded_img = F.pad(padded_img, (0, self.image_size - padded_img.shape[2], 0, self.image_size - padded_img.shape[1]), "constant", 0)
                
                # Replace the original image with the padded version
                img = padded_img
                
                # Calculate current position based on this digit's velocity and initial position
                shift_x = positions_x[i] + velocities_x[i] * t
                shift_y = positions_y[i] + velocities_y[i] * t
                
                # Wrap-around translation via torch.roll
                moved = torch.roll(
                    img,
                    shifts=(shift_y, shift_x),  # (vertical, horizontal)
                    dims=(1, 2),
                )
                frame += moved

            # Clamp to [0, 1] in case of overlap
            seq[t] = frame.clamp(0.0, 1.0)

        # Apply optional transform to full sequence
        if self.transform:
            seq = self.transform(seq)

        # Return sequence and labels (list or int)
        if self.num_digits == 1:
            return seq, labels[0]
        else:
            return seq, torch.tensor(labels, dtype=torch.long)


class FixedVelocityMovingMNIST(MovingMNISTDataset):
    """Same as MovingMNISTDataset, but every digit moves at a fixed (vx, vy)."""
    def __init__(self, vx, vy, **kwargs):
        super().__init__(velocity_range_x=(vx, vx),
                         velocity_range_y=(vy, vy),
                         **kwargs)


# Example usage:
train_dataset = MovingMNISTDataset(
    root='./data',
    train=True,
    seq_len=30,
    image_size=32,
    velocity_range_x=(-2, 2),
    velocity_range_y=(-2, 2),
    num_digits=2
)

test_dataset = MovingMNISTDataset(
    root='./data',
    train=True,
    seq_len=30,
    image_size=64,
    velocity_range_x=(-2, 2),
    velocity_range_y=(-2, 2),
    num_digits=2
)