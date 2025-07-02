import torch
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset


class RotatingMNISTDataset(Dataset):
    """
    PyTorch Dataset for generating sequences of rotating MNIST digits.

    Args:
        root (str): path to download/load the MNIST data.
        train (bool): if True, load training split; otherwise, test split.
        seq_len (int): number of frames in each sequence.
        image_size (int): height and width of the square output frames.
        angular_velocities (list of float): list of possible angular velocities in degrees/timestep to randomly select from.
        num_digits (int): number of MNIST digits to overlay in each sequence.
        transform (callable, optional): transform applied to the full sequence tensor.
        download (bool): whether to download MNIST if not present.
    """
    def __init__(
        self,
        root,
        train=True,
        seq_len=20,
        image_size=34,
        angular_velocities=[-10, -5, 0, 5, 10],
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
        self.angular_velocities = angular_velocities
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
        for _ in range(self.num_digits):
            if self.random:
                idx = np.random.randint(0, len(self.mnist))
            else:
                idx = self.rng.randint(0, len(self.mnist))

            img_pil, lbl = self.mnist[idx]
            img = self.to_tensor(img_pil)  # shape: (1, 28, 28)
            imgs.append(img)
            labels.append(lbl)

        # Sample angular velocities and initial angles for each digit
        angular_velocities = []
        initial_angles = []
        
        for _ in range(self.num_digits):
            # Sample angular velocity for this digit
            if self.random:
                omega = np.random.choice(self.angular_velocities)
            else:
                omega = self.rng.choice(self.angular_velocities)
            angular_velocities.append(omega)
            
            # Sample initial angle
            if self.random:
                initial_angle = np.random.uniform(0, 360)
            else:
                initial_angle = self.rng.uniform(0, 360)
            initial_angles.append(initial_angle)

        # Prepare output sequence
        seq = torch.zeros(
            self.seq_len, 1, self.image_size, self.image_size, dtype=imgs[0].dtype
        )

        # Calculate padding needed to keep rotated digits in frame
        # For a 28x28 digit, we need padding to accommodate diagonal rotation
        pad_size = int(np.ceil(28 * np.sqrt(2) / 2) - 14)  # Half of diagonal minus half of original size

        # Generate each frame by rotating each digit and summing
        for t in range(self.seq_len):
            frame = torch.zeros(1, self.image_size, self.image_size)
            
            for i, img in enumerate(imgs):
                # Pad the MNIST image to accommodate rotation
                padded_img = F.pad(img, (pad_size, pad_size, pad_size, pad_size), "constant", 0)
                
                # Calculate current angle based on angular velocity and initial angle
                current_angle = initial_angles[i] + angular_velocities[i] * t
                
                # Create rotation transform (2x3 matrix for 2D affine transformation)
                cos_theta = np.cos(np.radians(current_angle))
                sin_theta = np.sin(np.radians(current_angle))
                
                # Center of rotation
                center = padded_img.shape[1] / 2
                
                # Create the full 2x3 affine matrix
                rotation_matrix = torch.tensor([
                    [cos_theta, -sin_theta, 0],
                    [sin_theta, cos_theta, 0]
                ], dtype=torch.float32)
                
                # Apply rotation using grid sampling
                grid = F.affine_grid(
                    rotation_matrix.unsqueeze(0),
                    padded_img.unsqueeze(0).shape,
                    align_corners=True
                )
                rotated = F.grid_sample(
                    padded_img.unsqueeze(0),
                    grid,
                    align_corners=True,
                    mode='bilinear',
                    padding_mode='zeros'
                ).squeeze(0)
                
                # Center the rotated digit in the frame
                center_offset = (self.image_size - rotated.shape[1]) // 2
                if center_offset > 0:
                    rotated = F.pad(rotated, (center_offset, center_offset, center_offset, center_offset), "constant", 0)
                
                # Crop or pad to match desired image size
                if rotated.shape[1] > self.image_size:
                    start = (rotated.shape[1] - self.image_size) // 2
                    rotated = rotated[:, start:start+self.image_size, start:start+self.image_size]
                
                frame += rotated

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


# Example usage:
train_dataset = RotatingMNISTDataset(
    root='./data',
    train=True,
    seq_len=20,
    image_size=34,
    angular_velocities=[-20, -10, -5, 0, 5, 10, 20],
    num_digits=1
)

test_dataset = RotatingMNISTDataset(
    root='./data',
    train=False,
    seq_len=20,
    image_size=34,
    angular_velocities=[-20, -10, -5, 0, 5, 10, 20],
    num_digits=1
)
# loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

if __name__ == "__main__":
    import os
    from torchvision.utils import make_grid
    
    # Create output directory if it doesn't exist
    os.makedirs('examples', exist_ok=True)
    
    # Get a few example sequences
    num_examples = 4
    sequences = []
    labels = []
    
    for i in range(num_examples):
        seq, lbl = train_dataset[i]
        sequences.append(seq)
        labels.append(lbl)
    
    # Plot each sequence
    for i, (seq, lbl) in enumerate(zip(sequences, labels)):
        # Create a figure with subplots for each frame
        fig, axes = plt.subplots(1, seq.shape[0], figsize=(20, 2))
        if seq.shape[0] == 1:
            axes = [axes]
            
        # Plot each frame
        for t, ax in enumerate(axes):
            # Get the frame and convert to numpy
            frame = seq[t, 0].numpy()  # shape: (H, W)
            ax.imshow(frame, cmap='gray')
            ax.axis('off')
            if t == 0:
                ax.set_title(f'Sequence {i+1}, Label: {lbl}')
        
        # Save the figure
        plt.savefig(f'examples/sequence_{i+1}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
    
    print(f"Saved {num_examples} example sequences to the 'examples' directory")
