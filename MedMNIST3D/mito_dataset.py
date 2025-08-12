import os
import numpy as np
import torch
from torch.utils.data import Dataset

class MitoPatchDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        """
        Custom dataset for loading 3D mitochondrial patches.

        Args:
            csv_path (str): Path to the CSV file containing [z, y, x, label].
            data_dir (str): Directory where .npy patch files are stored.
            transform (callable, optional): Optional transform to apply to each sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        with open(csv_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                z, y, x, label = line.strip().split(',')
                filename = f"z{z}_y{y}_x{x}_label{label}.npy"
                self.samples.append((filename, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing file: {filepath}")

        voxel = np.load(filepath).astype(np.float32)

        # Remove extra dimensions like (64,64,32,1) or (1,64,64,32)
        while voxel.ndim > 3:
            voxel = np.squeeze(voxel)

        if voxel.ndim != 3:
            raise ValueError(f"Expected 3D volume after squeeze, got shape {voxel.shape}")

        if self.transform:
            voxel = self.transform(voxel)

        voxel = np.expand_dims(voxel, axis=0)  # Add channel dim: [1, D, H, W]
        image = torch.from_numpy(voxel)

        return image, torch.tensor(label, dtype=torch.long)























# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class MitoPatchDataset(Dataset):
#     def __init__(self, csv_path, data_dir, transform=None):
#         """
#         Custom dataset for loading 3D mitochondrial patches.
        
#         Args:
#             csv_path (str): Path to the CSV file containing [filename, label].
#             data_dir (str): Directory where .npy patch files are stored.
#             transform (callable, optional): Optional transform to apply to each sample.
#         """
#         self.data_dir = data_dir
#         self.transform = transform
        
#         # self.samples = []
#         # with open(csv_path, 'r') as f:
#         #     lines = f.readlines()[1:]  # Skip header
#         #     for line in lines:
#         #         filename, label = line.strip().split(',')
#         #         self.samples.append((filename, int(label)))


#         self.samples = []
#         with open(csv_path, 'r') as f:
#             lines = f.readlines()[1:]  # Skip header
#             for line in lines:
#                 z, y, x, label = line.strip().split(',')
#                 filename = f"{z}_{y}_{x}.npy"
#                 self.samples.append((filename, int(label)))





    
#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         filename, label = self.samples[idx]
#         filepath = os.path.join(self.data_dir, filename)


#         label = int(label)

#         voxel = np.load(filepath).astype(np.float32)

#         if voxel.shape == (64, 64, 32, 1):
#             voxel = voxel.squeeze(axis=-1)
#         elif voxel.ndim == 4:
#             voxel = np.squeeze(voxel)
#         elif voxel.ndim == 5:
#             voxel = np.squeeze(voxel)

#         if voxel.ndim != 3:
#             raise ValueError(f"Expected 3D data after preprocessing, got shape {voxel.shape}")

#         if self.transform:
#             voxel = self.transform(voxel)

#         voxel = np.expand_dims(voxel, axis=0)
#         image = torch.from_numpy(voxel)
#         image = image.permute(0, 3, 1, 2)

#         # return image, torch.tensor([label], dtype=torch.long)
#         return image, torch.tensor(label, dtype=torch.long)


