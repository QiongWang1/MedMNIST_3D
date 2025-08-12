
# download_inference_patch.py
import os
import csv
import numpy as np
from cloudvolume import CloudVolume

# === Parameters ===
csv_path = 'mito_infer_centers.csv'
save_dir = 'mito_infer_downloaded_patch'
vol_path = 'precomputed://gs://h01-release/data/20210601/4nm_raw'
mip = 1  # Use mip=1 as instructed
patch_size = (32, 32, 8)  # (x, y, z) → consistent with MedMNIST3D

# === Setup ===
os.makedirs(save_dir, exist_ok=True)

# Initialize volume
vol = CloudVolume(vol_path, mip=mip, parallel=True, use_https=True, progress=True)

# === Read mito_labels.csv and download patches ===
with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for idx, row in enumerate(reader):
        z = int(row['z'])
        y = int(row['y'])
        x = int(row['x'])
        label = int(row['label'])

        # Compute patch range
        dx, dy, dz = patch_size
        x_range = (x - dx // 2, x + dx // 2)
        y_range = (y - dy // 2, y + dy // 2)
        z_range = (z - dz // 2, z + dz // 2)

        # Load patch
        try:
            patch = vol[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]
        except Exception as e:
            print(f"❌ Failed to load patch {idx} at ({z},{y},{x}): {e}")
            continue

        # Save patch
        fname = f"z{z}_y{y}_x{x}_label{label}.npy"
        fpath = os.path.join(save_dir, fname)
        np.save(fpath, patch)

        print(f"[{idx+1:03}] ✅ Saved {fname}, shape: {patch.shape}")



