
# download_h01_patch.py
import os
import csv
import numpy as np
from cloudvolume import CloudVolume

# === Parameters ===
csv_path = 'mito_labels.csv'
save_dir = 'mito_downloaded_patch'
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












# This following code is for test data.
# from cloudvolume import CloudVolume
# import numpy as np
# import os

# vol_path = 'precomputed://gs://h01-release/data/20210601/c3/'
# vol = CloudVolume(vol_path, progress=True, parallel=True, use_https=True)

# # List of patch centers (x, y, z)

# patch_centers = [
#     (249666, 188178, 1944),
#     (246105, 188555, 2002),
#     (246113, 188678, 2002),
#     (245987, 189382, 2002),
#     (246199, 189805, 2002),
#     (246053, 189865, 2002),
#     (246006, 189648, 2002),
#     (245813, 190462, 2002),
#     (245738, 190761, 2002),
#     (245901, 191085, 2002),

#     (245907, 191381, 2002),
#     (247810, 193163, 2002),
#     (247506, 193242, 2002),
#     (247349, 192801, 2002),
#     (247304, 193339, 1990),
#     (247509, 193526, 1990),
#     (247616, 192005, 1990),
#     (246847, 192519, 1990),
#     (248034, 192479, 1990),
#     (248244, 192425, 1990),
#     # ➕ continue adding more (x, y, z) centers
# ]


# # Patch size
# dx, dy, dz = 64, 64, 32

# save_dir = "mito_downloaded_patch"
# os.makedirs(save_dir, exist_ok=True)

# for center in patch_centers:
#     x, y, z = center
#     x_range = (x - dx // 2, x + dx // 2)
#     y_range = (y - dy // 2, y + dy // 2)
#     z_range = (z - dz // 2, z + dz // 2)

#     patch = vol[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]
    
#     patch_name = f"x{x_range[0]}_{x_range[1]}_y{y_range[0]}_{y_range[1]}_z{z_range[0]}_{z_range[1]}"
#     save_path = os.path.join(save_dir, f"{patch_name}.npy")
#     np.save(save_path, patch)

#     print(f"✅ Saved {patch_name}, shape: {patch.shape}")

