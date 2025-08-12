# inference_preview_patch.py

import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Define the input folder containing all .npy patch files
input_dir = "/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_infer_patch"

# Define the output folder to store the preview images
output_dir = "inference_mito_slices_preview"

# Get all .npy files in the input directory
all_patch_files = [f for f in sorted(os.listdir(input_dir)) if f.endswith(".npy")]

# Randomly select 10 files
random.seed(42)  
selected_files = random.sample(all_patch_files, min(10, len(all_patch_files)))

n_files = 0
for patch_filename in selected_files:
    print(f"Processing: {patch_filename}")
    n_files += 1

    # Load the 3D patch
    patch_path = os.path.join(input_dir, patch_filename)
    patch = np.load(patch_path)
    print(f"  Loaded patch with shape: {patch.shape}")

    # Create a subfolder for this patch's preview images
    preview_dir = os.path.join(output_dir, patch_filename.replace(".npy", ""))
    os.makedirs(preview_dir, exist_ok=True)

    # Save one slice per z-plane
    for i in range(patch.shape[2]):
        plt.imshow(patch[:, :, i, 0], cmap='gray')
        plt.title(f"Slice {i} of patch")
        plt.axis('off')
        output_path = os.path.join(preview_dir, f"slice_{i}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"    Saved: {output_path}")

print(f"\n✅ All done! Processed {n_files} random patch files.")
