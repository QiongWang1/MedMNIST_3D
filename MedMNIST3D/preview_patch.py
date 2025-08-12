import os
import numpy as np
import matplotlib.pyplot as plt

# Define the input folder containing all .npy patch files
input_dir = "mito_downloaded_patch"

# Define the output folder to store the preview images
output_dir = "mito_slices_preview"

# Loop over all files in the input folder
n_files = 0
for patch_filename in sorted(os.listdir(input_dir)):
    if not patch_filename.endswith(".npy"):
        continue  # Skip non-NPY files

    print(f"Processing: {patch_filename}")
    n_files += 1

    # Load the 3D patch
    patch_path = os.path.join(input_dir, patch_filename)
    patch = np.load(patch_path)
    print(f"  Loaded patch with shape: {patch.shape}")

    # Create a subfolder for this patch's preview images
    preview_dir = os.path.join(output_dir, patch_filename.replace(".npy", ""))
    os.makedirs(preview_dir, exist_ok=True)

    # Save one slice every 8 frames along the z-axis
    # # for i in range(0, patch.shape[2], 8):
    # for i in range(patch.shape[2]):
    #     plt.imshow(patch[:, :, i, 0], cmap='gray')
    #     plt.title(f"Slice {i} of patch")
    #     plt.axis('off')
    #     plt.savefig(os.path.join(preview_dir, f"slice_{i}.png"))
    #     plt.close()
    #     print(f"    Saved: {output_path}")

    for i in range(patch.shape[2]):
        plt.imshow(patch[:, :, i, 0], cmap='gray')
        plt.title(f"Slice {i} of patch")
        plt.axis('off')
        output_path = os.path.join(preview_dir, f"slice_{i}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"    Saved: {output_path}")

print(f"\n✅ All done! Processed {n_files} patch files.")


