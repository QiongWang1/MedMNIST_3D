#bbox_to_csv_inference.py

import pandas as pd

# Input/output
input_txt = "/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_test_set/mito_cls_bbox_590612150_full.txt"
output_csv = "mito_infer_centers.csv"

# Read .txt file: 7 columns
df = pd.read_csv(input_txt, sep="\s+", header=None)

# Drop the last column (volume ID)
df = df.iloc[:, :6]

# Rename columns for clarity
df.columns = ["z_start", "z_end", "y_start", "y_end", "x_start", "x_end"]

# Calculate center points
df["z"] = ((df["z_start"] + df["z_end"]) // 2).astype(int)
df["y"] = ((df["y_start"] + df["y_end"]) // 2).astype(int)
df["x"] = ((df["x_start"] + df["x_end"]) // 2).astype(int)

# Keep only center coordinates
df_center = df[["z", "y", "x"]]

# Add virtual tag columns (all set to 0, this will not affect the inference results)
df_center["label"] = 0

# Save to CSV
df_center.to_csv(output_csv, index=False)

print(f"✅ Inference center CSV with dummy labels saved as {output_csv}, total entries: {len(df_center)}")
print(f"📝 Note: Added dummy label column (all 0s) for compatibility - this won't affect inference results")