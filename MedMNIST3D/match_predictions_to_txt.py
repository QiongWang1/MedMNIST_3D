# match_predictions_to_txt.py

import pandas as pd

# Input file paths
txt_file = "/projects/weilab/qiongwang/MedMNIST_experiments/MedMNIST_experiments/MedMNIST3D/mito_test_set/mito_cls_bbox_590612150_full.txt"
pred_csv = "./inference_results/inference_250703_142336/infer_predictions.csv"
output_file = txt_file.replace(".txt", "_with_predLabel_confidence.txt")

# Load predictions CSV and build a lookup dictionary: {(z, y, x): (predicted_label, confidence)}
pred_df = pd.read_csv(pred_csv)
pred_dict = {
    (int(row["z"]), int(row["y"]), int(row["x"])): (int(row["predicted_label"]), float(row["confidence"]))
    for _, row in pred_df.iterrows()
}

# Read the original .txt file
with open(txt_file, "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    # Parse 7 values: z_start, z_end, y_start, y_end, x_start, x_end, volume
    parts = list(map(int, line.strip().split()))
    z_start, z_end, y_start, y_end, x_start, x_end, volume = parts

    # Calculate the center point
    z_center = (z_start + z_end) // 2
    y_center = (y_start + y_end) // 2
    x_center = (x_start + x_end) // 2
    key = (z_center, y_center, x_center)

    # Look up predicted_label and confidence
    predicted_label, confidence = pred_dict.get(key, (-1, -1.0))  # default if not found

    # Create updated line with two new columns
    new_line = f"{z_start} {z_end} {y_start} {y_end} {x_start} {x_end} {volume} {predicted_label} {confidence:.4f}\n"
    new_lines.append(new_line)

# Save the updated lines to a new file
with open(output_file, "w") as f:
    f.writelines(new_lines)

print(f"✅ Done: Output saved with predicted labels and confidence → {output_file}")
