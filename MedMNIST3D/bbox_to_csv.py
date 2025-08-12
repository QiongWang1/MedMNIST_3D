#bbox_to_csv.py
import csv

def parse_bbox_file(filepath, label):
    centers = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            zmin, zmax, ymin, ymax, xmin, xmax = map(int, parts)
            z = (zmin + zmax) // 2
            y = (ymin + ymax) // 2
            x = (xmin + xmax) // 2
            centers.append([z, y, x, label])
    return centers

def main():
    output_csv = 'mito_labels.csv'
    all_centers = []

    all_centers += parse_bbox_file('bbox_fp.txt', label=0)  # FP
    all_centers += parse_bbox_file('bbox_tp.txt', label=1)  # TP

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['z', 'y', 'x', 'label'])
        writer.writerows(all_centers)

    print(f"✅ Saved {output_csv} with {len(all_centers)} entries.")

if __name__ == '__main__':
    main()
