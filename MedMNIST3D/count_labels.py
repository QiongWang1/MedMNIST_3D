import csv
from collections import Counter

label_counts = Counter()

with open('mito_labels.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        label = int(row['label'])
        label_counts[label] += 1

print("✅ Label counts:")
for label, count in label_counts.items():
    print(f"Label {label}: {count} samples")
