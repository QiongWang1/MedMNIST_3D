#split_mito_csv.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Read original data
df = pd.read_csv('mito_labels.csv')
print(f"Total samples: {len(df)}")
print(f"Label distribution:")
label_counts = df['label'].value_counts()
print(label_counts)

# For very small datasets (< 50 samples), use a more conservative split
# Strategy: 60% train, 10% val, 30% test (to get 6 test samples from 20 total)
# This ensures test has 6 samples as requested

if len(df) <= 20:
    print("Small dataset detected. Using custom split strategy for Train: 9, Val: 5, Test: 6.")
    
    # First split: 70% train+val, 30% test (6 samples for test)
    train_val_df, test_df = train_test_split(
        df, 
        test_size=6,  # Directly specify 6 samples for test
        stratify=df['label'] if label_counts.min() >= 2 else None,
        random_state=42
    )
    
    # Second split: from remaining 14 samples, split into train (9) and val (5)
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=5,  # Directly specify 5 samples for validation
        stratify=train_val_df['label'] if train_val_df['label'].value_counts().min() >= 2 else None,
        random_state=42
    )

else:
    # Standard split for larger datasets
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

# Save splits
train_df.to_csv('mito_train.csv', index=False)
val_df.to_csv('mito_val.csv', index=False)
test_df.to_csv('mito_test.csv', index=False)

print(f"\nFinal split:")
print(f"Train: {len(train_df)} samples")
print(f"Val: {len(val_df)} samples")
print(f"Test: {len(test_df)} samples")

print(f"\nLabel distribution per split:")
print("Train:", train_df['label'].value_counts().to_dict())
print("Val:", val_df['label'].value_counts().to_dict())
print("Test:", test_df['label'].value_counts().to_dict())

print("\nSplit completed successfully!")


def show_distribution(name, df):
    print(f"{name}: total={len(df)}, pos={df['label'].sum()}, neg={(df['label']==0).sum()}")

show_distribution("Train", train_df)
show_distribution("Val", val_df)
show_distribution("Test", test_df)
