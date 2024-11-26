import os
import shutil
import random

# Paths
label_path = "/home/hi/zhuzi/data/mpii_final/Label"
kf_path = "/home/hi/zhuzi/data/mpii_final/k_fold"

# Create k-fold directories
folds = ["1", "2", "3", "4"]
for fold in folds:
    fold_dir = os.path.join(kf_path, fold)
    os.makedirs(os.path.join(fold_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(fold_dir, "test"), exist_ok=True)

# List all files in Label path
all_files = os.listdir(label_path)
random.shuffle(all_files)  # Shuffle files randomly

# Split files into four parts
k = 4
split_files = [all_files[i::k] for i in range(k)]

# Map fold numbers to train and test splits
fold_mapping = {
    "1": (split_files[1] + split_files[2] + split_files[3], split_files[0]),  # BCD for train, A for test
    "2": (split_files[0] + split_files[2] + split_files[3], split_files[1]),  # ACD for train, B for test
    "3": (split_files[0] + split_files[1] + split_files[3], split_files[2]),  # ABD for train, C for test
    "4": (split_files[0] + split_files[1] + split_files[2], split_files[3]),  # ABC for train, D for test
}

# Distribute files according to fold mapping
for fold, (train_files, test_files) in fold_mapping.items():
    # Copy train files
    for file in train_files:
        shutil.copy(os.path.join(label_path, file), os.path.join(kf_path, fold, "train"))

    # Copy test files
    for file in test_files:
        shutil.copy(os.path.join(label_path, file), os.path.join(kf_path, fold, "test"))

print("K-fold setup completed.")
