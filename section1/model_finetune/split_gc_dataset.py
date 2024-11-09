# 这个数据集是用来，划分GC数据集，将GC数据集划分成4/5的train和1/5的test

import os
import shutil
import random

# 定义源和目标文件夹
label_dir = '/home/hi/zhuzi/data/GCOutput/Label'
model_fineture_dir = os.path.join(label_dir, 'model_fineture')
train_fineture_dir = os.path.join(model_fineture_dir, 'train')
test_fineture_dir = os.path.join(model_fineture_dir, 'test')

# 创建目标文件夹
os.makedirs(train_fineture_dir, exist_ok=True)
os.makedirs(test_fineture_dir, exist_ok=True)

# 统计标签文件
label_files = []
for folder in ['train', 'test', 'val']:
    folder_path = os.path.join(label_dir, folder)
    if os.path.exists(folder_path):
        label_files.extend([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.label')])

# 随机打乱文件列表
random.shuffle(label_files)

# 划分文件并拷贝
train_files = label_files[:int(len(label_files) * 0.8)]
test_files = label_files[int(len(label_files) * 0.8):]

for file in train_files:
    shutil.copy(file, train_fineture_dir)

for file in test_files:
    shutil.copy(file, test_fineture_dir)

print(f"文件已成功拷贝到 {train_fineture_dir} 和 {test_fineture_dir}")
