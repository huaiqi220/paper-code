'''
把GC数据集划分成DIF类型数据集

'''

import os
import sys
import random

def getNewLines(lines,i):
    return lines[i][:-1] + " || " + lines[random.randint(0, len(lines) - 1)][:-1]

def getFullLines(lines):
    new_lines = []
    for i in range(len(lines)):
        new_lines.append(getNewLines(lines,i))
    return new_lines


label_root_path = "/home/hi/zhuzi/data/GCOutput/Label/model_fineture"


new_folder = "/home/hi/zhuzi/data/GCOutput/Label/diflabel"
train_folder = os.path.join(new_folder,"train")
test_folder = os.path.join(new_folder,"test")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for file in os.listdir(os.path.join(label_root_path,"train")):
    origin_file = os.path.join(label_root_path,"train",file)
    lines = []
    with open(origin_file,'r') as f:
        lines = f.readlines()
    new_lines = getFullLines(lines[1:])
    save_path = os.path.join(train_folder,file)
    with open(save_path,'w') as f:
        f.writelines(line + '\n' for line in new_lines)  
    

for file in os.listdir(os.path.join(label_root_path,"test")):
    origin_file = os.path.join(label_root_path,"test",file)
    lines = []
    with open(origin_file,'r') as f:
        lines = f.readlines()
    new_lines = getFullLines(lines[1:])
    save_path = os.path.join(test_folder,file)
    with open(save_path,'w') as f:
        f.writelines(line + '\n' for line in new_lines)
