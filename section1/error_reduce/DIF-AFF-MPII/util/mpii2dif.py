'''
把MPII数据集划分成DIF类型数据集

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


label_root_path = "/home/hi/zhuzi/data/mpii/Label/K_Fold_diff"
for folder in ["1","2","3","4"]:
    cur_folder = os.path.join(label_root_path,folder)
    new_folder = os.path.join(label_root_path,"diflabel",folder)

    train_folder = os.path.join(new_folder,"train")
    test_folder = os.path.join(new_folder,"test")

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    for file in os.listdir(os.path.join(cur_folder,"train")):
        origin_file = os.path.join(cur_folder,"train",file)
        lines = []
        with open(origin_file,'r') as f:
            lines = f.readlines()
        new_lines = getFullLines(lines[1:])
        save_path = os.path.join(train_folder,file)
        with open(save_path,'w') as f:
            f.writelines(line + '\n' for line in new_lines)  
        

    for file in os.listdir(os.path.join(cur_folder,"test")):
        origin_file = os.path.join(cur_folder,"test",file)
        lines = []
        with open(origin_file,'r') as f:
            lines = f.readlines()
        new_lines = getFullLines(lines[1:])
        save_path = os.path.join(test_folder,file)
        with open(save_path,'w') as f:
            f.writelines(line + '\n' for line in new_lines)
