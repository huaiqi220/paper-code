import os
import shutil
import random

# 设置原始和目标文件夹路径
source_dir_112 = '/data/8_FFHQ/112'
source_dir_crappified_112 = '/data/8_FFHQ/crappified_112'
target_dir_112 = '/data/8_FFHQ/test/112'
target_dir_crappified_112 = '/data/8_FFHQ/test/crappified_112'

# 确保目标文件夹存在
os.makedirs(target_dir_112, exist_ok=True)
os.makedirs(target_dir_crappified_112, exist_ok=True)

def move_random_files(source_dir_112,source_dir_c112, target_dir_112, target_dir_c112):
    for i in range(70000):
        willCut = random.random() < 0.2
        if(willCut):
            ofilename = str(i).zfill(5) + ".png"
            cofilename = "crappified_" + str(i).zfill(5) + ".png"
            source_path = os.path.join(source_dir_112, ofilename)
            target_path = os.path.join(target_dir_112, ofilename)
            csource_path = os.path.join(source_dir_c112, cofilename)
            ctarget_path = os.path.join(target_dir_c112, cofilename)
            shutil.move(source_path, target_path)
            shutil.move(csource_path, ctarget_path)
            print(f'Moved {ofilename} to {target_dir_112}')

move_random_files(source_dir_112,source_dir_crappified_112, target_dir_112, target_dir_crappified_112)

print("Files have been moved successfully.")
