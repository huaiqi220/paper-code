import os
import numpy as np
import pandas as pd
import sys
import torch
import cv2
import random


'''
将原始格式标签转换为diff模型可使用的标签

'''


label_path = "/data/4_gc/2_gcout/Label/test"
output_path = "/data/4_gc/2_gcout/Label_Diff/test"

file_list = os.listdir(label_path)

print(file_list)

for item in file_list:
    label_path1 = os.path.join(label_path,item)
    odata = []
    result = []

    with open(label_path1) as f:
        line = f.readlines()
        line.pop(0)
        odata = line
        f.close()
    
    f1 = 0
    f2 = 1
    random.shuffle(odata)
    while f2 < len(odata):
        fl1 = odata[f1]
        fl2 = odata[f2]

        fl1 = fl1.split(" ")
        point1 = fl1[4]

        point1 = [float(point1.split(",")[0]),float(point1.split(",")[1])]

        lefteye1 = fl1[1]
        righteye1 = fl1[2]

        fl2 = fl2.split(" ")
        point2 = fl2[4]
        point2 = [float(point2.split(",")[0]),float(point2.split(",")[1])]

        lefteye2 = fl2[1]
        righteye2 = fl2[2]

        rec1 = str(lefteye1) + " " + str(lefteye2) + " " + ",".join([str(point2[0] - point1[0]),str(point2[1] - point1[1])]) + " l" + "\n"
        rec2 = str(righteye1) + " " + str(righteye2) + " " + ",".join([str(point2[0] - point1[0]),str(point2[1] - point1[1])]) + " r" + "\n"
        result.append(rec1)
        result.append(rec2)
        f1 = f1 + 1
        f2 = f2 + 1
    
    output_path1 = os.path.join(output_path,item)
    with open(output_path1, "w") as file:
        # 遍历列表中的每个元素并将其写入文件
        file.write("eye1 eye2 point mark")
        for item in result:
            file.write(item )  # 写入列表元素并添加换行符
    






# if __name__ == "__main__":
#   label = "/data300m2/output/gazecapture/Label/train"
#   image = "/data300m2/output/gazecapture/Image"
#   trains = os.listdir(label)
#   trains = [os.path.join(label, j) for j in trains]
#   print(trains)
#   print(image)
#   d = txtload(trains, image, 10)
#   print(len(d))
#   (data, label) = d.__iter__()
