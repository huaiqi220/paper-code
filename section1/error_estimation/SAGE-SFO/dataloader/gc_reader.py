import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import random
import torch
from torch.utils.data.distributed import DistributedSampler


def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    self.map = {}
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if len(line) < 20:
            continue
          if header: line.pop(0)
          self.map[int(i.split("/")[-1].split(".")[0])] = line
          self.lines.extend(line)
    else:
      with open(i) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)
        # map[int(i.split("/")[-1].split(".")[0])] = line

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    def get_relation(q_label, c_labels):
        # q_label: Tensor of shape [2] (X_q's label)
        # c_labels: List of Tensors, each of shape [2] (X_c's 9 labels)
        relation = []
        for c_label in c_labels:
            if q_label[0] < c_label[0] and q_label[1] < c_label[1]:
                relation.append(0)  # 左上
            elif q_label[0] >= c_label[0] and q_label[1] < c_label[1]:
                relation.append(1)  # 右上
            elif q_label[0] >= c_label[0] and q_label[1] >= c_label[1]:
                relation.append(2)  # 右下
            elif q_label[0] < c_label[0] and q_label[1] >= c_label[1]:
                relation.append(3)  # 左下
        return torch.tensor(relation, dtype=torch.int64)
    
    def func(line):
      line = line.strip().split(" ")

      name = line[0]
      # device = line[5]
      point = line[4]
      faceb = line[6].split(",")
      leftb = line[7].split(",")
      rightb = line[8].split(",")
      bbox = faceb + leftb + rightb
      face = line[0]
      lefteye = line[1]
      righteye = line[2]
      grid = line[3]

      label = np.array(point.split(",")).astype("float")
      label = torch.from_numpy(label).type(torch.FloatTensor)

      rect = np.array(bbox).astype("float")
      rect = torch.from_numpy(rect).type(torch.FloatTensor)

      rimg = cv2.imread(os.path.join(self.root, righteye))
      rimg = cv2.resize(rimg, (64, 64))/255.0
      rimg = rimg.transpose(2, 0, 1)

      limg = cv2.imread(os.path.join(self.root, lefteye))
      limg = cv2.resize(limg, (64, 64))/255.0
      limg = limg.transpose(2, 0, 1)
      
      fimg = cv2.imread(os.path.join(self.root, face))
      fimg = cv2.resize(fimg, (224, 224))/255.0
      fimg = fimg.transpose(2, 0, 1)
  
      grid = cv2.imread(os.path.join(self.root, grid), 0)
      grid = np.expand_dims(grid, 0)

      img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
              "right":torch.from_numpy(rimg).type(torch.FloatTensor),
              "face":torch.from_numpy(fimg).type(torch.FloatTensor),
              # "grid":torch.from_numpy(grid).type(torch.FloatTensor),
              "name":name,
              "rects":rect,
              "label":label,
              "device": "Android"}

      return img
    line = self.lines[idx]
    id = int(line.strip().split(" ")[0].split("/")[0])
    # 获取 X_q 和 X_c 数据
    X_q = func(line)
    X_c = [func(self.map[id][i]) for i in range(9)]
    # 计算位置关系向量
    relation_vector = get_relation(X_q["label"], [xc["label"] for xc in X_c])

    return [X_q, X_c, relation_vector]
  


class caliloader(Dataset): 
  def __init__(self, lines, root, header=True):
    self.lines = lines
    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[0]
    # device = line[5]
    point = line[4]
    faceb = line[6].split(",")
    leftb = line[7].split(",")
    rightb = line[8].split(",")
    bbox = faceb + leftb + rightb
    face = line[0]
    lefteye = line[1]
    righteye = line[2]
    grid = line[3]

    label = np.array(point.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    rect = np.array(bbox).astype("float")
    rect = torch.from_numpy(rect).type(torch.FloatTensor)

    rimg = cv2.imread(os.path.join(self.root, righteye))
    rimg = cv2.resize(rimg, (64, 64))/255.0
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, lefteye))
    limg = cv2.resize(limg, (64, 64))/255.0
    limg = limg.transpose(2, 0, 1)
    
    fimg = cv2.imread(os.path.join(self.root, face))
    fimg = cv2.resize(fimg, (224, 224))/255.0
    fimg = fimg.transpose(2, 0, 1)
 
    grid = cv2.imread(os.path.join(self.root, grid), 0)
    grid = np.expand_dims(grid, 0)

    img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
            "right":torch.from_numpy(rimg).type(torch.FloatTensor),
            "face":torch.from_numpy(fimg).type(torch.FloatTensor),
            # "grid":torch.from_numpy(grid).type(torch.FloatTensor),
            "name":name,
            "rects":rect,
            "label":label,
            "device": "Android"}

    return img





def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  distributed_sampler = DistributedSampler(dataset)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=distributed_sampler,pin_memory=True)
  # load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
  return load

def calitxtload(lines, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = caliloader(lines, imagepath, header)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,shuffle=shuffle,pin_memory=True)
  return load





if __name__ == "__main__":
  label = "/home/hi/zhuzi/data/GCOutput/Label/train"
  image = "/home/hi/zhuzi/data/GCOutput/Image"
  trains = os.listdir(label)
  trains = [os.path.join(label, j) for j in trains]
  print(trains)
  print(image)
  d = txtload(trains, image, 10)
  print(len(d))
  (data, label) = d.__iter__()

