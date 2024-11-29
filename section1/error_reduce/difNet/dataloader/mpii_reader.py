import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import random
import torch
from torch.utils.data.distributed import DistributedSampler


def aug_line(line, width, height):
  bbox = np.array(line[2:5])
  bias = round(30 * random.uniform(-1, 1))
  bias = max(np.max(-bbox[0, [0, 2]]), bias)
  bias = max(np.max(-2 * bbox[1:, [0, 2]] + 0.5), bias)

  line[2][0] += int(round(bias))
  line[2][1] += int(round(bias))
  line[2][2] += int(round(bias))
  line[2][3] += int(round(bias))

  line[3][0] += int(round(0.5 * bias))
  line[3][1] += int(round(0.5 * bias))
  line[3][2] += int(round(0.5 * bias))
  line[3][3] += int(round(0.5 * bias))

  line[4][0] += int(round(0.5 * bias))
  line[4][1] += int(round(0.5 * bias))
  line[4][2] += int(round(0.5 * bias))
  line[4][3] += int(round(0.5 * bias))

  line[5][2] = line[2][2] / width
  line[5][3] = line[2][0] / height

  line[5][6] = line[3][2] / width
  line[5][7] = line[3][0] / height

  line[5][10] = line[4][2] / width
  line[5][11] = line[4][0] / height
  return line


def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        # print(i)
        # print(path)
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(i) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = str(line[0].split("/")[0])
    # device = line[5]
    point = line[6]

    ratio = line[9].split(",")

    label = np.array(point.split(",")).astype("float")
    ratio = np.array(ratio).astype("float")
    label = label*ratio*0.1
    label = torch.from_numpy(label).type(torch.FloatTensor)
    # faceb = line[6].split(",")
    # leftb = line[7].split(",")
    # rightb = line[8].split(",")
    rect = [float(p) for p in line[13].split(",")]
    # bbox = faceb + leftb + rightb
    face = line[0]
    # full = line[1]
    lefteye = line[1]
    righteye = line[2]
    grid = line[3]

    rect = np.array(rect).astype("float")
    rect = torch.from_numpy(rect).type(torch.FloatTensor)

    rimg = cv2.imread(os.path.join(self.root, righteye))
    rimg = cv2.resize(rimg, (48, 72))/255.0
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, lefteye))
    limg = cv2.resize(limg, (48, 72))/255.0
    limg = limg.transpose(2, 0, 1)
    
    fimg = cv2.imread(os.path.join(self.root, face))
    fimg = cv2.resize(fimg, (224, 224))/255.0
    fimg = fimg.transpose(2, 0, 1)
 
    grid = cv2.imread(os.path.join(self.root, grid), 0)
    # grid = cv2.resize(grid, (112, 112))/255.0
    grid = np.expand_dims(grid, 0)

    img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
            "right":torch.from_numpy(rimg).type(torch.FloatTensor),
            "face":torch.from_numpy(fimg).type(torch.FloatTensor),
            "grid":torch.from_numpy(grid).type(torch.FloatTensor),
            "name":name,
            "rects":rect,
            "label":label,
            "device": "Android"}

    return img

class caliloader(Dataset):
  def __init__(self, lines, root, header=True):
    self.lines = lines
    self.root = root
  
  def __len__(self):
    return len(self.lines)
  
  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = str(line[0].split("/")[0])
    # device = line[5]
    point = line[6]

    ratio = line[9].split(",")

    label = np.array(point.split(",")).astype("float")
    ratio = np.array(ratio).astype("float")
    label = label*ratio*0.1
    label = torch.from_numpy(label).type(torch.FloatTensor)
    # faceb = line[6].split(",")
    # leftb = line[7].split(",")
    # rightb = line[8].split(",")
    rect = [float(p) for p in line[13].split(",")]
    # bbox = faceb + leftb + rightb
    face = line[0]
    # full = line[1]
    lefteye = line[1]
    righteye = line[2]
    grid = line[3]

    rect = np.array(rect).astype("float")
    rect = torch.from_numpy(rect).type(torch.FloatTensor)

    rimg = cv2.imread(os.path.join(self.root, righteye))
    rimg = cv2.resize(rimg, (112, 112))/255.0
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, lefteye))
    limg = cv2.resize(limg, (112, 112))/255.0
    limg = limg.transpose(2, 0, 1)
    
    fimg = cv2.imread(os.path.join(self.root, face))
    fimg = cv2.resize(fimg, (224, 224))/255.0
    fimg = fimg.transpose(2, 0, 1)
 
    grid = cv2.imread(os.path.join(self.root, grid), 0)
    # grid = cv2.resize(grid, (112, 112))/255.0
    grid = np.expand_dims(grid, 0)

    img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
            "right":torch.from_numpy(rimg).type(torch.FloatTensor),
            "face":torch.from_numpy(fimg).type(torch.FloatTensor),
            "grid":torch.from_numpy(grid).type(torch.FloatTensor),
            "name":name,
            "rects":rect,
            "label":label,
            "device": "Android"}

    return img

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  # print(labelpath)
  # print(imagepath)
  dataset = loader(labelpath, imagepath, header)
  # distributed_sampler = DistributedSampler(dataset)
  print(f"[Read Data]: Total num: {len(dataset)}")
  # load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=distributed_sampler,pin_memory=True)
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,pin_memory=True)
  return load


def calitxtload(lines, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = caliloader(lines, imagepath, header)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,shuffle=shuffle,pin_memory=True)
  return load

if __name__ == "__main__":
  label = "/data/4_gc/2_gcout/Label/train"
  image = "/data/4_gc/2_gcout/Image"
  trains = os.listdir(label)
  trains = [os.path.join(label, j) for j in trains]
  print(trains)
  print(image)
  d = txtload(trains, image, 10)
  print(len(d))
  (data, label) = d.__iter__()

