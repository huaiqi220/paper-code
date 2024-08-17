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
          # if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(i) as f:
        self.lines = f.readlines()
        # if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")
    point = line[2]
    point = point.split(",")
    # print(point)
    point = [point[0][1:],point[1][:-1]]

    eye1 = line[0]
    eye2 = line[1]

    label = np.array(point).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    eye1 = cv2.imread(os.path.join(self.root, eye1))
    eye1 = cv2.resize(eye1, (48, 72))/255.0
    eye1 = eye1.transpose(2, 0, 1)

    eye2 = cv2.imread(os.path.join(self.root, eye2))
    eye2 = cv2.resize(eye2 , (48, 72))/255.0
    eye2 = eye2.transpose(2, 0, 1)
    

    img = {"eye1":torch.from_numpy(eye1).type(torch.FloatTensor),
            "eye2":torch.from_numpy(eye2).type(torch.FloatTensor),
            # "grid":torch.from_numpy(grid).type(torch.FloatTensor),
            "label":label,
            "device": "Android"}

    return img

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
  dataset = loader(labelpath, imagepath, header)
  distributed_sampler = DistributedSampler(dataset)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=distributed_sampler)
  return load


if __name__ == "__main__":
  label = "/data300m2/output/gazecapture/Label_dif/train"
  image = "/data300m2/output/gazecapture/Image"
  trains = os.listdir(label)
  trains = [os.path.join(label, j) for j in trains]
  print(trains)
  print(image)
  d = txtload(trains, image, 10)
  print(len(d))
  (data, label) = d.__iter__()

