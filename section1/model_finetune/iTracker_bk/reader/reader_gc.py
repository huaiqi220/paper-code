import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch

def gazeto2d(gaze):
  yaw = np.arctan2(-gaze[0], -gaze[2])
  pitch = np.arcsin(-gaze[1])
  return np.array([yaw, pitch])

class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    if isinstance(path, list):
      for i in path:
        with open(i) as f:
          line = f.readlines()
          if header: line.pop(0)
          self.lines.extend(line)
    else:
      with open(path) as f:
        self.lines = f.readlines()
        if header: self.lines.pop(0)

    self.root = root

  def __len__(self):
    return len(self.lines)

  def __getitem__(self, idx):
    line = self.lines[idx]
    line = line.strip().split(" ")

    name = line[0]
    device = line[9]
    point = line[4]

    face = line[0]
    lefteye = line[1]
    righteye = line[2]
    grid = line[3]

    label = np.array(point.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)

    rimg = cv2.imread(os.path.join(self.root, righteye))
    rimg = cv2.resize(rimg, (224, 224))/255.0
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, lefteye))
    limg = cv2.resize(limg, (224, 224))/255.0
    limg = limg.transpose(2, 0, 1)
    
    fimg = cv2.imread(os.path.join(self.root, face))/255.0
    fimg = cv2.resize(fimg, (224, 224))/255.0
    fimg = fimg.transpose(2, 0, 1)
 
    grid = cv2.imread(os.path.join(self.root, grid), 0)
    grid = np.expand_dims(grid, 0)

    img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
            "right":torch.from_numpy(rimg).type(torch.FloatTensor),
            "face":torch.from_numpy(fimg).type(torch.FloatTensor),
            "grid":torch.from_numpy(grid).type(torch.FloatTensor),
            "name":name,
            "device": device}

    return img, label

def txtload(labelpath, imagepath, batch_size, shuffle=False, num_workers=0, header=True, pin_memory=True):
  dataset = loader(labelpath, imagepath, header)
  distributed_sampler = DistributedSampler(dataset)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,sampler=distributed_sampler, pin_memory=True)
  return load


if __name__ == "__main__":
  label = "/data300m2/output/gazecapture/Label/train"
  image = "/data300m2/output/gazecapture/Image"
  trains = os.listdir(label)
  trains = [os.path.join(label, j) for j in trains]
  d = txtload(trains, image, 10)
  print(len(d))
  (data, label) = d.__iter__()

