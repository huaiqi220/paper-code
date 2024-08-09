import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import torch
import dataloader.init_cali as init_cali
from torch.utils.data.distributed import DistributedSampler
from math import exp, sqrt, pi
import torch.nn.functional as F
import matplotlib.pyplot as plt
import util.htools as htools
import config





class loader(Dataset): 
  def __init__(self, path, root, header=True):
    self.lines = []
    self.loaded_vectors = init_cali.load_vectors_from_file(config.GazeCapture_Cali_path)
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

    cur_person_id = line[0].split("/")[0]
    face_path = line[0]
    left_path = line[1]
    right_path = line[2]
    grid_path = line[3]

    point = line[4]
    cali = self.loaded_vectors[int(cur_person_id)]
    cali = torch.from_numpy(cali).type(torch.FloatTensor)


    label = np.array(point.split(",")).astype("float")
    label = torch.from_numpy(label).type(torch.FloatTensor)
    label = htools.pog2heatmap(label)

    # rect = np.array(bbox).astype("float")
    # rect = torch.from_numpy(rect).type(torch.FloatTensor)

    rimg = cv2.imread(os.path.join(self.root, right_path))
    rimg = cv2.resize(rimg,  (224, 224))/255.0
    rimg = rimg.transpose(2, 0, 1)

    limg = cv2.imread(os.path.join(self.root, left_path))
    limg = cv2.resize(limg,  (224, 224))/255.0
    limg = limg.transpose(2, 0, 1)
    
    fimg = cv2.imread(os.path.join(self.root, face_path))
    fimg = cv2.resize(fimg, (112, 112))/255.0
    fimg = fimg.transpose(2, 0, 1)
 
    grid = cv2.imread(os.path.join(self.root, grid_path), 0)
    grid = np.expand_dims(grid, 0)

    img = {"left":torch.from_numpy(limg).type(torch.FloatTensor),
            "right":torch.from_numpy(rimg).type(torch.FloatTensor),
            "face":torch.from_numpy(fimg).type(torch.FloatTensor),
            "grid":torch.from_numpy(grid).type(torch.FloatTensor),
            "name":str(cur_person_id),
            "cali":cali,
            # "rects":rect,
            "label":label,
            "device": "Android"}

    return img

def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):

  dataset = loader(labelpath, imagepath, header)
  distributed_sampler = DistributedSampler(dataset)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=distributed_sampler)
  # load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
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
  # heatmap = pog2heatmap([5, 10])
  # heatmap = heatmap.squeeze().numpy()
  # plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
  # plt.colorbar()  # Add a colorbar to the side
  # plt.title('Heatmap Visualization')
  # plt.savefig('heatmap.png')
