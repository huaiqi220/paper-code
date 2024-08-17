import numpy as np
import cv2 
import os
from torch.utils.data import Dataset, DataLoader
import random
import torch
from config import sage_model_config as config
from torch.utils.data.distributed import DistributedSampler
from math import exp, sqrt, pi
def gauss(x, stdv=0.5):
    return exp(-(1/(2*stdv))*(x**2))/(sqrt(2*pi*stdv))

class loader(Dataset): 
  def __init__(self, file_path, label,regions):
    self._file_path = file_path
    self._dots = label
    self._region = regions
    self._size = self._region.shape[0]

  def __len__(self):
    return self._size

  def __getitem__(self, idx):
    # sampleID = self._region[:2] # subjectID, frameID
    sampleID = self._region[0]
    frameID = self._region[1]
    # leye_img = cv2.imread(self._file_path[idx][0], 0)
    # reye_img = cv2.imread(self._file_path[idx][1], 0)
    # face_img = cv2.imread(self._file_path[idx], 0) # Grey degree
    face_img = cv2.imread(self._file_path[idx]) # RGB
    region = self._region[idx]
    # print(region)
    face_img = face_img[region[5] : (region[5] + region[2]), region[4] : (region[4] + region[3])]
    leye_img = face_img[region[10] : (region[10] + region[7]), region[9] : (region[9] + region[8])]
    reye_img = face_img[region[15] : (region[15] + region[12]), region[14] : (region[14] + region[13])]
    leye_img = cv2.resize(leye_img, (config.eyeIm_size,config.eyeIm_size))/255.0
    reye_img = cv2.resize(reye_img, (config.eyeIm_size,config.eyeIm_size))/255.0
    face_img = cv2.resize(face_img, (config.faceIm_size,config.faceIm_size))/255.0
    # print(leye_img.shape, reye_img.shape, face_img.shape)
    leye_img = leye_img.transpose(2, 0, 1)
    reye_img = reye_img.transpose(2, 0, 1)
    face_img = face_img.transpose(2, 0, 1)
    leye_img = cv2.flip(leye_img, 1)
    # eyelandmark = torch.from_numpy(region[8:11] + region[13:16]).type(torch.FloatTensor) / 640.0
    eyelandmark = torch.from_numpy(np.concatenate((region[8:11], region[13:16]), axis=0)).type(torch.FloatTensor) / 640.0
    # print(region[24])
    region[24] = np.clip(region[24], 0, 2)
    orientation = torch.nn.functional.one_hot(torch.tensor(region[24]), num_classes=3)
    label = torch.from_numpy(self._dots[idx]).type(torch.FloatTensor)
    if(config.heatmap):
      hmFocus_size = 17 
      if(config.mobile):
        hmFocus_size = 9
      HM_FOCUS_IM = np.zeros((5, hmFocus_size, hmFocus_size, 1))
      stdv_list = [0.2, 0.25, 0.3, 0.35, 0.4]
      for level in range(5):  # 5 levels of std to constuct heatmap
          stdv = stdv_list[level]  # 3/(12-level)
          for i in range(hmFocus_size):
              for j in range(hmFocus_size):
                  distanceFromCenter = 2 * \
                      np.linalg.norm(np.array([i-int(hmFocus_size/2),
                                                j-int(hmFocus_size/2)]))/((hmFocus_size)/2)
                  gauss_prob = gauss(distanceFromCenter, stdv)
                  HM_FOCUS_IM[level, i, j, 0] = gauss_prob
      HM_FOCUS_IM[level, :, :, 0] /= np.sum(HM_FOCUS_IM[level, :, :, 0])
      heatmap_im = torch.from_numpy(HM_FOCUS_IM[0, :, :, :]).type(torch.FloatTensor)
      heatmap_im = heatmap_im.permute(2, 0, 1) 
      pad_top = int(label[0] * config.scale + config.hm_size / 2 - hmFocus_size / 2)
      pad_left = int(label[1] * config.scale + config.hm_size / 2 - hmFocus_size / 2)
      heatmap_im = torch.nn.functional.pad(heatmap_im, 
                                           (pad_left, 
                                            config.hm_size - pad_left - hmFocus_size, 
                                            pad_top, 
                                            config.hm_size - pad_top - hmFocus_size), 
                                           mode='constant', value=0)
      label = heatmap_im
    # if (config.arc == 'SAGE'):
    #     return (orientation, eyelandmark, leye_im, reye_im, label)
    # else:
    #     return (orientation, face_grid_im, face_im, leye_im, reye_im, label)
    # print(leye_img.shape, reye_img.shape, face_img.shape, orientation.shape, eyelandmark.shape, label.shape)
    # print("-------------------------------")
    img = {"left":torch.from_numpy(leye_img).type(torch.FloatTensor),
          "right":torch.from_numpy(reye_img).type(torch.FloatTensor),
          "face":torch.from_numpy(face_img).type(torch.FloatTensor),
          "orientation":orientation,
          "eyelandmark":eyelandmark,
          # "grid":torch.from_numpy(grid).type(torch.FloatTensor),
          "name":str(sampleID),
          "frameID":str(frameID),
          # "rects":rect,
          "label":label,
          "device": "Android"}  
    return img

    

def txtload(file_path, label, regions, batch_size, shuffle=True, num_workers=0):
  # print(labelpath)
  # print(imagepath)
  dataset = loader(file_path, label, regions)
  distributed_sampler = DistributedSampler(dataset)
  print(f"[Read Data]: Total num: {len(dataset)}")
  load = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,sampler=distributed_sampler,pin_memory=True)
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

