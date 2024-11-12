import torch
import torch.nn as nn
import torch.nn.functional as F
#from Torch.ntools import AlexNet
import math
import torchvision





if __name__ == '__main__':
    m = model()
    feature = {"left": torch.zeros(10,1, 36,60),
                "right": torch.zeros(10,1, 36,60)
                }
    feature = {"faceImg": torch.zeros(10, 3, 224, 224), "leftEyeImg": torch.zeros(10, 3, 112, 112),
               "rightEyeImg": torch.zeros(10, 3, 112, 112), "faceGridImg": torch.zeros(10, 12),
               "label": torch.zeros(10, 2), "frame": "test.jpg"}

