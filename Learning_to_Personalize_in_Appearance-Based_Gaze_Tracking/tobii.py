import torch
import torch.nn as nn
import torchvision.models as models

import torch.functional as F


class ResNet18Conv(nn.Module):
    def __init__(self):
        super(ResNet18Conv, self).__init__()
        # Load the pre-trained ResNet-18 model
        resnet18 = models.resnet18(pretrained=False)
        # Extract the convolutional part (up to the last conv layer)
        self.features = nn.Sequential(*list(resnet18.children())[:-2])
    
    def forward(self, x):
        return self.features(x)
    
class SameModel(nn.Module):
    def __init__(self):
        super(SameModel, self).__init__()
        self.face_cnn = ResNet18Conv()
        self.eye_cnn = ResNet18Conv()
