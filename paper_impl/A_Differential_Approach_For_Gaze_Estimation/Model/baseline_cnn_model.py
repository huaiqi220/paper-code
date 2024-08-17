import torch
import torch.nn as nn
import torch.nn.functional as F
#from Torch.ntools import AlexNet
import math
import torchvision

'''
本代码对应PAMI 2019 A Differential Approach for Gaze Estimation 论文中的baseline CNN模型
个性化微调->差分模型

'''

class eyeGazeNet(nn.Module):
    def __init__(self):
        super(eyeGazeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

    def forward(self,eye_image):
        x = self.conv1(eye_image)
        x = self.conv2(x)
        x = self.conv3(x)
        return x



class oriGazeNet(nn.Module):
    def __init__(self):
        super(oriGazeNet,self).__init__()
        self.eyeModel = eyeGazeNet()

        self.flattern = nn.Sequential(
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(4608, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
    
    def forward(self,x):
        x = self.eyeModel(x)
        x = self.flattern(x)
        x = self.linear(x)
        return x

if __name__ == '__main__':
    m = oriGazeNet()
    feature = torch.ones(10,3,48,72)
    a = m(feature)
    print(a.shape)
