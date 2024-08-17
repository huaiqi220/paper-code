import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNetV2,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4

# SAGE模型 eye cnn backbone
class AlexBackbone(nn.Module):
    def __init__(self):
        super(AlexBackbone, self).__init__()
        # 定义卷积层
        self.eye_conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=2)  # 注意调整 in_channels
        self.eye_conv2 = nn.Conv2d(96, 256, 5)
        self.eye_conv3 = nn.Conv2d(256, 384, 3)
        self.eye_conv4 = nn.Conv2d(384, 64, 1)

        # 定义批量归一化层
        self.bn_leye_conv1 = nn.BatchNorm2d(96)
        self.bn_leye_conv2 = nn.BatchNorm2d(256)
        self.bn_leye_conv3 = nn.BatchNorm2d(384)
        self.bn_leye_conv4 = nn.BatchNorm2d(64)
        self.bn_eye_dense = nn.BatchNorm1d(128)

        # 定义全连接层
        self.eye_dense1 = nn.Linear(4096, 128)  # 注意调整输入特征的数量

    def forward(self, input_leye):
        # 左眼
        leye = self.eye_conv1(input_leye)
        leye = F.relu(leye)
        leye = F.max_pool2d(leye, kernel_size=3, stride=2)
        leye = self.bn_leye_conv1(leye)

        leye = self.eye_conv2(leye)
        leye = F.relu(leye)
        leye = F.max_pool2d(leye, kernel_size=3, stride=2)
        leye = self.bn_leye_conv2(leye)

        leye = self.eye_conv3(leye)
        leye = F.relu(leye)
        leye = self.bn_leye_conv3(leye)

        leye = self.eye_conv4(leye)
        leye = F.relu(leye)
        leye = self.bn_leye_conv4(leye)

        leye = torch.flatten(leye, 1) # 展平操作，1 表示从第二维开始展平
        leye = self.eye_dense1(leye)
        leye = self.bn_eye_dense(leye)
        return leye

class Backbone(nn.Module):
    def __init__(self,backbone_name):
        super(Backbone,self).__init__()
        if backbone_name == "MobileNetV2":
            self.backbone = MobileNetV2()
        elif backbone_name == "EfficientNetB0":
            self.backbone = efficientnet_b0()
        elif backbone_name == "EfficientNetB1":
            self.backbone = efficientnet_b1()
        elif backbone_name == "EfficientNetB2":
            self.backbone = efficientnet_b2()
        elif backbone_name == "EfficientNetB3":
            self.backbone = efficientnet_b3()
        elif backbone_name == "EfficientNetB4":
            self.backbone = efficientnet_b4()
        self.backbone.classifier = nn.Sequential(nn.Identity())
        self.dropout = nn.Dropout1d(0.5)
        self.bn1 = nn.BatchNorm1d(1280)
        self.Linear = nn.Linear(1280,128)
        self.relu = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(128)

    
    def forward(self, input_leye):
        leye = self.backbone(input_leye)
        leye = self.dropout(leye)
        leye = self.bn1(leye)
        leye = self.Linear(leye)
        leye = self.relu(leye)
        leye = self.bn2(leye)
        return leye

# 使用示例
# model = EyeNetwork()
# output = model(input_leye_tensor)
model = Backbone("EfficientNetB1")
res  = model(torch.ones(10,3,112,112))
print(res.shape)
