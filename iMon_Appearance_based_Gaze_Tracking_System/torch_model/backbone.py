import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2,efficientnet_b0,efficientnet_b1,efficientnet_b2,efficientnet_b3,efficientnet_b4


''' model config '''
import model_config as config

eyeIm_shape = (config.eyeIm_size, config.eyeIm_size, config.channel)

dropout_rate = 0.5

class DynamicLinear(nn.Module):
    def __init__(self, out_features):
        super(DynamicLinear, self).__init__()
        self.out_features = out_features
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        in_features = x.size(1)
        if not hasattr(self, 'fc'):
            self.fc = nn.Linear(in_features, self.out_features).to(x.device)
        return self.fc(x)

# SAGE模型 AlexBackbone
class AlexBackbone(nn.Module):
    def __init__(self):
        super(AlexBackbone, self).__init__()
        # 定义卷积层
        self.eye_conv1 = nn.Sequential(
            nn.Conv2d(config.channel,96,kernel_size=11,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.BatchNorm2d(96),
        )
        self.eye_conv2 = nn.Sequential(
            # 未指定stride时默认是1
            nn.Conv2d(96,256,kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            nn.BatchNorm2d(256),
        )
        self.eye_conv3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(384),
        )
        self.eye_conv4 = nn.Sequential(
            nn.Conv2d(384,64,kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.flatten = nn.Sequential(
            nn.Flatten(),
            DynamicLinear(128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

        )


    def forward(self, input_leye):
        x = self.eye_conv1(input_leye)
        x = self.eye_conv2(x)
        x = self.eye_conv3(x)
        x = self.eye_conv4(x)
        x = self.flatten(x)
        return x


class Backbone(nn.Module):
    def __init__(self,backbone_name):
        super(Backbone,self).__init__()
        if backbone_name == "MobileNetV2":
            self.backbone = mobilenet_v2(pretrained=True).features
            self.feature_size = 1280  # MobileNetV2's last conv layer has 1280 channels
        elif backbone_name == "EfficientNetB0":
            self.backbone = efficientnet_b0(pretrained=False).features
            self.feature_size = 1280
        elif backbone_name == "EfficientNetB1":
            self.backbone = efficientnet_b1(pretrained=False).features
            self.feature_size = 1280
        elif backbone_name == "EfficientNetB2":
            self.backbone = efficientnet_b2(pretrained=False).features
            self.feature_size = 1408
        elif backbone_name == "EfficientNetB3":
            self.backbone = efficientnet_b3(pretrained=False).features
            self.feature_size = 1536
        elif backbone_name == "EfficientNetB4":
            self.backbone = efficientnet_b4(pretrained=False).features
            self.feature_size = 1792
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
            nn.BatchNorm1d(self.feature_size),
            nn.Linear(self.feature_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

    def forward(self, input_leye):
        x = self.backbone(input_leye)
        x = self.linear(x)
        print(x.shape)

        return x
    

if __name__ == "__main__":
    # model = Backbone("EfficientNetB3")
    model = AlexBackbone()
    input_test = torch.ones(10,3,112,112)
    x = model(input_test)
    print(x.shape)