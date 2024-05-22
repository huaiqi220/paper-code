import torch
import torch.nn as nn
import torch.nn.functional as F
#from Torch.ntools import AlexNet
import math
import torchvision
import sage_model_config as config
import eyebackbone

def conv_bn(input_x, conv):
    x = conv(input_x)
    x = F.relu(x)
    # 在PyTorch中，BatchNorm层的命名是可选的，通常不需要手动命名
    bn = nn.BatchNorm2d(x.size(1))
    x = bn(x)
    return x

def relu_bn(input_x):
    x = F.relu(input_x)
    bn = nn.BatchNorm2d(x.size(1))
    x = bn(x)
    return x

## 开启混合精度

class grid_linear1(nn.Module):
    def __init__(self):
        super(grid_linear1, self).__init__()
        self.grid_linear1 = nn.Linear(256, 256) 
    
    def forward(self, x):
        x = self.grid_linear1(x)
        x = torch.relu(x)
        return x

class grid_linear2(nn.Module):
    def __init__(self):
        super(grid_linear2, self).__init__()
        self.grid_linear2 = nn.Linear(128, 128) 
    
    def forward(self, x):
        x = self.grid_linear2(x)
        x = torch.relu(x)
        return x

class grid_linear3(nn.Module):
    def __init__(self):
        super(grid_linear3, self).__init__()
        self.grid_linear3 = nn.Linear(128, 128) 
    
    def forward(self, x):
        x = self.grid_linear3(x)
        x = torch.relu(x)
        return x

class heatmap_conv1(nn.Module):
    def __init__(self):
        super(heatmap_conv1, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(7, 7), padding=3)  

    def forward(self, x):
        x = self.conv1(x)
        return x

class heatmap_conv2(nn.Module):
    def __init__(self):
        super(heatmap_conv2, self).__init__()
        self.heatmap_conv2 = nn.Conv2d(1, 1, kernel_size=(7, 7), padding=3)  

    def forward(self, x):
        x = self.heatmap_conv2(x)
        return x

class heatmap_conv3(nn.Module):
    def __init__(self):
        super(heatmap_conv3, self).__init__()
        self.heatmap_conv3 =  nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1) 

    def forward(self, x):
        x = torch.relu(self.heatmap_conv3(x))
        return x
    
    
    
    eyeIm_shape = (config.eyeIm_size,config.eyeIm_size,config.channel)

class SAGE_Imon(nn.Module):
    def __init__(self, base_model='MobileNetV2', heatmap=False):
        super(SAGE_Imon,self).__init__()
        base_model_list = ['AlexNet', 'MobileNetV2', 'EfficientNetB0', 'EfficientNetB1',
                       'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4']
        if (base_model not in base_model_list):
            print('base_model --' + base_model + '-- does not exist')
            print('the default model is MobileNetV2 ')

        if base_model == "AlexNet":
            self.backbone = eyebackbone.AlexBackbone()
        else:
            self.backbone = eyebackbone.Backbone(base_model)
        # nn.Linear输入维度不是64，64暂定
        self.landmark_net = nn.Sequential(
            nn.Linear( 6 + 3,64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        self.bn = nn.BatchNorm1d(128)
        if(heatmap):
            self.merge_linear = nn.Sequential(
                nn.Linear(128 + 128 + 16 + 3,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128,int(config.hm_size**2/4)),
                nn.ReLU(),
                nn.BatchNorm1d(int(config.hm_size**2/4))
            )
        else:
            self.merge_linear = nn.Sequential(
                nn.Linear(128 + 128 + 16 + 3,128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
            )
            self.merge_linear2 = nn.Sequential(
                nn.Linear(128 + 3,2)
            )


    def forward(self,imageleye,imagereye,eyelandmark,iorientation):
        # leye
        leye = self.backbone(imageleye)
        reye = self.backbone(imagereye)
        print(eyelandmark.shape)
        print(iorientation.shape)
        ''' iorientation [batchsize , 3] eyelandmark [batchsize , 6] '''
        landmark = torch.cat((iorientation, eyelandmark), dim=1)
        landmark = self.landmark_net(landmark)
        if(config.heatmap):
            # 3 + 16 + 128 + 128
            merge = torch.cat([iorientation, landmark,leye,reye], dim=1)
            merge = self.merge_linear(merge)
            heatmap = merge.view(-1, 1, int(config.hm_size/2), int(config.hm_size/2))
            hconv1 = heatmap_conv1()
            # print("heatmap before cv1 : " + str(heatmap.shape))
            heatmap = conv_bn(heatmap,hconv1)
            heatmap = F.interpolate(heatmap, scale_factor=2, mode='bilinear', align_corners=False)
            # 假设 heatmap_conv2 是一个 Conv2d 层
            hconv2 = heatmap_conv2()
            heatmap = hconv2(heatmap)
            # 使用 ReLU 激活函数
            heatmap = F.relu(heatmap)
            return heatmap

        else:
            merge = torch.cat([iorientation, landmark,leye,reye], dim=1)
            merge = self.merge_linear(merge)
            merge = torch.cat([iorientation,merge],dim=1)
            merge = self.merge_linear2(merge)
            return merge


if __name__ == '__main__':
    # m = SAGE_Imon("AlexNet",config.heatmap)
    m = SAGE_Imon("MobileNetV2",config.heatmap)
    feature = {"faceImg": torch.zeros(10, 3, 224, 224), "leftEyeImg": torch.zeros(10, 3, 112, 112),
               "rightEyeImg": torch.zeros(10, 3, 112, 112), "faceGridImg": torch.zeros(10, 12),
               "label": torch.zeros(10, 2), "frame": "test.jpg","eyelandmark":torch.ones(10,6),"iorientation":torch.ones(10,3)}
    # a = m(feature["leftEyeImg"], feature["rightEyeImg"], feature["eyelandmark"], feature["iorientation"])
    # print(a.shape)
    from thop import profile
    flops, params = profile(m,inputs=(feature["leftEyeImg"], feature["rightEyeImg"], feature["eyelandmark"], feature["iorientation"],))
    print(f'FLOPs: {flops}')




        



        

