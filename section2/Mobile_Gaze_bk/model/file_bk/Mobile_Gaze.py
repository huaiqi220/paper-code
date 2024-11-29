import torch
import torch.nn as nn
import torchvision.models as models

import config_bk
import torch.nn.functional as F

class ResNet18Conv(nn.Module):
    def __init__(self):
        super(ResNet18Conv, self).__init__()
        # Load the pre-trained ResNet-18 model
        resnet18 = models.resnet18(pretrained=True)
        # Extract the convolutional part (up to the last conv layer)
        self.features = nn.Sequential(
            *list(resnet18.children())[:-1],
            nn.Flatten()
                                      )
    
    def forward(self, x):
        x = self.features(x)
        return x

def getMobileNetV2CNN():
    model = models.mobilenet_v2(pretrained=True)

    # 删除全连接层
    model.classifier = nn.Linear(1280, 512)
    return model


"""

这是mobile_gaze的2D坐标版本，输出的是2D坐标
"""
class mobile_gaze_2d(nn.Module):
    def __init__(self,hm_size,cali_size,grid_size):
        
        super(mobile_gaze_2d,self).__init__()
        self.heat_map_shape = hm_size
        self.cali_shape = cali_size
        '''grid size = grid height * grid width'''
        self.grid_size = grid_size
        self.face_cnn = getMobileNetV2CNN()
        self.eye_cnn = getMobileNetV2CNN()
        self.grid_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128,64)
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(512 * 3 + 64, 3072),
            nn.Linear(512 * 3 + 64, 1024),
            # nn.BatchNorm1d(3072),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072,3072),
            # nn.BatchNorm1d(3072),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072,512 + self.cali_shape)
            nn.Linear(1024,512 + self.cali_shape)
        )
        self.fc2 = nn.Sequential(
            # nn.Linear(512 + self.cali_shape, 3072),
            nn.Linear(512 + self.cali_shape, 1024),
            # nn.BatchNorm1d(3072),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072,3072),
            # nn.BatchNorm1d(3072),
            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072, self.heat_map_shape * self.heat_map_shape)
            nn.Linear(256, 2)
        )
    
    def forward(self, face, left, right, grid, cali):
        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left)
        right_feature = self.eye_cnn(right)
        grid_feature = self.grid_linear(grid)
        fc1_input = torch.cat((face_feature,left_feature,right_feature,grid_feature),dim=1)
        fc1_output = self.fc1(fc1_input)
        c, fc1_output = torch.split(fc1_output, [self.cali_shape, 512], dim=1)
        fc2_input = torch.cat((cali,fc1_output),dim=1)
        gaze_heatmap = self.fc2(fc2_input)
        # gaze_heatmap = torch.reshape(gaze_heatmap,(-1,self.heat_map_shape,self.heat_map_shape))
        return c, gaze_heatmap



"""
这是Mobiel-gaze的heatmap版本，输出的是heatmap
"""
class mobile_gaze_hm(nn.Module):
    def __init__(self,hm_size,cali_size,grid_size):
        
        super(mobile_gaze_hm,self).__init__()
        self.heat_map_shape = hm_size
        self.cali_shape = cali_size
        '''grid size = grid height * grid width'''
        self.grid_size = grid_size
        self.face_cnn = getMobileNetV2CNN()
        self.eye_cnn = getMobileNetV2CNN()
        self.grid_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(grid_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128,64)
        )
        self.fc1 = nn.Sequential(
            # nn.Linear(512 * 3 + 64, 3072),
            nn.Linear(512 * 3 + 64, 1024),
            # nn.BatchNorm1d(3072),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072,3072),
            # nn.BatchNorm1d(3072),
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072,512 + self.cali_shape)
            nn.Linear(1024,512 + self.cali_shape)
        )
        self.fc2_hm_l1 = nn.Sequential(
            # nn.Linear(512 + self.cali_shape, 3072),
            nn.Linear(512 + self.cali_shape, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,int(config_bk.hm_size**2/4)),
            nn.ReLU(),
            nn.BatchNorm1d(int(config_bk.hm_size**2/4))
        )
        self.fc2_hm_conv1 = nn.Conv2d(1, 1, kernel_size=(7, 7), padding=3) 
        self.fc2_hm_conv2 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 7), padding=3),
            nn.ReLU()
        )
    
    def forward(self, face, left, right, grid, cali):
        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left)
        right_feature = self.eye_cnn(right)
        grid_feature = self.grid_linear(grid)
        fc1_input = torch.cat((face_feature, left_feature, right_feature, grid_feature), dim=1)
        fc1_output = self.fc1(fc1_input)
        c, fc1_output = torch.split(fc1_output, [self.cali_shape, 512], dim=1)
        fc2_input = torch.cat((cali, fc1_output), dim=1)
        gaze_heatmap = self.fc2_hm_l1(fc2_input)
        gaze_heatmap = gaze_heatmap.view(-1,1,int(config_bk.hm_size/2), int(config_bk.hm_size/2))
        gaze_heatmap = self.fc2_hm_conv1(gaze_heatmap)
        gaze_heatmap = F.interpolate(gaze_heatmap,scale_factor=2,mode='bilinear',align_corners=False)
        gaze_heatmap = self.fc2_hm_conv2(gaze_heatmap)
        return c, gaze_heatmap 



if __name__ == "__main__":
    feature = {"face": torch.randn(10,3,112,112),"left":torch.randn(10,3,224,224),"right":torch.randn(10,3,224,224),"grid":torch.randn(10,1,25,40),"cali":torch.randn(10,8)}
    mobile = mobile_gaze_2d(32,8,25*40)
    c, hm = mobile(feature["face"],feature["left"],feature["right"],feature["grid"],feature["cali"])
    print(c)
    print(hm)
    print(c.shape)
    print(hm.shape)