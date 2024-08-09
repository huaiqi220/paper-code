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
        self.features = nn.Sequential(
            *list(resnet18.children())[:-1],
            nn.Flatten()
                                      )
    
    def forward(self, x):
        x = self.features(x)
        return x
    

class mobile_gaze(nn.Module):
    def __init__(self,hm_size,cali_size,grid_size):
        
        super(mobile_gaze,self).__init__()
        self.heat_map_shape = hm_size
        self.cali_shape = cali_size
        '''grid size = grid height * grid width'''
        self.grid_size = grid_size
        self.face_cnn = ResNet18Conv()
        self.eye_cnn = ResNet18Conv()
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
            nn.Linear(1024,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            # nn.Linear(3072, self.heat_map_shape * self.heat_map_shape)
            nn.Linear(1024, self.heat_map_shape * self.heat_map_shape)
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
        gaze_heatmap = torch.reshape(gaze_heatmap,(-1,self.heat_map_shape,self.heat_map_shape))
        return c, gaze_heatmap



if __name__ == "__main__":
    feature = {"face": torch.randn(10,3,112,112),"left":torch.randn(10,3,224,224),"right":torch.randn(10,3,224,224),"grid":torch.randn(10,1,25,40),"cali":torch.randn(10,8)}
    mobile = mobile_gaze(32,8,25*40)
    c, hm = mobile(feature["face"],feature["left"],feature["right"],feature["grid"],feature["cali"])
    print(c)
    print(hm)
    print(c.shape)
    print(hm.shape)