import torch
import torch.nn as nn
import torchvision.models as models

import torch.functional as F


# 这是最终去了Linear的CNN，输出维度512
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
    
class SameModel(nn.Module):
    def __init__(self):
        super(SameModel, self).__init__()
        self.face_cnn = ResNet18Conv()
        self.eye_cnn = ResNet18Conv()
        self.grid_cnn = ResNet18Conv()
        self.fc_combined = nn.Sequential(
            nn.Linear(512 * 4, 3072),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(3072,3072),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(3072,8 + 512)
        )
        self.fc_output = nn.Sequential(
            nn.Linear(512 + 8, 3072),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3072,3072),
            nn.BatchNorm1d(),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(3072, 32 * 32)            
        )
    
    def forward(self, face, left, right, grid, cali):
        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left)
        right_feature = self.eye_cnn(right)
        grid_feature = self.grid_cnn(grid)
        cat_feature = torch.cat((face_feature,left_feature,right_feature,grid_feature),dim=1)
        fc1_out = self.fc_combined(cat_feature)
        c, fc2_input = torch.split(fc1_out, [8, 512], dim=1)
        fc2_input = torch.cat((fc2_input,cali),dim=1)
        gaze_out_heatmap = self.fc_output(fc2_input)

        return c, gaze_out_heatmap


if __name__ == "__main__":
    feature = torch.ones(10,3,224,224)
    res = ResNet18Conv()
    print(res)
    print(res(feature).shape)
