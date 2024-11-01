import torch
import torch.nn as nn
import torchvision.models as models

import config
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
    model.classifier = nn.Sequential(
        nn.Linear(1280, 512),
        nn.ReLU()
    )
    return model


"""
这是mobile_gaze的2D坐标版本，输出的是2D坐标
"""
class mobile_gaze_2d(nn.Module):
    def __init__(self, hm_size, cali_size, grid_size):
        super(mobile_gaze_2d, self).__init__()
        self.heat_map_shape = hm_size
        self.cali_shape = cali_size
        self.num_users = 4096
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
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(512 * 3 + 64, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512 + self.cali_shape, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 2)
        )

        self.cali_vectors = nn.Parameter(torch.randn(self.num_users, cali_size))

    def forward(self, face, left, right, grid, user_id):
        # 检查 user_id 的最大值是否在允许范围内
        assert user_id.max() < self.num_users, "user_id exceeds the number of users available"

        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left)
        right_feature = self.eye_cnn(right)
        grid_feature = self.grid_linear(grid)

        fc1_input = torch.cat((face_feature, left_feature, right_feature, grid_feature), dim=1)
        fc1_output = self.fc1(fc1_input)

        # 从 cali_vectors 中索引得到对应的校准向量
        cali_forward = (torch.sigmoid(self.cali_vectors[user_id]) > 0.5).float()

        # 将校准向量与 fc1 输出连接起来
        fc2_input = torch.cat((cali_forward, fc1_output), dim=1)
        gaze_output = self.fc2(fc2_input)

        return gaze_output


if __name__ == "__main__":
    feature = {
        "face": torch.randn(10, 3, 112, 112),
        "left": torch.randn(10, 3, 224, 224),
        "right": torch.randn(10, 3, 224, 224),
        "grid": torch.randn(10, 1, 25, 40),
    }
    user_ids = torch.randint(0, 4096, (10,))  # 随机生成 user_id 的索引，范围为 [0, 4096)
    mobile = mobile_gaze_2d(32, 12, 25 * 40)
    gaze_output = mobile(feature["face"], feature["left"], feature["right"], feature["grid"], user_ids)
    print(gaze_output)
    print(gaze_output.shape)
