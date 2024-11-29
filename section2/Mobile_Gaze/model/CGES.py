import torch
import torch.nn as nn
import torchvision.models as models
from model import STE
import config
import torch.nn.functional as F



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
        self.cali_shape = config.k
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
            # 这里要配合下面FC2选择来改
            nn.Linear(1024, 512)
        )

        '''小FC2'''
        # self.fc2_small = nn.Sequential(
        #     nn.Linear(256 + self.cali_shape, 128),
        #     # nn.Linear(256, 128),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(128, 64),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(64, 2)
        # )

        '''原始版本FC2'''
        self.fc2_origin_version = nn.Sequential(
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
        '''
            这是校准向量全连接成256维度的fcc
            但目前分析cat与+模式效果差不多，因此
            目前cat模式用不到
        '''
        # self.fcc = nn.Sequential(
        #     nn.Linear(self.cali_shape,256),
        #     nn.BatchNorm1d(256),
        #     nn.ReLU(),
        #     nn.Dropout(),
        # )



        self.cali_vectors = nn.Parameter(torch.randn(self.num_users, self.cali_shape))

    def getTheGazeFeature(self, face, left, right, grid, user_id,mode,cali_vec=None):
        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left)
        right = torch.flip(right, [3])
        right_feature = self.eye_cnn(right)
        grid_feature = self.grid_linear(grid)

        fc1_input = torch.cat((face_feature, left_feature, right_feature, grid_feature), dim=1)
        fc1_output = self.fc1(fc1_input)
        return fc1_output

    def computeGaze(self,cali_forward,fc1_output):
        # fc2_input = torch.cat((cali_forward, fc1_output), dim=1)
        # gaze_output = self.fc2(fc2_input)
        # return gaze_output
        # 将校准向量与 fc1 输出连接起来
        # print(cali_forward.shape)
        # cali_forward = self.fcc(cali_forward)

        fc2_input = torch.cat((cali_forward, fc1_output), dim=1)
        # fc2_input = fc1_output + cali_forward
        gaze_output = self.fc2_origin_version(fc2_input)
        return gaze_output

    def forward(self, face, left, right, grid, user_id,mode,cali_vec=None):
        # 检查 user_id 的最大值是否在允许范围内
        if mode == "train":
            assert user_id.max() < self.num_users, "user_id exceeds the number of users available"

        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left)
        right = torch.flip(right, [3])
        right_feature = self.eye_cnn(right)
        grid_feature = self.grid_linear(grid)

        fc1_input = torch.cat((face_feature, left_feature, right_feature, grid_feature), dim=1)
        fc1_output = self.fc1(fc1_input)

        # 使用 STE 方法对校准向量进行离散化
        if mode == "train":
            cali_forward = STE.BinarizeSTE_origin.apply(self.cali_vectors[user_id])
        elif mode == "inference":
            # 这里的写法，推理一个batchsize数据必须来自同一个人
            cali_vec = cali_vec.expand(face.shape[0], -1)
            cali_forward = cali_vec
        else:
            raise ValueError("mode should be either 'train' or 'inference'")

        # 将校准向量与 fc1 输出连接起来
        # cali_forward = self.fcc(cali_forward)

        fc2_input = torch.cat((cali_forward, fc1_output), dim=1)
        # fc2_input = fc1_output + cali_forward
        gaze_output = self.fc2_origin_version(fc2_input)

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
