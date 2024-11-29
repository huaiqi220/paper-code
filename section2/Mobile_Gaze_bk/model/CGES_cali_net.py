class MobileGaze2DWithCalibrationNet(nn.Module):
    def __init__(self, hm_size, cali_size, grid_size):
        super(MobileGaze2DWithCalibrationNet, self).__init__()
        self.heat_map_shape = hm_size
        self.cali_shape = cali_size
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
            nn.Linear(512 + cali_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(256, 2)
        )
        self.calibration_net = CalibrationNetwork(cali_size)

    def forward(self, face, left_eye, right_eye, grid, cali_sample=None, cali_label=None, mode='train'):
        face_feature = self.face_cnn(face)
        left_feature = self.eye_cnn(left_eye)
        right_feature = self.eye_cnn(right_eye)
        grid_feature = self.grid_linear(grid)
        
        fc1_input = torch.cat((face_feature, left_feature, right_feature, grid_feature), dim=1)
        fc1_output = self.fc1(fc1_input)

        if mode == 'train':
            # 使用校准样本通过校准网络生成校准向量
            cali_forward = self.calibration_net(cali_sample, cali_label)
        elif mode == 'inference':
            # 推理阶段，直接输入校准样本并生成校准向量
            cali_forward = self.calibration_net(cali_sample, cali_label)
        else:
            raise ValueError("mode should be either 'train' or 'inference'")
        
        # 将校准向量与主特征拼接
        fc2_input = torch.cat((fc1_output, cali_forward), dim=1)
        gaze_output = self.fc2(fc2_input)

        return gaze_output


import torch
import torch.nn as nn
import torch.nn.functional as F


class CalibrationNetwork(nn.Module):
    def __init__(self, cali_size):
        super(CalibrationNetwork, self).__init__()
        self.cali_size = cali_size
        
        # 校准网络可以是一个简单的 MLP
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224 + 2, 512),  # 输入为眼部图像 (3x224x224) 和 2D 注视点
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, cali_size)  # 输出为校准向量
        )

    def forward(self, eye_image, gaze_point):
        # eye_image: (B, 3, 224, 224), gaze_point: (B, 2)
        # 将图像展平并与 gaze_point 连接
        eye_flat = eye_image.view(eye_image.size(0), -1)
        input_features = torch.cat([eye_flat, gaze_point], dim=1)
        cali_vector = self.network(input_features)
        return cali_vector


def combined_loss(gaze_output, gaze_label, cali_vector, cali_ground_truth, lambda_cali=0.1):
    # 主任务的损失：预测 gaze 的损失
    gaze_loss = F.mse_loss(gaze_output, gaze_label)
    
    # 校准向量的损失：鼓励生成的校准向量与 ground truth 靠近（如果有先验）
    cali_loss = F.mse_loss(cali_vector, cali_ground_truth)
    
    # 综合损失
    total_loss = gaze_loss + lambda_cali * cali_loss
    return total_loss

def infer(model, face, left_eye, right_eye, grid, cali_sample, cali_label):
    model.eval()
    with torch.no_grad():
        gaze_output = model(face, left_eye, right_eye, grid, cali_sample, cali_label, mode='inference')
    return gaze_output

