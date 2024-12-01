import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGE_Backbone(nn.Module):
    def __init__(self):
        super(SAGE_Backbone, self).__init__() 
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
        )

        self.landmark_fc = nn.Sequential(
            nn.Linear(8, 100),
            nn.ReLU(),
            nn.Linear(100, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(16 + 32 * 2 * 2 * 2, 16),
            nn.ReLU(),
        )
        
    def forward(self, left_eye, right_eye, landmark):
        left_eye = torch.flip(left_eye, [3])
        fleye = self.conv(left_eye)
        freye = self.conv(right_eye)
        fl = self.landmark_fc(landmark)
        fleye = torch.flatten(fleye, 1)
        freye = torch.flatten(freye, 1)
        f = torch.cat((fleye, freye, fl), dim=1)
        f = self.fc2(f)
        return f


class DirectionClassifier(nn.Module):
    def __init__(self, input_size):
        super(DirectionClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),  # 第一层全连接
            nn.ReLU(),
            nn.Linear(64, 32),  # 第二层全连接
            nn.ReLU(),
            nn.Linear(32, 4)   # 输出四个方向
        )
        self.softmax = nn.Softmax(dim=1)  # 对输出的四个方向进行归一化

    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x


class FinalRegression(nn.Module):
    def __init__(self, input_size):
        super(FinalRegression, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  
        )
        
    def forward(self, x):
        return self.fc(x)


class SAGE_SFO(nn.Module):
    def __init__(self, k=9):
        super(SAGE_SFO, self).__init__()
        self.backbone = SAGE_Backbone()
        self.k = k
        self.direction_classifier = DirectionClassifier(input_size=16 * 2)  # Features from Xq and Xc
        self.final_regression = FinalRegression(input_size=16 + 16 * k + 2 * k + k * 4)

    def forward(self, X_q, X_cY_c):
        Xq_leye = X_q['left_eye']
        Xq_reye = X_q['right_eye']
        Xq_landmark = X_q['landmark']
        Xqf = self.backbone(Xq_leye, Xq_reye, Xq_landmark)
        
        calibration_features = []
        direction_outputs = []
        for i in range(self.k):
            Xc_leye = X_cY_c[i]['left_eye']
            Xc_reye = X_cY_c[i]['right_eye']
            Xc_landmark = X_cY_c[i]['landmark']
            Yc = X_cY_c[i]['gaze']
            
            Xcf = self.backbone(Xc_leye, Xc_reye, Xc_landmark)
            Xcfy = torch.cat((Xcf, Yc), dim=1)
            calibration_features.append(Xcfy)
            
            combined_features = torch.cat((Xqf, Xcf), dim=1)
            direction_output = self.direction_classifier(combined_features)
            direction_outputs.append(direction_output)
        
        calibration_features = torch.stack(calibration_features, dim=1)  # Shape: [batch_size, k, feature_dim]
        direction_outputs = torch.stack(direction_outputs, dim=1)  # Shape: [batch_size, k, 4]
        
        combined_features = torch.cat((Xqf, calibration_features.view(Xqf.size(0), -1), direction_outputs.view(Xqf.size(0), -1)), dim=1)
        print(combined_features.shape)
        print(calibration_features.view(Xqf.size(0), -1).shape)
        print(direction_outputs.view(Xqf.size(0), -1).shape)
        gaze_prediction = self.final_regression(combined_features)
        return gaze_prediction, direction_outputs
    

def sage_sfo_loss(gaze, direction, lgaze, ldirection, calibration_data, lambda_weight=1.0):
    """
    实现包含 gaze 和 direction 的损失函数。

    参数：
    - gaze: 模型预测的 gaze，形状为 [batch_size, 2]。
    - direction: 模型预测的 direction 概率分布，形状为 [batch_size, K, 4]。
    - lgaze: 查询样本的真实 gaze 标签，形状为 [batch_size, 2]。
    - ldirection: 真实的相对方向标签，形状为 [batch_size, K]，取值范围为 0, 1, 2, 3。
    - calibration_data: 校准数据 gaze 标签，形状为 [batch_size, K, 2]。
    - lambda_weight: 权重参数，平衡 gaze 和 direction 损失。

    返回：
    - total_loss: 总损失值。
    """
    # 计算 gaze 损失
    gaze_diff = gaze - lgaze
    gaze_loss = torch.mean(torch.norm(gaze_diff, dim=1) ** 2)  # L2 损失

    # 计算方向分类损失（交叉熵）
    K = calibration_data.size(1)
    batch_size = gaze.size(0)
    
    # Reshape ldirection to match the prediction dimension
    ldirection = ldirection.view(-1)  # [batch_size * K]
    direction = direction.view(batch_size * K, -1)  # [batch_size * K, 4]

    direction_loss = F.cross_entropy(direction, ldirection)

    # 总损失
    total_loss = gaze_loss + lambda_weight * direction_loss
    return total_loss




if __name__ == "__main__":
    # 创建输入数据
    batch_size = 2  # 示例 batch 大小
    image_size = (64, 64)  # 图像尺寸
    landmark_dim = 8  # landmark 维度
    k = 9  # 校准样本数量

    # 构造 Query 样本 X_q
    X_q = {
        "left_eye": torch.randn(batch_size, 3, *image_size),  # 左眼图像
        "right_eye": torch.randn(batch_size, 3, *image_size),  # 右眼图像
        "landmark": torch.randn(batch_size, landmark_dim)  # landmark
    }

    # 构造校准样本 X_cY_c
    X_cY_c = []
    for i in range(k):
        X_cY_c.append({
            "left_eye": torch.randn(batch_size, 3, *image_size),  # 左眼图像
            "right_eye": torch.randn(batch_size, 3, *image_size),  # 右眼图像
            "landmark": torch.randn(batch_size, landmark_dim),  # landmark
            "gaze": torch.randn(batch_size, 2)  # gaze 坐标
        })

    # 初始化模型
    model = SAGE_SFO(k=k)

    # 执行一次前向传播
    gaze_prediction, direction_outputs = model(X_q, X_cY_c)

    # 打印输出
    print("Gaze Prediction Shape:", gaze_prediction.shape)  # [batch_size, 2]
    print("Direction Outputs Shape:", direction_outputs.shape)  # [batch_size, k, 4]

    lgaze = torch.randn(batch_size, 2)  # [batch_size, 2]
    ldirection = torch.randint(0, 4, (batch_size, k)) # [batch_size, K]

    # 模拟校准数据
    calibration_data = torch.randn(batch_size, k, 2)  # [batch_size, K, 2]

    # 计算损失
    loss = sage_sfo_loss(gaze_prediction, direction_outputs, lgaze, ldirection, calibration_data, lambda_weight=0.5)
    print("Loss:", loss.item())