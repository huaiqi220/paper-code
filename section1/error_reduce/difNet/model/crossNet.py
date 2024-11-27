import torch
import torch.nn as nn


'''
这是眼部图像的CNN特征提取器
'''
class eyeGazeNet(nn.Module):
    def __init__(self):
        super(eyeGazeNet,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )



    def forward(self,eye_image):
        x = self.conv1(eye_image)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


'''
这是原始版本3D的单帧预测
输出2是yaw和pitch
'''
class SingleNN(nn.Module):
    def __init__(self):
        super(SingleNN,self).__init__()
        self.eyeModel = eyeGazeNet()

        self.flattern = nn.Sequential(
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(4608, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self,x):
        x = self.eyeModel(x)
        x = self.flattern(x)
        x = self.linear(x)
        return x

'''
这是SingleNN方法PoG原始版本，输入左右眼
输出2D PoG
'''
class SingleNNPoG(nn.Module):
    def __init__(self):
        super(SingleNNPoG,self).__init__()
        self.eyeModel = eyeGazeNet()

        self.flattern = nn.Sequential(
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(4608 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
    '''
    输入之前记得翻转right
    '''
    def forward(self,left,right):
        left = self.eyeModel(left)
        right = self.eyeModel(right)
        left = self.flattern(left)
        right = self.flattern(right)
        f = torch.cat((left,right),dim=1)
        gaze = self.linear(f)
        return gaze

'''
这是DifNN方法PoG原始版本，输入frame1 frame2
默认都是以 2 减去 1 作为label
右眼输入之前都必须翻转
输出2D PoG
'''
class DifNNPoG(nn.Module):
    def __init__(self):
        super(DifNNPoG,self).__init__()
        self.eyeModel = eyeGazeNet()

        self.flattern = nn.Sequential(
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(4608 * 2, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
        )

        self.dl = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64,2),
        )

    '''
    输出右眼必须翻转
    '''
    def forward(self,frame1,frame2):
        left1 = frame1[0]
        right1 = frame1[1]
        left1 = self.eyeModel(left1)
        right1 = self.eyeModel(right1)
        left1 = self.flattern(left1)
        right1 = self.flattern(right1)
        f1 = torch.cat((left1,right1),dim=1)
        fl1 = self.linear(f1)

        left2 = frame2[0]
        right2 = frame2[1]
        left2 = self.eyeModel(left2)
        right2 = self.eyeModel(right2)
        left2 = self.flattern(left2)
        right2 = self.flattern(right2)
        f2 = torch.cat((left2,right2),dim=1)
        fl2 = self.linear(f2)

        f = torch.cat((fl1,fl2),dim=1)
        gaze = self.dl(f)
        return gaze


'''
这是原始版本3D的差分
输出2是yaw和pitch差值
'''
class DifNN(nn.Module):
    def __init__(self):
        super(DifNN, self).__init__()

        self.eyeModel = eyeGazeNet()
        self.flattern = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, eye_image1, eye_image2):
        e1 = self.eyeModel(eye_image1)
        e1 = self.flattern(e1)
        e2 = self.eyeModel(eye_image2)
        e2 = self.flattern(e2)
        x = torch.cat((e1, e2), 1)
        x = self.linear(x)
        return x




if __name__ == '__main__':
    left1 =  torch.ones(10,3,48,72)
    left2 = torch.ones(10, 3, 48, 72)
    right1 = torch.ones(10,3,48,72)
    right2 = torch.ones(10,3,48,72)

    s = SingleNNPoG()

    d = DifNNPoG()

    print(s(left1,right1))
    print(d([left1,right1],[left2,right2]))
