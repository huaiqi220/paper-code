import torch
import torch.nn as nn
from difNet import crossNet



class difGazeNet(nn.Module):
    def __init__(self):
        super(difGazeNet,self).__init__()

        self.eyeModel = crossNet.eyeGazeNet()
        self.flattern = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(9216, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

  
    def forward(self,eye_image1,eye_image2):
        e1 = self.eyeModel(eye_image1)
        e1 = self.flattern(e1)
        e2 = self.eyeModel(eye_image2)
        e2 = self.flattern(e2)
        x = torch.cat((e1, e2), 1)
        x = self.linear(x)
        return x




if __name__ == '__main__':
    m = difGazeNet()
    feature = { "left":torch.ones(10,3,48,72),"right":torch.ones(10,3,48,72)}
    a = m(feature["left"],feature["right"])
    print(a.shape)
    # feature = {"left": torch.zeros(10,1, 36,60),
    #             "right": torch.zeros(10,1, 36,60)
    #             }
    # feature = {"faceImg": torch.zeros(10, 3, 224, 224), "leftEyeImg": torch.zeros(10, 3, 112, 112),
    #            "rightEyeImg": torch.zeros(10, 3, 112, 112), "faceGridImg": torch.zeros(10, 12),
    #            "label": torch.zeros(10, 2), "frame": "test.jpg"}
    # a = m(feature["leftEyeImg"], feature["rightEyeImg"], feature["faceImg"], feature["faceGridImg"])
    # print(a.shape)
