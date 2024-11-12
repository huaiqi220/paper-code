import torch
import torch.nn as nn
import torch.nn.functional as F
#from Torch.ntools import AlexNet
import math
import torchvision


import torch
import torch.nn as nn
import torch.nn.functional as F

class SAGE(nn.Module):
    def __init__(self):
        super(SAGE, self).__init__()
        
        # Define convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        
        # Define fully connected layers
        self.fc1 = nn.Linear(8, 100)  # assuming input image of size 64x64
        self.fc2 = nn.Linear(100, 16)
        self.fc3 = nn.Linear(16, 16)
        self.fc4 = nn.Linear(16 + 64 * 8 * 8, 16)
        self.fc5 = nn.Linear(16, 2)
        
    def forward(self, left_eye, right_eye, landmark_features):
        # Apply shared convolutional layers to both eyes
        left_eye = torch.flip(left_eye, [2])
        left_eye = self.preprocess_eye(left_eye)
        right_eye = self.preprocess_eye(right_eye)
        
        # Concatenate left and right eye features
        combined_features = torch.cat((left_eye, right_eye), dim=1)
        print(combined_features.shape)
        # Flatten for fully connected layers
        x = combined_features.view(combined_features.size(0), -1)
        # print(x.shape)
        landmark_features = F.relu(self.fc1(landmark_features))
        landmark_features = F.relu(self.fc2(landmark_features))
        landmark_features = F.relu(self.fc3(landmark_features))
        # Apply fully connected layers
        x  = torch.cat((landmark_features, x), dim=1)
        x = F.relu(self.fc4(x))
        gaze_output = self.fc5(x)

        # Apply device-specific parameters (affine transformation)
        # gaze_output = self.apply_device_params(gaze_output, device_params)
        
        return gaze_output
    
    def preprocess_eye(self, eye):
        x = F.relu(self.conv1(eye))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x
    
    # def apply_device_params(self, gaze_output, device_params):
    #     # Dummy implementation for applying affine transformation using device parameters
    #     # Replace with proper transformation logic
    #     return gaze_output

# Test the model with dummy data
if __name__ == "__main__":
    model = SAGE()
    left_eye = torch.randn(1, 3, 64, 64)  # (batch_size, channels, height, width)
    right_eye = torch.randn(1, 3, 64, 64)
    landmark_features = torch.randn(1, 8)  # Example: 8-dimensional landmark features
    device_params = torch.randn(1, 6)  # Example: device-specific parameters waov (2) and wloc (4)

    output = model(left_eye, right_eye, landmark_features)
    print("Output gaze coordinates:", output)



