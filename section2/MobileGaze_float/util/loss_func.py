import torch
import torch.nn as nn







def heatmap_loss(y_true, y_pred):
    if torch.sum(y_pred) == 0:
        return torch.tensor(1.0)
    else:
        # y_pred /= torch.sum(y_pred)
        # y_true /= torch.sum(y_true)
        # return torch.sum(torch.abs(y_pred - y_true)) / 2.0
        normalized_y_pred = y_pred / torch.sum(y_pred)
        normalized_y_true = y_true / torch.sum(y_true)
        """修改一下输出,加个小偏置,避免heatmap训练时候出现的divided by zero"""
        return (1e-8 + torch.sum(torch.abs(normalized_y_pred - normalized_y_true))) / 2.0
    
class WeightedL1Loss(nn.Module):
    def __init__(self, weight_factor=10.0):
        super(WeightedL1Loss, self).__init__()
        self.l1 = nn.L1Loss(reduction='sum')  # Use 'none' to get loss per element
        self.weight_factor = weight_factor

    def forward(self, output, target):
        # Compute the base L1 loss
        loss = self.l1(output, target)
        
        # Create a mask where the target is non-zero
        weight_mask = torch.where(target != 0, self.weight_factor, 1.0)
        
        # Apply the weights to the loss
        weighted_loss = loss * weight_mask
        
        # Return the mean of the weighted loss
        return weighted_loss.mean()
    
