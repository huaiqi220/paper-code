import torch


def heatmap_loss(y_true, y_pred):
    if torch.sum(y_pred) == 0:
        return torch.tensor(1.0)
    else:
        # y_pred /= torch.sum(y_pred)
        # y_true /= torch.sum(y_true)
        # return torch.sum(torch.abs(y_pred - y_true)) / 2.0
        normalized_y_pred = y_pred / torch.sum(y_pred)
        normalized_y_true = y_true / torch.sum(y_true)
        return torch.sum(torch.abs(normalized_y_pred - normalized_y_true)) / 2.0