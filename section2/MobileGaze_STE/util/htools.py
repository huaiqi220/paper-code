import numpy as np
from math import exp, sqrt, pi
import torch
from torch.functional import F
import config_bk
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import os

'''
计算高斯分布

'''
def gauss(x, stdv=0.5):
    return exp(-(1/(2*stdv))*(x**2))/(sqrt(2*pi*stdv))

def pog2heatmap(label):
    hmFocus_size = 17  # if (config.mobile) else 9  # in pixel unit

    HM_FOCUS_IM = np.zeros((5, hmFocus_size, hmFocus_size, 1))

    stdv_list = [0.2, 0.25, 0.3, 0.35, 0.4]
    for level in range(5):  # 5 levels of std to constuct heatmap
        stdv = stdv_list[level]  # 3/(12-level)
        for i in range(hmFocus_size):
            for j in range(hmFocus_size):
                distanceFromCenter = 2 * \
                    np.linalg.norm(np.array([i-int(hmFocus_size/2),
                                            j-int(hmFocus_size/2)]))/((hmFocus_size)/2)
                gauss_prob = gauss(distanceFromCenter, stdv)
                HM_FOCUS_IM[level, i, j, 0] = gauss_prob
    # HM_FOCUS_IM[level, :, :, 0] /= np.sum(HM_FOCUS_IM[level, :, :, 0])
    # heatmap_im = convert_image_dtype(HM_FOCUS_IM[0, :, :, :], tf.float32)
    # heatmap_im = pad_to_bounding_box(heatmap_im,
    #                                   int(label[0]*scale+hm_size/2-hmFocus_size/2),
    #                                   int(label[1]*scale+hm_size/2-hmFocus_size/2),
    #                                   hm_size, hm_size)
    heatmap_im = HM_FOCUS_IM[config_bk.hm_level, :, :, :].astype(np.float32)
    heatmap_im = heatmap_im.transpose(2, 0, 1)
    heatmap_im = torch.from_numpy(heatmap_im)

    # Calculate padding values
    # print(label)
    y_center = label[1] * config_bk.scale
    if y_center > 64:
        y_center = 64
    if y_center < -64:
        y_center = -64
    x_center = label[0] * config_bk.scale
    if x_center > 64:
        x_center = 64
    if x_center < -64:
        x_center = -64
    pad_top = int(y_center + config_bk.hm_size / 2 - hmFocus_size / 2)
    pad_left = int(x_center + config_bk.hm_size / 2 - hmFocus_size / 2)  
    # Since PyTorch's pad function takes padding as (left, right, top, bottom), calculate these values
    pad_bottom = config_bk.hm_size - (pad_top + heatmap_im.shape[1])
    pad_right = config_bk.hm_size - (pad_left + heatmap_im.shape[2])
    
    # Apply padding
    # print(pad_left, pad_right, pad_top, pad_bottom)
    heatmap_im = F.pad(heatmap_im, (pad_left, pad_right, pad_top, pad_bottom))

    return heatmap_im


def save_first_image(tensor, filename):
    # 选择第一个tensor，并移除channel维度 (1, 128, 128 -> 128, 128)
    dims = tensor.dim()
    if dims == 4:
    # 处理 N * C * H * W 的情况
        if tensor.shape[1] == 1:
            # 移除 channel 维度 (N, 1, H, W -> N, H, W)
            image_tensor = tensor[0, 0, :, :]
        else:
            raise ValueError("The second dimension must be 1 for grayscale images in N * C * H * W format.")
    elif dims == 3:
        # 处理 N * H * W 的情况
        image_tensor = tensor[0, :, :]
    else:
        raise ValueError("Tensor must be either N * C * H * W or N * H * W")
    # 如果tensor在GPU上，先将其移到CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # 转换为 PIL 图像
    transform = transforms.ToPILImage()
    image = transform(image_tensor)

    # 保存图像
    image.save(os.path.join("./images",filename))

# if __name__ == "__main__":
#     heatmap = pog2heatmap([-100,10])
#     heatmap = heatmap.squeeze().numpy()
#     plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
#     plt.colorbar()  # Add a colorbar to the side
#     plt.title('Heatmap Visualization')
#     plt.savefig('heatmap.png')