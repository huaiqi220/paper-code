import numpy as np
from math import exp, sqrt, pi
import torch
from torch.functional import F
import config
import matplotlib.pyplot as plt



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
    heatmap_im = HM_FOCUS_IM[config.hm_level, :, :, :].astype(np.float32)
    heatmap_im = heatmap_im.transpose(2, 0, 1)
    heatmap_im = torch.from_numpy(heatmap_im)

    # Calculate padding values
    # print(label)
    y_center = label[1] * config.scale
    if y_center > 64:
        y_center = 64
    if y_center < -64:
        y_center = -64
    x_center = label[0] * config.scale
    if x_center > 64:
        x_center = 64
    if x_center < -64:
        x_center = -64
    pad_top = int(y_center + config.hm_size / 2 - hmFocus_size / 2)
    pad_left = int(x_center + config.hm_size / 2 - hmFocus_size / 2)  
    # Since PyTorch's pad function takes padding as (left, right, top, bottom), calculate these values
    pad_bottom = config.hm_size - (pad_top + heatmap_im.shape[1])
    pad_right = config.hm_size - (pad_left + heatmap_im.shape[2])
    
    # Apply padding
    # print(pad_left, pad_right, pad_top, pad_bottom)
    heatmap_im = F.pad(heatmap_im, (pad_left, pad_right, pad_top, pad_bottom))

    return heatmap_im



def pog2heatmap2(label):
    hmFocus_size = 17  # Focus size for heatmap

    # Create a Gaussian heatmap template
    stdv_list = [0.2, 0.25, 0.3, 0.35, 0.4]
    HM_FOCUS_IM = np.zeros((5, hmFocus_size, hmFocus_size))
    for level, stdv in enumerate(stdv_list):
        for i in range(hmFocus_size):
            for j in range(hmFocus_size):
                distanceFromCenter = 2 * np.linalg.norm(np.array([i, j]) - np.array([hmFocus_size/2, hmFocus_size/2])) / (hmFocus_size / 2)
                HM_FOCUS_IM[level, i, j] = np.exp(-0.5 * (distanceFromCenter / stdv) ** 2)
        HM_FOCUS_IM[level] /= np.sum(HM_FOCUS_IM[level])  # Normalize

    # Use the first level of heatmap as the base
    heatmap_im = HM_FOCUS_IM[0].astype(np.float32)
    heatmap_im = torch.from_numpy(heatmap_im).unsqueeze(0)  # Shape (1, 17, 17)

    # Compute the center position
    center_x = label[0] * config.scale + config.hm_size // 2
    center_y = label[1] * config.scale + config.hm_size // 2

    # Define the bounding box to place the heatmap
    top_left_x = int(max(center_x - hmFocus_size // 2, 0))
    top_left_y = int(max(center_y - hmFocus_size // 2, 0))
    bottom_right_x = int(min(center_x + hmFocus_size // 2, config.hm_size))
    bottom_right_y = int(min(center_y + hmFocus_size // 2, config.hm_size))

    # Pad the heatmap to fit within the desired bounding box
    padding = (
        top_left_x, max(config.hm_size - bottom_right_x, 0),
        top_left_y, max(config.hm_size - bottom_right_y, 0)
    )

    heatmap_im = F.pad(heatmap_im, padding, mode='constant', value=0)

    # Crop or resize to 128x128 if necessary
    if heatmap_im.shape[1] > 128 or heatmap_im.shape[2] > 128:
        heatmap_im = F.interpolate(heatmap_im.unsqueeze(0), size=(128, 128), mode='bilinear', align_corners=False).squeeze(0)

    return heatmap_im


if __name__ == "__main__":
    heatmap = pog2heatmap([-100,10])
    heatmap = heatmap.squeeze().numpy()
    plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
    plt.colorbar()  # Add a colorbar to the side
    plt.title('Heatmap Visualization')
    plt.savefig('heatmap.png')