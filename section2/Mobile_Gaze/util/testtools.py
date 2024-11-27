import numpy as np
import torch

def select_by_quadrants_gc(lines, k):
    """
    从预测空间中选择上下左右四个方向的 k 个样本。
    
    Args:
        lines (list): 包含所有数据行的列表。
        k (int): 每个方向选取的样本数。
    
    Returns:
        tuple: selected_cali_lines, remaining_lines
    """
    # 提取每行的 label
    labels = []
    line_data = []
    for line in lines:
        line = line.strip().split(" ")
        point = line[4]
        label = np.array(point.split(",")).astype("float")
        labels.append(label)
        line_data.append(line)
    
    labels = np.array(labels)  # 转换为 numpy 数组
    x_coords, y_coords = labels[:, 0], labels[:, 1]
    
    # 计算中心点，用于分类上下左右
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)
    
    # 分别找到上下左右的点索引
    up_indices = np.argsort(y_coords[y_coords < y_center])[-k:]
    down_indices = np.argsort(y_coords[y_coords >= y_center])[:k]
    left_indices = np.argsort(x_coords[x_coords < x_center])[-k:]
    right_indices = np.argsort(x_coords[x_coords >= x_center])[:k]
    
    # 合并索引，确保不重复选取
    selected_indices = list(set(up_indices) | set(down_indices) | set(left_indices) | set(right_indices))
    
    # 创建 selected 和 remaining 数据集
    selected_cali_lines = [lines[idx] for idx in selected_indices]
    remaining_lines = [lines[idx] for idx in range(len(lines)) if idx not in selected_indices]
    
    # 如果剩余数据少于10，返回提示信息
    if len(remaining_lines) < 10:
        print("该用户数据较少，跳过测试")
        return [], []

    return selected_cali_lines, remaining_lines


def select_by_quadrants_mpii(lines, k):
    """
    从预测空间中选择上下左右四个方向的 k 个样本。

    Args:
        lines (list): 包含所有数据行的列表。
        k (int): 每个方向选取的样本数。

    Returns:
        tuple: selected_cali_lines, remaining_lines
    """
    # 提取每行的 label
    labels = []
    line_data = []
    for line in lines:
        line = line.strip().split(" ")
        point = line[6]
        ratio = line[9].split(",")
        label = np.array(point.split(",")).astype("float")
        ratio = np.array(ratio).astype("float")
        label = label * ratio * 0.1
        labels.append(label)
        line_data.append(line)

    labels = np.array(labels)  # 转换为 numpy 数组
    x_coords, y_coords = labels[:, 0], labels[:, 1]

    # 计算中心点，用于分类上下左右
    x_center = np.mean(x_coords)
    y_center = np.mean(y_coords)

    # 分别找到上下左右的点索引
    up_indices = np.argsort(y_coords[y_coords < y_center])[-k:]
    down_indices = np.argsort(y_coords[y_coords >= y_center])[:k]
    left_indices = np.argsort(x_coords[x_coords < x_center])[-k:]
    right_indices = np.argsort(x_coords[x_coords >= x_center])[:k]

    # 合并索引，确保不重复选取
    selected_indices = list(set(up_indices) | set(down_indices) | set(left_indices) | set(right_indices))

    # 创建 selected 和 remaining 数据集
    selected_cali_lines = [lines[idx] for idx in selected_indices]
    remaining_lines = [lines[idx] for idx in range(len(lines)) if idx not in selected_indices]

    # 如果剩余数据少于10，返回提示信息
    if len(remaining_lines) < 10:
        print("该用户数据较少，跳过测试")
        return [], []

    return selected_cali_lines, remaining_lines
