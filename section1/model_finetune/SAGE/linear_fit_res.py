import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression

# 定义函数来解析 error.log 文件并计算每个文件夹的 user_id_error 和 error 平均值
def calculate_errors(root_dir):
    user_id_errors = []
    original_errors = []

    for folder_name in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder_name, 'origin_test')
        error_log_path = os.path.join(folder_path, 'error.log')

        if not os.path.exists(error_log_path):
            continue

        # 读取 error.log 文件
        with open(error_log_path, 'r') as f:
            lines = f.readlines()[1:-2]

        # 从日志文件中提取感兴趣的列
        data = [line.strip().split(',')[1:] for line in lines]
        df = pd.DataFrame(data, columns=[ 'output_x', 'output_y', 'label_x', 'label_y', 'error'], dtype=float)
        
        # 获取输出、标签和误差列
        output_x, output_y = df['output_x'].values, df['output_y'].values
        label_x, label_y = df['label_x'].values, df['label_y'].values
        error = df['error'].values

        # 拟合 output_x -> label_x 和 output_y -> label_y 的线性回归模型
        model_x = LinearRegression().fit(output_x.reshape(-1, 1), label_x)
        model_y = LinearRegression().fit(output_y.reshape(-1, 1), label_y)

        # 使用拟合的模型计算 cx, cy
        cx = model_x.predict(output_x.reshape(-1, 1))
        cy = model_y.predict(output_y.reshape(-1, 1))

        # 计算预测点 (cx, cy) 和标签点 (label_x, label_y) 之间的欧氏距离
        euclidean_distances = np.sqrt((cx - label_x) ** 2 + (cy - label_y) ** 2)
        user_id_error = np.mean(euclidean_distances)
        user_id_errors.append(user_id_error)

        # 计算原始 error 列的平均值
        original_error_mean = np.mean(error)
        original_errors.append(original_error_mean)

    # 计算所有文件夹的拟合后平均误差的平均值
    avg_user_id_error = np.mean(user_id_errors) if user_id_errors else None

    # 计算拟合后平均误差的标准差
    std_user_id_error = np.std(user_id_errors) if user_id_errors else None

    error_mean  = np.mean(original_errors)

    # 计算拟合后平均误差相较于原始 error 平均值的 p 值
    if user_id_errors and original_errors:
        _, p_value = ttest_rel(user_id_errors, original_errors)
    else:
        p_value = None

    return avg_user_id_error, std_user_id_error, p_value, error_mean

# 主函数
if __name__ == "__main__":
    root_directory = "/home/hi/zhuzi/paper-code/section1/model_finetune/AFF-Net/evaluation/MPII/fold1/cali_num_72_False"  # 替换为包含所有子文件夹的目录
    avg_error, std_error, p_val, error_mean = calculate_errors(root_directory)

    # 输出结果
    print(f"初始误差: {error_mean}")
    print(f"拟合后误差: {avg_error}")
    print(f"拟合后误差标准差: {std_error}")
    print(f"P 值: {p_val}")
