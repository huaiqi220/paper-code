''''
从evaluation那个日志格式里读每个人的校准前校准后
最后来算他们的结果、标准差和p-value
'''

import numpy as np
import pandas as pd
from scipy import stats
import os

log_path = "./evaluation/GazeCapture/cat、低复杂度fc2、k4、STE硬离散/cali_num_15_False_1e-07_4"
data = []
persons = os.listdir(log_path)

# 读取每个人的校准前和校准后的误差
for person in persons:
    cur_path = os.path.join(log_path, person)
    origin_res = os.path.join(cur_path, "origin_test")
    cali_res = os.path.join(cur_path, "calibration_test")

    # 读取 origin_error
    print(origin_res)
    with open(os.path.join(origin_res, "error.log"), "r") as f:
        origin_error = float(f.readlines()[-1].split(": ")[-1])

    # 读取 cali_error
    with open(os.path.join(cali_res, "error.log"), "r") as f:
        cali_error = float(f.readlines()[-1].split(": ")[-1])

    data.append([person, origin_error, cali_error])

# 转换数据为DataFrame
df = pd.DataFrame(data, columns=['Person', 'Origin_Error', 'Cali_Error'])

# 计算 origin_error 和 cali_error 的平均值和标准差
origin_mean = df['Origin_Error'].mean()
cali_mean = df['Cali_Error'].mean()
origin_std = df['Origin_Error'].std()
cali_std = df['Cali_Error'].std()

# 计算 p 值（配对 t 检验）
t_stat, p_value = stats.ttest_rel(df['Origin_Error'], df['Cali_Error'])

# 打印结果
print("原始误差 (Origin_Error) 的平均值:", origin_mean)
print("校准误差 (Cali_Error) 的平均值:", cali_mean)
print("原始误差 (Origin_Error) 的标准差:", origin_std)
print("校准误差 (Cali_Error) 的标准差:", cali_std)
print("校准前后误差的 p 值:", p_value)
