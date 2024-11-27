'''这是主实验画不同k的箱线图的代码'''


# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy import stats
# import os


# ''''
# 从evaluation那个日志格式里读每个人的校准前校准后
# 最后来算他们的结果、标准差和p-value
# '''

# path_list = ["4","6","8","10","12"]



# all_data = []
# for path in path_list:
#     log_path = os.path.join(".",path)
#     data = []
#     persons = os.listdir(log_path)



#     # 读取每个人的校准前和校准后的误差
#     for person in persons:
#         cur_path = os.path.join(log_path, person)
#         origin_res = os.path.join(cur_path, "origin_test")
#         cali_res = os.path.join(cur_path, "calibration_test")

#         # 读取 origin_error
#         print(origin_res)
#         with open(os.path.join(origin_res, "error.log"), "r") as f:
#             origin_error = float(f.readlines()[-1].split(": ")[-1])

#         # 读取 cali_error
#         with open(os.path.join(cali_res, "error.log"), "r") as f:
#             cali_error = float(f.readlines()[-1].split(": ")[-1])

#         data.append([person, origin_error])

#     # 转换数据为DataFrame
#     df = pd.DataFrame(data, columns=['Person', 'Origin_Error'])
#     all_data.append(df)





# # 示例数据（替换为你的实际数据）
# np.random.seed(42)
# data1 = np.random.rand(100, 5)  # 100行5列的浮点数据
# data2 = np.random.rand(100, 5)

# # 创建箱线图
# plt.figure(figsize=(10, 6))

# # 绘制第一个Numpy数组的箱线图
# plt.boxplot(data1, positions=np.arange(1, 6), patch_artist=True,
#             boxprops=dict(facecolor='yellow', color='black'),
#             medianprops=dict(color='red'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='black'))

# # 绘制第二个Numpy数组的箱线图
# plt.boxplot(data2, positions=np.arange(1, 6) + 0.5, patch_artist=True,
#             boxprops=dict(facecolor='blue', color='black'),
#             medianprops=dict(color='red'),
#             whiskerprops=dict(color='black'),
#             capprops=dict(color='black'))

# # 添加图例
# plt.legend(['Data1 (Yellow)', 'Data2 (Blue)'], loc='upper right')

# # 添加轴标签
# plt.xticks(np.arange(1, 6) + 0.25, [f'Col{i}' for i in range(1, 6)])
# plt.xlabel('Columns')
# plt.ylabel('Values')
# plt.title('Boxplots of Two Numpy Arrays')

# # 显示图像
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import font_manager

font = font_manager.FontProperties(fname='C://Windows//Fonts//simsun.ttc', size=12)

# 路径列表
path_list = ["4", "6", "8", "10", "12"]

# 存储所有路径的数据
origin_errors = []  # 校准前误差
cali_errors = []    # 校准后误差

for path in path_list:
    log_path = os.path.join(".", path)
    origin_data = []
    cali_data = []
    persons = os.listdir(log_path)

    # 读取每个人的校准前和校准后误差
    for person in persons:
        cur_path = os.path.join(log_path, person)
        origin_res = os.path.join(cur_path, "origin_test")
        cali_res = os.path.join(cur_path, "calibration_test")

        # 读取 origin_error
        with open(os.path.join(origin_res, "error.log"), "r") as f:
            origin_error = float(f.readlines()[-1].split(": ")[-1])
        origin_data.append(origin_error)

        # 读取 cali_error
        with open(os.path.join(cali_res, "error.log"), "r") as f:
            cali_error = float(f.readlines()[-1].split(": ")[-1])
        cali_data.append(cali_error)

    origin_errors.append(origin_data)
    cali_errors.append(cali_data)

# 绘制箱线图
plt.figure(figsize=(10, 5))

# 绘制校准前误差箱线图
plt.subplot(1, 2, 1)
colors = ['#FFFFE0', '#ADD8E6', '#90EE90', '#FFB6C1', '#D3D3D3']
for idx, data in enumerate(origin_errors):
    plt.boxplot(
        data,
        whis=1.0,
        positions=[idx + 1],
        patch_artist=True,
        showfliers=False,  # 移除离群值
        boxprops=dict(facecolor=colors[idx], color='black'),
        medianprops=dict(color='red'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )
plt.xticks(range(1, len(path_list) + 1), path_list)
plt.yticks(np.arange(0.6, 3.6, 0.2))  # Y轴刻度从0.5到3.5
plt.xlabel('校准向量位数',fontproperties=font)
plt.ylabel('误差(厘米)',fontproperties=font)
plt.title('原始误差箱线图',fontproperties=font)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

# 绘制校准后误差箱线图
plt.subplot(1, 2, 2)
for idx, data in enumerate(cali_errors):
    plt.boxplot(
        data,
        whis=1.0,
        positions=[idx + 1],
        patch_artist=True,
        showfliers=False,  # 移除离群值
        boxprops=dict(facecolor=colors[idx], color='black'),
        medianprops=dict(color='red'),
        whiskerprops=dict(color='black'),
        capprops=dict(color='black'),
    )
plt.xticks(range(1, len(path_list) + 1), path_list)
plt.yticks(np.arange(0.6, 3.1, 0.2))  # Y轴刻度从0.5到3.5
plt.xlabel('校准向量位数',fontproperties=font)
plt.ylabel('误差(厘米)',fontproperties=font)
plt.title('校准误差箱线图',fontproperties=font)
plt.grid(True, linestyle='--', alpha=0.7)  # 添加网格线

# 显示图像
plt.tight_layout()
# plt.show()
plt.savefig('boxplot.svg', format='svg')
