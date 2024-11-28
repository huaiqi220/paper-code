
commit = "GC数据集梯度约束STE，不加loss"

''' super params '''
k = 12


'''
1 原始版本STE，不加loss约束
2 原始版本STE，加loss熵鼓励
3 tanh模拟软STE
4 梯度约束STE，不加loss
5 梯度约束STE，加loss约束
6 原始版本STE，加loss约束
'''
current = 1


'''Heatmap params'''
mobile = True
heatmap = False
r = 'r0.2'
hm_size = 128
scale = 5 if mobile else 2
hm_level = 4




'''Data params'''
batch_size = 128
epoch = 14
lr = 1e-3
train_decay_rate = 0.5
'''从第几epoch开始调整lr'''
lr_decay_start_step = 6
lr_decay_cycle = 2

'''data path'''
GazeCapture_root = "/home/hi/zhuzi/data/GCOutput/"
MPIIFaceGaze_root = "/home/hi/zhuzi/data/mpii/"
''' 1  2 3 4'''
mpii_K = "1"

cur_dataset = "GazeCapture"

'''save path'''
save_path = "./checkpoint"
model_name = "MobileNetV2-Gaze-PoG"
save_start_step = 6
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section2/Mobile_Gaze/checkpoint/GazeCapture/训练一个k12的基准测试模型，在这个模型基础上测试校准样本数量，原始版本Fc2，cat特征合并方式/128_14_0.001_1/Iter_14_MobileNetV2-Gaze-PoG.pt"
test_begin_step = 26
test_end_step = 32
test_steps = 2
test_log_path = "./log"



'''Personal Cali Train && Test Params'''
cali_batch_size = 8
cali_epoch = 20
cali_lr = 1e-7
cali_last_layer = False
'''
float32
binary
'''
cali_vector_type = "binary"

# k =  4 * c
cali_image_num = 15

cur_rank = 3




