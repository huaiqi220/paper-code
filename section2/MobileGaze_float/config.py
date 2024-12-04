

commit = "GC数据集浮点数类型校准向量第二次实验、已经修改fc"





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

''' super params '''
k = 12

test_model_path = "/home/hi/zhuzi/paper-code/section2/MobileGaze_float/checkpoint/GazeCapture/GC数据集浮点数类型校准向量第二次实验、已经修改fc/128_14_0.001_12/Iter_10_MobileNetV2-Gaze-PoG.pt"
test_begin_step = 26
test_end_step = 32
test_steps = 2
test_log_path = "./log"
cur_rank = 3





'''Personal Cali Train && Test Params'''
cali_batch_size = 8
cali_epoch = 50
cali_lr = 1e-3
'''
float32
binary
'''
cali_vector_type = "float32"

# k =  4 * c
cali_image_num = 24






