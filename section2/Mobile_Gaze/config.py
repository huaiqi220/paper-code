
commit = "尝试使用STE_origin,传统STE,在loss加入正则化项"




''' super params '''
k = 12



'''Heatmap params'''
mobile = True
heatmap = False
r = 'r0.2'
hm_size = 128
scale = 5 if mobile else 2
hm_level = 4




'''Data params'''
batch_size = 128
epoch = 12
lr = 1e-3
train_decay_rate = 0.1
'''从第几epoch开始调整lr'''
lr_decay_start_step = 6
lr_decay_cycle = 2

'''data path'''
GazeCapture_root = "/home/hi/zhuzi/data/GCOutput/"
MPIIFaceGaze_root = "/home/hi/zhuzi/data/mpii/"

cur_dataset = "GazeCapture"

'''save path'''
save_path = "./checkpoint"
model_name = "MobileNetV2-Gaze-PoG"
save_start_step = 6
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section2/Mobile_Gaze/checkpoint/GazeCapture/128_32_0.001/Iter_32_MobileNetV2-Gze-PoG.pt"
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
cali_image_num = 150

cur_rank = 4




