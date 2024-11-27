
commit = "GC数据集、DIF-NN-POG第一次训练"




''' super params '''
k = 4



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

cur_dataset = "GazeCapture"

'''save path'''
save_path = "./checkpoint"
model_name = "DIF_NN_POG"
save_start_step = 6
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section1/error_reduce/difNet/checkpoint/GazeCapture/GC数据集、DIF-NN-POG第一次训练/128_14_0.001/Iter_14_DIF_NN_POG.pt"
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

cur_rank = 7




