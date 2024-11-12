
'''Data params'''
batch_size = 256
epoch = 32
lr = 1e-3
train_decay_rate = 0.1
'''从第几epoch开始调整lr'''
lr_decay_start_step = 24
lr_decay_cycle = 2

'''data path'''
GazeCapture_root = "/home/hi/zhuzi/data/GCOutput/"
MPIIFaceGaze_root = "/home/hi/zhuzi/data/mpii/"

cur_dataset = "MPII"

'''save path'''
save_path = "./checkpoint"
model_name = "iTracker"
save_start_step = 24
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section1/model_finetune/iTracker/checkpoint/MPII/256_32_0.001/Iter_32_iTracker.pt"
test_begin_step = 26
test_end_step = 32
save_steps = 2
test_log_path = "./log"



'''Personal Cali Train && Test Params'''
cali_batch_size = 3
cali_epoch = 8
cali_lr = 1e-5
cali_last_layer = True

cali_image_num = 72
cali_rank = 2


