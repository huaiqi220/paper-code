
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
model_name = "AFF-Net"
save_start_step = 24
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section1/model_finetune/AFF-Net/checkpoint/MPII/256_32_0.001/Iter_32_AFF-Net.pt"
test_begin_step = 26
test_end_step = 32
test_steps = 2
test_log_path = "./log"



'''Personal Cali Train && Test Params'''
cali_batch_size = 3
cali_epoch = 8
cali_lr = 1e-5
cali_last_layer = False

# k =  4 * c
cali_image_num = 72




