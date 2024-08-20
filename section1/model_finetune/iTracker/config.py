
'''Data params'''
batch_size = 256
epoch = 20
lr = 1e-3
train_decay_rate = 0.1
'''从第几epoch开始调整lr'''
lr_decay_start_step = 10
lr_decay_cycle = 2

'''data path'''
GazeCapture_root = "/data/4_gc/2_gcout/"
MPIIFaceGaze_root = "/data/5_mp/mpiifg/"

cur_dataset = "GazeCapture"

'''save path'''
save_path = "./checkpoint"
model_name = "iTracker"
save_start_step = 0
save_step = 1
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/zhuzi/code/2024-5/paper-code/section1/model_finetune/iTracker/checkpoint/GazeCapture/512_20_0.001/Iter_1_iTracker.pt"
test_begin_step = 8
test_end_step = 12
save_steps = 1



'''Personal Cali Train && Test Params'''
cali_batch_size = 3
cali_epoch = 8
cali_lr = 1e-5
cali_last_layer = False

cali_image_num = 45
cali_rank = 0


