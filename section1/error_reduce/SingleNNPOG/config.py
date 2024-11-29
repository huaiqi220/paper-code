
commit = "测试SingleNNPOG GC数据集"

'''Data params'''
batch_size = 256
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
model_name = "SingleNNPOG"
save_start_step = 10
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section1/error_reduce/SingleNNPOG/checkpoint/GazeCapture/256_14_0.001/Iter_14_SingleNNPOG.pt"
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
cali_image_num = 15



cur_rank = 0