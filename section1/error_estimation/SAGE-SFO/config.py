
commit = "这是SAGE-SFO模型的第一次训练"

few_shot_num = 9

'''Data params'''
batch_size = 32
epoch = 16
lr = 1e-3
train_decay_rate = 0.1
'''从第几epoch开始调整lr'''
lr_decay_start_step = 8
lr_decay_cycle = 2

'''data path'''
GazeCapture_root = "/home/hi/zhuzi/data/GCOutput/"
MPIIFaceGaze_root = "/home/hi/zhuzi/data/mpii/"

cur_dataset = "GazeCapture"

'''save path'''
save_path = "./checkpoint"
model_name = "SAGE-SFO"
save_start_step = 10
save_step = 2
test_save_path = "./evaluation"


'''test params'''
test_model_path = "/home/hi/zhuzi/paper-code/section1/model_finetune/AFF-Net/checkpoint/MPII/256_32_0.001/Iter_32_AFF-Net.pt"
test_begin_step = 26
test_end_step = 32
test_steps = 2
test_log_path = "./log"






