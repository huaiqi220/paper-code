
'''loss params'''
loss_alpha = 0.2

hm_loss_alpha = 10




'''Heatmap params'''
mobile = True
heatmap = False
r = 'r0.2'
hm_size = 128
scale = 5 if mobile else 2
hm_level = 4


'''Data params'''
batch_size = 128
epoch = 24
lr = 1e-3
train_decay_rate = 0.7
'''从第几epoch开始调整lr'''
lr_decay_start_step = 16
lr_decay_cycle = 2

'''data path'''
GazeCapture_root = "/home/hi/zhuzi/data/GCOutput/"
MPIIFaceGaze_root = "/data/5_mp/mpiifg/"
GazeCapture_Cali_path = "not use"

'''save path'''
save_path = "./checkpoint"
model_name = "MobileNetV2-Gaze"
save_start_step = 20
save_step = 2


'''test params'''
test_begin_step = 8
test_end_step = 12
save_steps = 1
test_load_path = "."




