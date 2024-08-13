
'''loss params'''
loss_alpha = 0.2




'''Heatmap params'''
mobile = True
heatmap = False
r = 'r0.2'
hm_size = 128
scale = 5 if mobile else 2
hm_level = 4

'''Data params'''
batch_size = 64
epoch = 100
lr = 1e-2
train_decay = 0.7
'''从第几epoch开始调整lr'''
decay_step = 30

'''data path'''
GazeCapture_root = "/data/4_gc/2_gcout/"
MPIIFaceGaze_root = "/data/5_mp/mpiifg/"
GazeCapture_Cali_path = "/home/zhuzi/code/2024-5/paper-code/Learning_to_Personalize_in_Appearance-Based_Gaze_Tracking/dataloader/calibration_vectors.json"

'''save path'''
save_path = "./checkout"
model_name = "MobileNetV2-Gaze"
save_start_step = 60
save_step = 4


'''test params'''
test_begin_step = 8
test_end_step = 12
save_steps = 1
test_load_path = "."



