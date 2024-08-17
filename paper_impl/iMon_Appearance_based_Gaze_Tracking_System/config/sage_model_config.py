import os

path = os.path.expanduser('/data/4_gc/3_gcsage/')
enhanced_path = os.path.expanduser('PATH_TO_ENHANCED_DATASET')
log_path = "./log/"

'''Train/Test'''
test = False

'''Data params'''
mobile = True
regions = 'default'  # 'fan': improved pre-processing /'default': GazeCapture pre-processing
enhanced = False

'''Model params'''
arc = 'SAGE'  # 'SAGE', 'iTracker'
base_model = 'MobileNetV2' # 'AlexNet', 'MobileNetV2', 'EfficientNetB3'
weights = None
pretrained_path = path + 'model/euclidean/' + base_model + '/'
# pretrained_model = None  # pretrained_path + 'MODEL.hdf5'
pretrained_model = pretrained_path + "Iter_49_SAGE_MobileNetV2_scratch_m_112_112_3_default.pt"
current_lr = 0.001
current_best_val_loss = 10
current_training_round = 0
dropout = 0.5

'''Heatmap params'''
heatmap = False
r = 'r0.2'
hm_size = 128
scale = 4 if mobile else 2

'''Image params'''
channel = 3  # 1:grayscale, 3:rgb
wholeIm_size = 112
faceIm_size = 112
eyeIm_size = 112
faceGrid_size = 25

'''Training params'''
batch_size = 256
steps_per_epoch = 10*100
epochs = 150
decay_epoch = 1
