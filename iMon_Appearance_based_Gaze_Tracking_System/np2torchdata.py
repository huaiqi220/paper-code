import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import decode_utils
import mitdata_utils as mitutils
import pandas as pd
from config import sage_model_config as config
import loaderFnp


''' TODO: 这两函数直接从Numpy获得Distribute之后的torch dataset'''
def process_path(file_path, label, region,shuffle=False,num_workers=8):
    return loaderFnp.txtload(file_path, label, region,config.batch_size,shuffle=shuffle,num_workers=num_workers)

def process_path_enhanced(file_path, label, region):
    pass

def get_torch_dataset(dots, regions, shuffle= False,num_workers=8):
    indices = np.arange(len(dots))
    if shuffle:
        np.random.shuffle(indices)
    shuffled_dots = dots[indices, :]
    shuffled_regions = regions[indices, :]
    filenames = []
    for i in range(len(shuffled_dots)):
        region = shuffled_regions[i, :].astype(int)
        filenames.append(decode_utils.get_frame_path(region[0], region[1]))  
    # print(filenames[:5])
    # 至此，每一行分别为path , x, y, landmarks
    # print(filenames[:5], shuffled_dots[:5, -3:-1], shuffled_regions[:5])
    list_ds = (filenames,shuffled_dots[:,-3:-1], shuffled_regions)
    if config.enhanced:
        dataset = process_path_enhanced(list_ds[0],list_ds[1],list_ds[2])
    else:
        dataset  = process_path(list_ds[0],list_ds[1],list_ds[2],shuffle=shuffle,num_workers=num_workers)
    return dataset
        















if __name__ == "__main__":
    dataset_dict = mitutils.prep_meta_data()
    print(dataset_dict['val'][2][:5])
    dots_train, regions_train, df_info_train = dataset_dict['train']
    dots_val, regions_val, df_info_val = dataset_dict['val']
    dots_train = np.concatenate([dots_train, dots_val])
    regions_train = np.concatenate([regions_train, regions_val])
    df_info_train = pd.concat([df_info_train, df_info_val])
    dots_val, regions_val, df_info_val = dataset_dict['test']
    print('train data:', dots_train.shape, regions_train.shape, df_info_train.shape)
    print('val data:', dots_val.shape, regions_val.shape, df_info_val.shape)
    get_torch_dataset(dots_train, regions_train, shuffle=True)