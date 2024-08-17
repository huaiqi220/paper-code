import config.sage_model_config as config
import time
import mitdata_utils as mitutils
import numpy as np
import pandas as pd
import os
import torch
import sage
import losses
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import np2torchdata as n2t
import sys



def test_heatmap(ddp_model,dots_val,regions_val,df_info_val,dots_train,regions_train,df_info_train,rank):
    # df_pred_scopes = losses.get_pred_scope(df_info_train, regions_train,
    #                                     dots_train, df_info_val)
    # subjectIDs, subjectID_indices = np.unique(regions_val[:, 0], return_index=True) 
    # ecl_errs = []
    # pred_count = 0
    # invalid_frames = []
    # valid_frames = []
    # dot_errs = []
    # dot_errs1 = []

    # single_preds = []
    # step = 10   
    # for subjectID_idx in range(0, len(subjectIDs), step):
    #     subjectID_list = subjectIDs[subjectID_idx:subjectID_idx+step]
    #     print('subjectID_list', subjectID_list)
    #     dots = dots_val[np.isin(dots_val[:, 0], subjectID_list), :]
    #     regions = regions_val[np.isin(dots_val[:, 0], subjectID_list), :]
    #     ds = n2t.get_torch_dataset(dots, regions, shuffle=False,num_workers=8)
    #     # batch_size = 64 if (len(dots) > 64) else 64
    #     for i, data in enumerate(ds):
    #         preds = ddp_model(data["left"].to(rank), data["right"].to(rank), data['eyelandmark'].to(rank), data['orientation'].to(rank))
    #         preds = preds[:, :, :, 0]
    #         dots = dots[-len(preds):, :]
    #         regions = regions[-len(preds):, :]
    pass


def main():
        # 初始化进程组
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # Print the configuration
    print('#Architecture', config.arc, ' #Heatmap', config.heatmap,
        ' # Mobile', config.mobile, ' #Test ', config.test)
    print('#Regions', config.regions, ' #Enhanced', config.enhanced)
    t = time.time()
    dataset_dict = mitutils.prep_meta_data()
    # print(dataset_dict['val'][2][:5])
    dots_train, regions_train, df_info_train = dataset_dict['train']
    dots_val, regions_val, df_info_val = dataset_dict['val']
    dots_train = np.concatenate([dots_train, dots_val])
    regions_train = np.concatenate([regions_train, regions_val])
    df_info_train = pd.concat([df_info_train, df_info_val])
    dots_val, regions_val, df_info_val = dataset_dict['test']
    print('train data:', dots_train.shape, regions_train.shape, df_info_train.shape)
    print('val data:', dots_val.shape, regions_val.shape, df_info_val.shape)
    # =================get load model name and path =================
    #  base_model = 'AlexNet', 'MobileNetV2', 'EfficientNetB3'
    base_model = config.base_model
    mobile_str = 'm' if config.mobile else 't'
    weights_str = 'scratch' if (config.weights is None) else str(config.weights)
    model_name = config.arc + '_' + base_model + '_' + weights_str + '_' + \
        mobile_str + '_' + \
        str(config.faceIm_size) + '_' + str(config.eyeIm_size) + '_' + \
        str(config.channel) + '_' + config.regions
    if (config.enhanced):
        model_name += '_enhanced'
    # Create folder to store model
    model_path = config.path + 'model/euclidean/'
    if (config.heatmap):
        model_path = config.path + 'model/heatmap/'
        model_name += '_hm_' + str(config.r)
    model_path = model_path + base_model + '/'
    if(not os.path.exists(model_path)):
        os.makedirs(model_path)
    print('Base model: ', base_model, '-', model_name, ' Model path: ', model_path)
    # ===================== end ======================
    if torch.cuda.is_available():
        # 获取可用GPU数量
        num_gpus = torch.cuda.device_count()
        print("可用GPU数量:", num_gpus)
    else:
        print("CUDA不可用")

    heatmap = config.heatmap
    model = sage.SAGE_Imon(rank,base_model,heatmap).to(rank)
    ddp_model = DDP(model,find_unused_parameters=True)
    device = torch.device("cuda" + ":" + str(rank))

    # 如果是测试，加载保存的模型文件
    if(config.test and config.pretrained_model is not None):
        state_dict = torch.load(config.pretrained_model)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
    # 这个Losses是仿照tf写的torch版本，还没测过，如果需要排查结果需要check一下
    loss = losses.heatmap_loss if config.heatmap else losses.euclidean_loss

    '''TRAIN/TEST'''
    if(config.test):
        # test_dataset_dict = n2t.get_torch_dataset(dots_val,regions_val,shuffle=True,num_workers=8)
        if(config.heatmap):
            print("此处编写加载模型heatmap测试逻辑")
            # test_heatmap(ddp_model,dots_val,regions_val,df_info_val,rank)

        else:
            print("此处编写加载模型欧式距离测试逻辑")
            test_euclidean(ddp_model,dots_val,regions_val,rank)
        return 0

    '''Train dataset'''
    dataset_dict = n2t.get_torch_dataset(dots_train,regions_train,shuffle=True,num_workers=8)
    # loss_function = losses.heatmap_loss if config.heatmap else losses.euclidean_loss
    loss_function = nn.MSELoss() if config.heatmap else losses.euclidean_loss
    val_dataset_dict = n2t.get_torch_dataset(dots_val,regions_val,shuffle=True,num_workers=8)
    print("构建优化器")
    base_lr = config.current_lr
    optimizer = torch.optim.Adam(ddp_model.parameters(), base_lr, weight_decay=0.0005)
    print("训练")
    length = len(dataset_dict)
    log_save_path = os.path.join(config.log_path, str(config.test) + "_" + str(config.arc) + "_" + str(base_lr))
    if not os.path.exists(log_save_path):
        # 如果文件夹不存在，则创建它
        os.makedirs(log_save_path)
    cur_min_loss = config.current_best_val_loss
    with open(os.path.join(log_save_path, "train_log"), 'w') as outfile:
        for i in range(1,config.epochs):
            # step the lr
            if i <= config.decay_epoch:
                cur_lr = base_lr
            else:
                cur_lr = cur_lr * 0.8
            for param_group in optimizer.param_groups:
                param_group["lr"] = cur_lr

            time_begin = time.time()
            cur_loss = 10
            for j, data in enumerate(dataset_dict):
                gaze = ddp_model(data["left"].to(device), data["right"].to(device), data['eyelandmark'].to(device), data['orientation'].to(device))
                loss = loss_function(gaze, data["label"].to(device))
                optimizer.zero_grad()
                loss.mean().backward()
                cur_loss = loss.mean().item()
                optimizer.step()
                time_remain = (length - j - 1) * ((time.time() - time_begin) / (j + 1)) / 3600
                epoch_time = (length - 1) * ((time.time() - time_begin) / (j + 1)) / 3600
                time_remain_total = time_remain + epoch_time * (config.epochs - i)
                log = f"[{i}/{config.epochs}]: [{j}/{length}] loss:{cur_loss:.5f} lr:{cur_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                print(log)
                sys.stdout.flush()
                outfile.flush()
            # Validate the model
            with torch.no_grad():
                validation_loss = 0
                count = 0
                for j, data in enumerate(val_dataset_dict):
                    val_gazes = ddp_model(data["left"].to(device), data["right"].to(device), data['eyelandmark'].to(device), data['orientation'].to(device))
                    val_loss = loss_function(val_gazes, data["label"].to(device))
                    if rank == 0 and config.heatmap:
                        losses.show_heatmap(val_gazes, data["label"].to(device))
                    validation_loss += val_loss.sum().item()
                    count += len(data["label"])
                validation_loss /= count
                print(validation_loss)
                if(validation_loss < cur_min_loss and rank == 0):
                    log = f"Validation loss: {validation_loss} is less than the current min loss: {cur_min_loss}, saving the model."
                    outfile.write(log + "\n")
                    print(log)
                    sys.stdout.flush()
                    outfile.flush()
                    cur_min_loss = validation_loss
                    torch.save(ddp_model.state_dict(), os.path.join(model_path, f"Iter_{i}_{model_name}.pt"))
                # torch.save(ddp_model.state_dict(), os.path.join(model_path, f"Iter_{i}_{model_name}.pt"))
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
