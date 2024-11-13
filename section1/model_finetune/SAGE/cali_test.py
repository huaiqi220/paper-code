import torch
import sys
import os
import config
import random
from dataloader import gc_reader
from dataloader import mpii_reader
import logging
from model import SAGE as model
import math
import torch.nn as nn
import time
from torch.cuda.amp import autocast


def dis(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))


def test_func(name, calimodel, dataset,save_path,rank,cali_test):
    # print(calimodel)
    if cali_test:
        save_path = os.path.join(save_path,"calibration_test")
    else:
        save_path = os.path.join(save_path,"origin_test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    device = torch.device("cuda" + ":" + str(rank))
    calimodel.to(device)
    calimodel.eval()
    total = 0
    count = 0
    with torch.no_grad():
        with open(os.path.join(save_path, "error.log"), 'w') as outfile:
            outfile.write("subjcet,name,x,y,labelx,labely,error\n")
            for j, data in enumerate(dataset):
                data["face"] = data["face"].to(device)
                data["left"] = data["left"].to(device)
                data["right"] = data["right"].to(device)
                # data["grid"] = data["grid"].to(device)
                data["rects"] = data["rects"].to(device)
                labels = data["label"]
                # data["poglabel"] = data["poglabel"].to(device)
                gazes = calimodel(data["left"], data["right"], data["rects"])

                print(f'\r[Batch : {j}]', end='')
                #print(f'gazes: {gazes.shape}')
                for k, gaze in enumerate(gazes):
                    #print(f'gaze: {gaze}')
                    gaze = gaze.cpu().detach()
                    count += 1
                    acc = dis(gaze, labels[k])
                    total += acc
                    gaze = [str(u) for u in gaze.numpy()]
                    label = [str(u) for u in labels.numpy()[k]]
                    log = [name] + gaze + label + [str(acc)]
                    
                    outfile.write(",".join(log) + "\n")
            loger = f"[{name}] Total Num: {count}, avg: {total/count} \n"
            outfile.write(loger)
            print(loger)
    



def cali_train_func(name,calimodel,dataset,save_path,rank):
    """要把模型return回去"""
    device = torch.device("cuda" + ":" + str(rank))
    calimodel.to(device)
    calimodel.train()

    '''全部锁定，然后解锁最后fc'''
    if config.cali_last_layer:
        for param in calimodel.parameters():
            param.requires_grad = False
        for param in calimodel.fc4.parameters():
            param.requires_grad = True
        for param in calimodel.fc5.parameters():
            param.requires_grad = True


    loss_func = nn.MSELoss()
    lr = config.cali_lr
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, calimodel.parameters()), lr, weight_decay=0.0005)
    save_path = os.path.join(save_path,"train_log")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    ''' 开始训练 '''
    length = len(dataset)
    with open(os.path.join(save_path, "train_loss.log"), 'w') as outfile:
        for epoch in range(1, config.cali_epoch + 1):
            # if epoch >= config.lr_decay_start_step and epoch % config.lr_decay_cycle == 0:
            #     base_lr = base_lr * config.train_decay_rate
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = base_lr
            # 微调lr暂时不变
            time_begin = time.time()
            for i, data in enumerate(dataset):
                optimizer.zero_grad()
                with autocast():
                    data["face"] = data["face"].to(device)
                    data["left"] = data["left"].to(device)
                    data["right"] = data["right"].to(device)
                    # data["grid"] = data["grid"].to(device)
                    data["rects"] = data["rects"].to(device)
                    data["label"] = data["label"].to(device)
                    # data["poglabel"] = data["poglabel"].to(device)
                    gaze_out = calimodel(data["left"], data["right"], data["rects"])
                    loss = loss_func(gaze_out, data["label"])        


                loss.backward()
                optimizer.step()
                time_remain = (length - i - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                time_remain_total = time_remain + epoch_time * (config.cali_epoch - epoch)
                log = f"[{epoch}/{config.cali_epoch}]: [{i}/{length}] loss:{loss:.10f} lr:{lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                print(log)
                sys.stdout.flush()
                outfile.flush()
    calimodel.cpu()
    return calimodel





def cali_test_func(root_path, label):
    '''
    返回三个数字：校准前error，训练loss，校准后error
    '''
    rank = 5

    cur_id = label.split("/")[-1].split(".")[0]
    cali_folder = os.path.join(config.test_save_path,config.cur_dataset,config.cur_fold, "cali_num_" + str(config.cali_image_num) +"_" + str(config.cali_last_layer), cur_id)

    all_label = []
    with open(label, "r") as f:
        all_label = f.readlines()
        all_label.pop(0)

    # 部分用户采集图片很少
    if len(all_label) <= config.cali_image_num:
        print("该用户数据较少，跳过测试")
        return

    selected_cali_lines = random.sample(all_label, config.cali_image_num)

    remaining_lines = [line for line in all_label if line not in selected_cali_lines]
    if len(remaining_lines) < 10:
        print("该用户数据较少，跳过测试")
        return 
    
    if config.cur_dataset == "GazeCapture":
        all_test_dataset = gc_reader.calitxtload(all_label,os.path.join(root_path,"Image"),32,True,8,True)
        cali_train_dataset = gc_reader.calitxtload(selected_cali_lines,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)
        cali_test_dataset = gc_reader.calitxtload(remaining_lines,os.path.join(root_path,"Image"),32,True,8,True)

    if config.cur_dataset == "MPII":
        all_test_dataset = mpii_reader.calitxtload(all_label,os.path.join(root_path,"Image"),32,True,8,True)
        cali_train_dataset = mpii_reader.calitxtload(selected_cali_lines,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)
        cali_test_dataset = mpii_reader.calitxtload(remaining_lines,os.path.join(root_path,"Image"),32,True,8,True)


    test_model_path = config.test_model_path
    calimodel = model()
    statedict = torch.load(test_model_path)
    new_state_dict = {}
    for key, value in statedict.items():
    # 如果 key 以 "module." 开头，则去掉这个前缀
        new_key = key[7:]
        new_state_dict[new_key] = value
    calimodel.load_state_dict(new_state_dict)
    device = torch.device("cuda" + ":" + str(rank))
    # 全量测试
    test_func(cur_id,calimodel,all_test_dataset,cali_folder,rank,False)
    # 校准训练
    calimodel = cali_train_func(cur_id,calimodel,cali_train_dataset,cali_folder,rank)
    # 校准测试
    test_func(cur_id,calimodel,cali_test_dataset,cali_folder,rank,True)

                


if __name__ == "__main__":
    if config.cur_dataset == "GazeCapture":
        root_path = config.GazeCapture_root
    elif config.cur_dataset == "MPII":
        root_path = config.MPIIFaceGaze_root
    
    test_label_path = os.path.join(root_path,"Label","K_Fold_norm",config.cur_fold, "test")
    label_list = [os.path.join(test_label_path, item) for item in os.listdir(test_label_path)]
    for label in label_list:
        res = cali_test_func(root_path, label)
