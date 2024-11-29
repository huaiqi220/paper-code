from crossNet import SingleNNPoG as model
from dataloader import gc_reader
from dataloader import mpii_reader 
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
import math
import config

def dis(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

if __name__ == "__main__":

    """判断加载哪个数据集"""
    if config.cur_dataset == "GazeCapture":
        root_path = config.GazeCapture_root
    elif config.cur_dataset == "MPII":
        root_path = config.MPIIFaceGaze_root

    model_path = config.test_model_path
    model_name = config.model_name


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    label_path = os.path.join(root_path,"Label","model_fineture", "test")
    label_path = [os.path.join(label_path, item) for item in os.listdir(label_path)]

    log_path = config.test_log_path
    log_path = os.path.join(log_path, config.cur_dataset)


    if config.cur_dataset == "GazeCapture":
        dataset = gc_reader.txtload(label_path, os.path.join(root_path, "Image"), config.batch_size, shuffle=True,
                                num_workers=8)
    elif config.cur_dataset == "MPII":
        dataset = mpii_reader.txtload(label_path, os.path.join(root_path, "Image"), config.batch_size, shuffle=True,
                                num_workers=8)
    begin = config.test_begin_step
    end = config.test_end_step
    step = config.test_steps
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)

    epoch_log = open(os.path.join(log_path, f"epoch.log"), 'w+')
    for save_iter in range(begin, end+step, step):
        print("Model building")
        net = model()
        net = nn.DataParallel(net)
        state_dict = torch.load(os.path.join(model_path, f"Iter_{save_iter}_{model_name}.pt"))
        net.load_state_dict(state_dict)
        net=net.module
        net.to(device)
        
        net.eval()

        print(f"Test {save_iter}")
        length = len(dataset)
        total = 0
        count = 0
        loss_fn = torch.nn.MSELoss()
        SE_log = open('./SE.log', 'w')
        with torch.no_grad():
            with open(os.path.join(log_path, f"{save_iter}.log"), 'w') as outfile:
                outfile.write("subjcet,name,x,y,labelx,labely,error\n")
                for j, data in enumerate(dataset):
                    # data["face"] = data["face"].to(device)
                    data["left"] = data["left"].to(device)
                    data["right"] = data["right"].to(device)
                    # data["grid"] = data["grid"].to(device)
                    # data["rects"] = data["rects"].to(device)
                    # data["label"] = data["label"].to(device)
                    labels = data["label"]
                    
                    gazes = net(data["left"], data["right"])
                    
                    # names = data["name"]
                    names = "name"
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
                        name = names
                        
                        log = name + gaze + label + [str(acc)]
                        
                        outfile.write(",".join(log) + "\n")
                SE_log.close()
                loger = f"[{save_iter}] Total Num: {count}, avg: {total/count} \n"
                outfile.write(loger)
                epoch_log.write(loger)
                print(loger)

