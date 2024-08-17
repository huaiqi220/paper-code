import model
import A_Differential_Approach_For_Gaze_Estimation.dataloader.reader as reader
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
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

def dis(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    device = torch.device("cuda" + ":" + str(rank))

    config = yaml.safe_load(open("config.yaml"))
    config = config["test"]
    path = config["data"]["path"]
    model_name = config["load"]["model_name"]
    load_path = os.path.join(config["load"]["load_path"])

    #device = torch.device("cpu")
    save_name="evaluation"

    print(f"Test Set: tests")

    save_path = os.path.join(load_path, "checkpoint")

    if not os.path.exists(os.path.join(load_path, save_name)):
        os.makedirs(os.path.join(load_path, save_name))

    print("Read data")
    path1 = os.path.join(path,"Label","test")
    path1 = [os.path.join(path1, item) for item in os.listdir(path1)]
    dataset = reader.txtload(path1, os.path.join(path, "Image"), 256, num_workers=0, shuffle=False)

    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]
    epoch_log = open(os.path.join(load_path, f"{save_name}/epoch.log"), 'a')
    for save_iter in range(begin, end+step, step):
        print("Model building")
        net = model.model().to(rank)
        net = DDP(net)
        state_dict = torch.load(os.path.join(save_path, f"Iter_{save_iter}_{model_name}.pt"))
        net.load_state_dict(state_dict)
        net=net.module
        
        net.eval()

        print(f"Test {save_iter}")
        length = len(dataset)
        total = 0
        count = 0
        loss_fn = torch.nn.MSELoss()
        SE_log = open('./SE.log', 'w')
        with torch.no_grad():
            with open(os.path.join(load_path, f"{save_name}/{save_iter}.log"), 'w') as outfile:
                outfile.write("subjcet,name,x,y,labelx,labely,error\n")
                for j, data in enumerate(dataset):
                    data["face"] = data["face"].to(device)
                    data["left"] = data["left"].to(device)
                    data['right'] = data['right'].to(device)
                    data['rects'] = data['rects'].to(device)
                    labels = data["label"].to(device)

                    gazes = net(data["left"], data["right"], data['face'], data['rects'])
                    
                    names = data["name"]
                    print(f'\r[Batch : {j}]', end='')
                    #print(f'gazes: {gazes.shape}')
                    for k, gaze in enumerate(gazes):
                        #print(f'gaze: {gaze}')
                        gaze = gaze.cpu().detach()
                        count += 1
                        acc = dis(gaze, labels[k])
                        total += acc
                        gaze = [str(u) for u in gaze.numpy()]
                        label = [str(u) for u in labels.cpu().numpy()[k]]
                        name = names[k]
                        
                        log = [name] + gaze + label + [str(acc)]
                        
                        outfile.write(",".join(log) + "\n")
                SE_log.close()
                loger = f"[{save_iter}] Total Num: {count}, avg: {total/count} \n"
                outfile.write(loger)
                epoch_log.write(loger)
                print(loger)
    dist.destroy_process_group()

