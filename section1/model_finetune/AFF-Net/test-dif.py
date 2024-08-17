from difNet import diffNet
import td_reader as reader
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

def dis(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    config = config["test"]
    path = config["data"]["path"]
    model_name = config["load"]["model_name"]
    load_path = os.path.join(config["load"]["load_path"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    save_name="evaluation"

    print(f"Test Set: tests")

    save_path = os.path.join(load_path, "checkpoint","diff")

    if not os.path.exists(os.path.join(load_path, save_name)):
        os.makedirs(os.path.join(load_path, save_name))

    path1 = os.path.join(path, "Label_dif", "test")
    path1 = [os.path.join(path1, item) for item in os.listdir(path1)]
    print("Read data")
    dataset = reader.txtload(path1, os.path.join(path, "Image"), 256, num_workers=0, shuffle=False)

    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]
    epoch_log = open(os.path.join(load_path, f"{save_name}/epoch.log"), 'a')
    for save_iter in range(begin, end+step, step):
        print("Model building")
        net = diffNet.difGazeNet()
        net = nn.DataParallel(net)
        state_dict = torch.load(os.path.join(save_path, f"Iter_{save_iter}_{model_name}.pt"))
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
            with open(os.path.join(load_path, f"{save_name}/{save_iter}.log"), 'w') as outfile:
                outfile.write("subjcet,name,x,y,labelx,labely,error\n")
                for j, data in enumerate(dataset):
                    data["eye1"] = data["eye1"].to(device)
                    data["eye2"] = data["eye2"].to(device)
                    # data["faceImg"] = data["faceImg"].to(device)
                    # data["leftEyeImg"] = data["leftEyeImg"].to(device)
                    # data['rightEyeImg'] = data['rightEyeImg'].to(device)
                    # data['rects'] = data['rects'].to(device)
                    labels = data["label"]
                    
                    gazes = net(data["eye1"],data["eye2"])
                    
                    name = "test"
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
                        # name = names[0][k] + "," + names[1][k]
                        
                        log = [name] + gaze + label + [str(acc)]
                        
                        outfile.write(",".join(log) + "\n")
                SE_log.close()
                loger = f"[{save_iter}] Total Num: {count}, avg: {total/count} \n"
                outfile.write(loger)
                epoch_log.write(loger)
                print(loger)

