
import torch
import torch.distributed as dist
import torch.nn as nn

from dataloader import gc_loader as reader
import os
import time
import sys
from util import loss_func
import config_bk

from util import loss_func
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)

from model import Mobile_Gaze
from model import CGES
# from torch.cuda.amp import autocast
from torch.amp import autocast
from util import htools
import logging

'''
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 main.py

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 main.py

'''

def trainModel():

    # 初始化进程组
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    root_path = config_bk.GazeCapture_root
    model_name = config_bk.model_name

    save_path = os.path.join(config_bk.save_path,
                             str(config_bk.batch_size) +"_" + str(config_bk.epoch) + "_" + str(config_bk.lr))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_path = os.path.join(root_path,"Label","model_fineture", "train")
    label_path = [os.path.join(label_path, item) for item in os.listdir(label_path)]

    dataset = reader.txtload(label_path, os.path.join(root_path, "Image"), config_bk.batch_size, shuffle=True,
                             num_workers=8)
    ddp_model = CGES.mobile_gaze_2d(config_bk.hm_size, 12, 25 * 25).to(rank)
    # ddp_model = Mobile_Gaze.mobile_gaze_hm(config.hm_size, 8, 25 * 25).to(rank)
    device = torch.device("cuda" + ":" + str(rank))
    ddp_model = DDP(ddp_model)

    print("构建优化器")
    cali_loss = nn.MSELoss()
    hm_loss = loss_func.WeightedL1Loss(config_bk.hm_loss_alpha)
    d2_loss = nn.MSELoss()
    base_lr = config_bk.lr
    optimizer = torch.optim.SGD(ddp_model.parameters(), base_lr, weight_decay=0.0005)

    print("训练")
    length = len(dataset)
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config_bk.epoch + 1):
            if epoch >= config_bk.lr_decay_start_step and epoch % config_bk.lr_decay_cycle == 0:
                base_lr = base_lr * config_bk.train_decay_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr

            time_begin = time.time()
            for i, data in enumerate(dataset):
                optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    data["face"] = data["face"].to(device)
                    data["left"] = data["left"].to(device)
                    data["right"] = data["right"].to(device)
                    data["grid"] = data["grid"].to(device)
                    # data["cali"] = data["cali"].to(device)
                    data["label"] = data["label"].to(device)
                    data["poglabel"] = data["poglabel"].to(device)
                    data["name"] = data["name"].to(device)
                    gaze_out = ddp_model(data["face"], data["left"], data["right"], data["grid"], data["name"])
                    loss = nn.MSELoss()(gaze_out, data["poglabel"])
                    # c, gaze_out = ddp_model(data["face"], data["left"], data["right"], data["grid"], data["cali"])
                    # loss = loss_func.heatmap_loss(gaze_heatmap, data["label"]) + config.loss_alpha * cali_loss(c,data["cali"])
                    # loss = hm_loss(gaze_heatmap, data["label"]) + config.loss_alpha * cali_loss(c, data["cali"])
                    # loss = d2_loss(gaze_out,data["poglabel"]) + config.loss_alpha * cali_loss(c, data["cali"])
                    # loss = loss_func.heatmap_loss(data["label"],gaze_out) + config.loss_alpha * cali_loss(c, data["cali"])

                    """绘制heatmap，仅heatmap输出才解注释"""
                    # file_name = str(time.time())
                    # htools.save_first_image(gaze_out,file_name + "_out.png" )
                    # htools.save_first_image(data["label"], file_name + "_label.png" )
                    


                loss.backward()
                optimizer.step()
                time_remain = (length - i - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                time_remain_total = time_remain + epoch_time * (config_bk.epoch - epoch)
                log = f"[{epoch}/{config_bk.epoch}]: [{i}/{length}] loss:{loss:.10f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                print(log)
                sys.stdout.flush()
                outfile.flush()

            if epoch >= config_bk.save_start_step and epoch % config_bk.save_step == 0  and rank == 0:
                torch.save(ddp_model.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))


    dist.destroy_process_group()

if __name__ == "__main__":
    trainModel()
