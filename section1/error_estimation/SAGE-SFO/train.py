
import torch
import torch.distributed as dist
import torch.nn as nn
from dataloader import gc_reader
from dataloader import mpii_reader 
import os
import time
import sys
import config
from torch.nn.parallel import DistributedDataParallel as DDP
torch.autograd.set_detect_anomaly(True)
from sage import SAGE_SFO
import sage
from torch.cuda.amp import autocast
import logging


'''
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 train.py

'''

def trainModel():

    # 初始化进程组
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")


    """判断加载哪个数据集"""
    if config.cur_dataset == "GazeCapture":
        root_path = config.GazeCapture_root
    elif config.cur_dataset == "MPII":
        root_path = config.MPIIFaceGaze_root

    
    model_name = config.model_name
    save_path = os.path.join(config.save_path,config.cur_dataset,
                             str(config.batch_size) +"_" + str(config.epoch) + "_" + str(config.lr) + "_" + config.commit)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_path = os.path.join(root_path,"Label","model_fineture", "train")
    label_path = [os.path.join(label_path, item) for item in os.listdir(label_path)]


    if config.cur_dataset == "GazeCapture":
        dataset = gc_reader.txtload(label_path, os.path.join(root_path, "Image"), config.batch_size, shuffle=True,
                                num_workers=2)
    elif config.cur_dataset == "MPII":
        dataset = mpii_reader.txtload(label_path, os.path.join(root_path, "Image"), config.batch_size, shuffle=True,
                                num_workers=4)
    
    '''不加这个，多机分布式训练时候会出问题'''
    device_id = rank % torch.cuda.device_count()   
    ddp_model = SAGE_SFO(9).to(rank)
    device = torch.device("cuda" + ":" + str(rank))
    ddp_model = DDP(ddp_model)

    print("构建优化器")
    loss_func = nn.MSELoss()
    base_lr = config.lr
    optimizer = torch.optim.Adam(ddp_model.parameters(), base_lr, weight_decay=0.0005)

    print("训练")
    length = len(dataset)
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config.epoch + 1):
            if epoch >= config.lr_decay_start_step and epoch % config.lr_decay_cycle == 0:
                base_lr = base_lr * config.train_decay_rate
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr

            time_begin = time.time()
            for i, data in enumerate(dataset):
                optimizer.zero_grad()
                Xq = data[0]
                Xc = data[1]
                relation_vector = data[2]
                X_q = {"left_eye": Xq["left"].to(device), "right_eye": Xq["right"].to(device), "landmark": Xq["rects"].to(device)}
                X_cY_c = []
                for item in Xc:
                    X_cY_c.append({"left_eye": item["left"].to(device), "right_eye": item["right"].to(device), "landmark": item["rects"].to(device), "gaze": item["label"].to(device)})
                gaze, direction = ddp_model(X_q, X_cY_c)
                loss = sage.sage_sfo_loss(gaze.cpu(), direction.cpu(), Xq["label"], relation_vector, 0.5)

                loss.backward()
                optimizer.step()
                time_remain = (length - i - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                time_remain_total = time_remain + epoch_time * (config.epoch - epoch)
                log = f"[{epoch}/{config.epoch}]: [{i}/{length}] loss:{loss:.10f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                print(log)
                sys.stdout.flush()
                outfile.flush()

            if epoch > config.save_start_step and epoch % config.save_step == 0  and rank == 0:
                torch.save(ddp_model.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))


    dist.destroy_process_group()

if __name__ == "__main__":
    trainModel()
