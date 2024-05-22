from Model import baseline_cnn_model
import torch
import torch.distributed as dist
import torch.nn as nn

import yaml
import dataloader.o_reader as reader
import os
import time
import sys

from torch.nn.parallel import DistributedDataParallel as DDP


torch.autograd.set_detect_anomaly(True)

from torch.cuda.amp import autocast


'''
在未改变标签的原始GC数据集上,直接通过右眼预测baseCNN作为基准方法
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train-basecnn-ddp.py

'''

def trainModel():
    # 初始化进程组
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    oconfig  = yaml.safe_load(open("config.yaml"))
    config = yaml.safe_load(open("config.yaml"))
    config = config["train"]
    path = config["data"]["path"]
    model_name = config["save"]["model_name"]

    save_path = os.path.join(config["save"]["save_path"], "checkpoint", "base_cnn",str(config["params"]["batch_size"]) +"_" + str(config["params"]["epoch"]) + "_" + str(config["params"]["lr"]) + "_amp")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("读取数据")
    if rank == 0:
        with open(os.path.join(save_path, 'cur_config.yaml'), 'w') as file:
            yaml.dump(oconfig, file)

    path1 = os.path.join(path, "Label", "train")
    path1 = [os.path.join(path1, item) for item in os.listdir(path1)]

    dataset = reader.txtload(path1, os.path.join(path, "Image"), config["params"]["batch_size"], shuffle=True,
                             num_workers=16)

    ddp_model = baseline_cnn_model.oriGazeNet().to(rank)
    device = torch.device("cuda" + ":" + str(rank))
    ddp_model = DDP(ddp_model)

    print("构建优化器")
    loss_op = nn.SmoothL1Loss()
    base_lr = config["params"]["lr"]
    cur_step = 0
    decay_steps = config["params"]["decay_step"]
    optimizer = torch.optim.Adam(ddp_model.parameters(), base_lr, weight_decay=0.0005)

    print("训练")
    length = len(dataset)
    cur_decay_index = 0
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
                base_lr = base_lr * config["params"]["decay"]
                cur_decay_index = cur_decay_index + 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr

            time_begin = time.time()
            for i, data in enumerate(dataset):
                optimizer.zero_grad()
                with autocast():
                    data["left"] = data["left"].to(device)
                    label = data["label"].to(device)
                    gaze = ddp_model(data["left"])
                    loss = loss_op(gaze, label)

                loss.backward()
                optimizer.step()
                time_remain = (length - i - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                time_remain_total = time_remain + epoch_time * (config["params"]["epoch"] - epoch)
                log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                print(log)
                sys.stdout.flush()
                outfile.flush()

            if epoch % config["save"]["step"] == 0 and rank == 0:
                torch.save(ddp_model.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))


    dist.destroy_process_group()

if __name__ == "__main__":
    trainModel()

# torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 train-dif-ddp.py