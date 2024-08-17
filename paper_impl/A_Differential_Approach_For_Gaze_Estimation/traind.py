import os
import time
import sys
import yaml
import argparse
import torch
import A_Differential_Approach_For_Gaze_Estimation.dataloader.reader as reader
import model
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# 初始化分布式训练环境


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--local-rank', type=int, help='local rank for dist')
    args = parser.parse_args() 

    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '7810'
    world_size = 4
    local_rank = args.local_rank
    os.environ['RANK'] = str(local_rank)
    print("1")
    
    dist.init_process_group(backend='nccl',init_method='env://', world_size=world_size,rank=local_rank)
    torch.cuda.set_device(local_rank)
    print("2")
    config = yaml.safe_load(open("config.yaml"))
    config = config["train"]
    path = config["data"]["path"]
    model_name = config["save"]["model_name"]

    save_path = os.path.join(config["save"]["save_path"], "checkpoint")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 调整设备分配以供分布式训练
    device = torch.device("cuda", torch.cuda.current_device())

    print("读取数据")
    path1 = os.path.join(path, "Label", "train")
    path1 = [os.path.join(path1, item) for item in os.listdir(path1)]

    dataset = reader.txtload(path1, os.path.join(path, "Image"), config["params"]["batch_size"], shuffle=True,
                             num_workers=16)

    # 使用DistributedSampler包装数据集
    distributed_sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=config["params"]["batch_size"], sampler=distributed_sampler)

    print("构建模型")
    net = model.model()
    net = net.cuda()

    net.train()
    net = nn.parallel.DistributedDataParallel(net)
    net.to(device)



    print("构建优化器")
    loss_op = nn.SmoothL1Loss().cuda()
    base_lr = config["params"]["lr"]
    cur_step = 0
    decay_steps = config["params"]["decay_step"]
    optimizer = torch.optim.Adam(net.parameters(), base_lr, weight_decay=0.0005)

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
                data["face"] = data["face"].to(device)
                data["left"] = data["left"].to(device)
                data['right'] = data['right'].to(device)
                data['rects'] = data['rects'].to(device)
                label = data["label"].to(device)

                gaze = net(data["left"], data["right"], data['face'], data['rects'])
                loss = loss_op(gaze, label) * 4

                optimizer.zero_grad()
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

            if epoch % config["save"]["step"] == 0 and local_rank == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))



                # CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 traind.py
