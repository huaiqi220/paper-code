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
from model import crossNet

torch.autograd.set_detect_anomaly(True)


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"


'''
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 train_sin.py

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
    save_path = os.path.join(config.save_path, config.cur_dataset, config.commit,
                             f"{config.batch_size}_{config.epoch}_{config.lr}")
    os.makedirs(save_path, exist_ok=True)

    label_path = os.path.join(root_path,"Label","model_fineture", "train")
    label_path = [os.path.join(label_path, item) for item in os.listdir(label_path)]


    if config.cur_dataset == "GazeCapture":
        dataset = gc_reader.txtload(label_path, os.path.join(root_path, "Image"), config.batch_size, shuffle=True,
                                num_workers=4)
    elif config.cur_dataset == "MPII":
        dataset = mpii_reader.txtload(label_path, os.path.join(root_path, "Image"), config.batch_size, shuffle=True,
                                num_workers=4)
    
    '''不加这个，多机分布式训练时候会出问题'''
    device_id = rank % torch.cuda.device_count()   
    device = torch.device(f"cuda:{device_id}")
    ddp_model = DDP(crossNet.SingleNNPoG().to(device))

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
                with torch.amp.autocast("cuda"):
                    data["left"] = data["left"].to(device)
                    data["right"] = data["right"].to(device)
                    data["label"] = data["label"].to(device)
                    gaze_out = ddp_model(data["left"],data["right"])
                    loss = loss_func(gaze_out, data["label"])
  

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
