
## GC
在配置cur_dataset = "GazeCapture"的情况下，使用
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d--rdzv_endpoint=localhost:29401 train.py
能够直接训练AFF-Net的GazeCapture数据集


## MPII
在配置cur_dataset = "MPII"的情况下，使用
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d--rdzv_endpoint=localhost:29401 train.py
能够直接训练AFF-Net的MPII数据集


## 测试

### 普通测试
普通测试就是直接在测试集上运行test.py，然后对所有结果求平均



### 个性化校准测试
