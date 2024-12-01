#!/bin/bash
set -e  # 遇到错误时停止脚本执行
eval "$(conda shell.bash hook)" 
conda activate torch211

cd /home/hi/zhuzi/paper-code/section2/MobileGaze_float/

# 定义运行函数，捕获错误
run_training() {
    local calik=$1
    torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 train.py --calik "$calik" || {
        echo "Error: Training failed for calik=$calik" >&2
        exit 1
    }
}

# 执行不同的 calik 参数
for calik in 12 10 8 6 4; do
    echo "Running training with calik=$calik..."
    run_training "$calik"
done

echo "All training jobs completed successfully!"