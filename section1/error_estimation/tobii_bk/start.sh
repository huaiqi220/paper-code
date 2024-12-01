cd /home/zhuzi/code/2024-5/paper-code/Learning_to_Personalize_in_Appearance-Based_Gaze_Tracking
conda activate gaze10
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29401 main.py
