from tensorboard import program

# 指定TensorBoard的日志目录，该目录包含您的JSON文件
logdir = "./log/example"

# 启动TensorBoard程序
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir,'--bind_all'])

# 启动TensorBoard的Web界面
tb.start()

# 等待TensorBoard进程完成
tb.join()
