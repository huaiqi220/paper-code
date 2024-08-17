## 毕业设计
### 小论文代码复现及实验结果

当前进度：

模型基本设计完成

Mobile-Gaze模型2D PoG MPII数据集调通

MPII数据集达到 < 2cm训练集error，还没做测试集

Mobile-Gaze模型2D PoG GC数据集调通，已经在得到初步结果

后一阶段任务——看下这heatmap咋回事，怎么训练都不收敛

Mobile-Gaze模型heatmap训练问题仍未解决


Mobile-Gaze如果想换backbone
直接修改Mobile_Gaze.py中的self.face_cnn 那块
改的时候别忘了把config里的名也改了，不然存错了位置

2024年8月16日01点33分补充——
使用sum模式的def heatmap_loss(y_true, y_pred):训练GC hm，反而并没有出现公司服务器的的什么梯度爆炸

明天上班分析一下原因
