

import model

import torch
import coremltools as ct


# 初始化模型并设置为评估模式
model = model.model()
# model.load_state_dict(torch.load("checkpoint/Iter_16_AFF-Net.pt"))
statedict = torch.load("checkpoint/Iter_16_AFF-Net.pt")
new_state_dict = {}
for key, value in statedict.items():
# 如果 key 以 "module." 开头，则去掉这个前缀
    new_key = key[7:]
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)
model.eval()

# 创建一个虚拟的输入张量，大小需要和实际输入数据一致
# example_input = torch.rand(1, 10)
feature = {"faceImg": torch.zeros( 224, 224, 3), "leftEyeImg": torch.zeros( 112, 112, 3),
            "rightEyeImg": torch.zeros( 112, 112, 3), "faceGridImg": torch.zeros(12),
            "label": torch.zeros(1, 2), "frame": "test.jpg"}
# input_list = [feature["leftEyeImg"], feature["rightEyeImg"], feature["faceImg"], feature["faceGridImg"]]



# 使用 TorchScript 进行模型跟踪
traced_model = torch.jit.trace(model, (feature["leftEyeImg"], feature["rightEyeImg"], feature["faceImg"], feature["faceGridImg"]))

# 将 traced_model 转换为 Core ML 格式
# mlmodel = ct.convert(traced_model, inputs=[ct.TensorType(shape=input_list.shape)])
mlmodel = ct.convert(
    traced_model, 
    inputs=[
        # ct.TensorType(shape=(10, 3, 112, 112)), ct.TensorType(shape=(10, 3, 112, 112)), ct.TensorType(shape=(10, 3, 224, 224)), ct.TensorType(shape=(10, 12))
        ct.TensorType(name="leftEyeImg", shape=feature["leftEyeImg"].shape),
        ct.TensorType(name="rightEyeImg", shape=feature["rightEyeImg"].shape),
        ct.TensorType(name="faceImg", shape=feature["faceImg"].shape),
        ct.TensorType(name="faceGridImg", shape=feature["faceGridImg"].shape),        
        ]

    )

# 保存 Core ML 模型
mlmodel.save("aff_net_ma.mlpackage")
