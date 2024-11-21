import os
import sys
import config
import random
from dataloader import gc_reader
from dataloader import mpii_reader
import torch
from model.CGES import mobile_gaze_2d as model
from torch import nn
import time
import math
import itertools
from tqdm import tqdm
from model import STE
from util import testtools




def generate_binary_vectors(k):
    # 使用 itertools.product 生成所有二进制向量
    binary_vectors = list(itertools.product([0, 1], repeat=k))
    # 转换成 PyTorch 张量
    binary_tensors = torch.tensor(binary_vectors, dtype=torch.float32)  # 如果需要浮点张量
    return binary_tensors


def dis(p1, p2):
    return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))


def test_func(name, calimodel, dataset,save_path,rank,cali_test,cali_vec):
    # print(calimodel)

    if cali_test:
        print("这次是校准测试")
        save_path = os.path.join(save_path,"calibration_test")
    else:
        print("这次是原始测试")
        save_path = os.path.join(save_path,"origin_test")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    device = torch.device("cuda" + ":" + str(rank))
    calimodel.to(device)
    calimodel.eval()
    total = 0
    count = 0
    with torch.no_grad():
        with open(os.path.join(save_path, "error.log"), 'w') as outfile:
            outfile.write("subjcet,name,x,y,labelx,labely,error\n")
            for j, data in enumerate(dataset):
                data["face"] = data["face"].to(device)
                data["left"] = data["left"].to(device)
                data["right"] = data["right"].to(device)
                data["grid"] = data["grid"].to(device)
                data["rects"] = data["rects"].to(device)
                data["name"] = data["name"].to(device)
                labels = data["label"]
                if cali_test == False:
                    cali_vec = torch.zeros(1,config.k).to(device)
                    # print(cali_vec.shape)
                else:    
                    cali_vec = cali_vec.to(device)
                # data["poglabel"] = data["poglabel"].to(device)
                gazes = calimodel(data["face"], data["left"], data["right"], data["grid"], data["name"],"inference",cali_vec)   

                print(f'\r[Batch : {j}]', end='')
                #print(f'gazes: {gazes.shape}')
                for k, gaze in enumerate(gazes):
                    #print(f'gaze: {gaze}')
                    gaze = gaze.cpu().detach()
                    count += 1
                    acc = dis(gaze, labels[k])
                    total += acc
                    gaze = [str(u) for u in gaze.numpy()]
                    label = [str(u) for u in labels.numpy()[k]]
                    log = [name] + gaze + label + [str(acc)]
                    
                    outfile.write(",".join(log) + "\n")
            loger = f"[{name}] Total Num: {count}, avg: {total/count} \n"
            outfile.write(loger)
            print(loger)



def float_cali_func(name,calimodel,dataset,save_path,rank):
    # 这个函数用来做浮点型校准向量，用于对比浮点类型与二进制类型
    #   校准向量的性能差异
    # 返回校准好的模型和校准参数
    device = torch.device("cuda" + ":" + str(rank))
    calimodel.to(device)
    calimodel.train()

    # 所有参数全部固定。
    for param in calimodel.parameters():
        param.requires_grad = False  
    
    loss_func = nn.MSELoss()
    lr = config.cali_lr

    # cali_vec = torch.randn(1, config.k,requires_grad=True).to(device)
    cali_vec = torch.randn(1, config.k, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([cali_vec], lr=lr)
    save_path = os.path.join(save_path,"train_log")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    length = len(dataset)
    with open(os.path.join(save_path, "train_loss.log"), 'w') as outfile:
        for epoch in range(1, config.cali_epoch + 1):
            time_begin = time.time()
            for i, data in enumerate(dataset):
                optimizer.zero_grad()
                with torch.amp.autocast("cuda"):
                    data["face"] = data["face"].to(device)
                    data["left"] = data["left"].to(device)
                    data["right"] = data["right"].to(device)
                    data["grid"] = data["grid"].to(device)
                    data["rects"] = data["rects"].to(device)
                    data["label"] = data["label"].to(device)
                    data["name"] = data["name"].to(device)
                    # data["poglabel"] = data["poglabel"].to(device)
                    gaze_out = calimodel(data["face"], data["left"], data["right"], data["grid"], data["name"],"inference",cali_vec)
                    loss = loss_func(gaze_out, data["label"])        

                loss.backward()
                optimizer.step()
                time_remain = (length - i - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                epoch_time = (length - 1) * ((time.time() - time_begin) / (i + 1)) / 3600
                time_remain_total = time_remain + epoch_time * (config.cali_epoch - epoch)
                log = f"[{epoch}/{config.cali_epoch}]: [{i}/{length}] loss:{loss:.10f} lr:{lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                print(log)
                sys.stdout.flush()
                outfile.flush()
    calimodel.cpu()
    cali_vec = cali_vec.cpu()
    return calimodel,cali_vec           



def binary_cali_func(name, calimodel, dataset, save_path, rank):
    # This function is used to find the optimal binary calibration vector
    print("Calculating calibration vector")
    save_path = os.path.join(save_path, "compute_cali_vec")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gaze_feature = None
    gaze_labels = None
    binary_tensors = generate_binary_vectors(config.k)
    cur_min_avg = float('inf')
    min_vec = None
    device = torch.device("cuda:" + str(rank))
    calimodel.to(device)
    calimodel.eval()
    cur_vec = None
    total = 0
    count = 0

    with torch.no_grad():
        # First pass to extract all gaze features from the dataset
        with open(os.path.join(save_path, "compute_cali_vec.log"), 'w') as outfile:
            for j, data in enumerate(dataset):
                data["face"] = data["face"].to(device)
                data["left"] = data["left"].to(device)
                data["right"] = data["right"].to(device)
                data["grid"] = data["grid"].to(device)
                data["rects"] = data["rects"].to(device)
                data["name"] = data["name"].to(device)
                labels = data["label"].to(device)

                # Get gaze feature from the model
                gaze_tensor = calimodel.getTheGazeFeature(
                    data["face"], data["left"], data["right"], data["grid"], data["name"], "inference", cur_vec
                )

                if gaze_feature is None:
                    gaze_feature = gaze_tensor
                    gaze_labels = labels
                else:
                    gaze_feature = torch.cat((gaze_feature, gaze_tensor), 0)
                    gaze_labels = torch.cat((gaze_labels, labels), 0)
        print("特征提取完毕，开始计算校准向量")
        # Second pass to compute average error for each binary tensor
        for vec in tqdm(binary_tensors, desc="Binary Calibration Compute"):
            cur_vec = vec.to(device)
            cur_vec = cur_vec.expand(gaze_feature.shape[0], -1)
            gazes = calimodel.computeGaze(cur_vec,gaze_feature)

            total = 0
            count = 0

            for k, gaze in enumerate(gazes):
                #print(f'gaze: {gaze}')
                gaze = gaze.cpu().detach()
                count += 1
                acc = dis(gaze, gaze_labels[k])
                total += acc


            avg_error = total / count

            if avg_error < cur_min_avg:
                min_vec = vec.cpu().clone()
                cur_min_avg = avg_error

    print("Best calibration vector for this subject is:")
    print(min_vec)
    print("The best calibration result is:")
    print(cur_min_avg)
    return min_vec




# def binary_cali_func(name,calimodel,dataset,save_path,rank):
#     # 这个函数用来做二进制校准向量，用于对比浮点类型与二进制类型
#     #   校准向量的性能差异
#     print("计算校准向量")
#     save_path = os.path.join(save_path,"compute_cali_vec")
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     gaze_feature = None
#     binary_tensors = generate_binary_vectors(config.k)
#     cur_min_avg = 1000
#     min_vec = None
#     device = torch.device("cuda" + ":" + str(rank))
#     calimodel.to(device)
#     calimodel.eval()
#     cur_vec = None
#     total = 0
#     count = 0
#     with torch.no_grad():
#         with open(os.path.join(save_path, "compute_cali_vec.log"), 'w') as outfile:
#             # outfile.write("subjcet,name,x,y,labelx,labely,error\n")
#             for j, data in enumerate(dataset):
#                 data["face"] = data["face"].to(device)
#                 data["left"] = data["left"].to(device)
#                 data["right"] = data["right"].to(device)
#                 data["grid"] = data["grid"].to(device)
#                 data["rects"] = data["rects"].to(device)
#                 data["name"] = data["name"].to(device)
#                 # labels = data["label"]
#                 # data["poglabel"] = data["poglabel"].to(device)
#                 # gazes = calimodel(data["face"], data["left"], data["right"], data["grid"], data["name"],"inference",cur_vec)   
#                 gaze_tensor = calimodel.getTheGazeFeature(data["face"], data["left"], data["right"], data["grid"], data["name"],"inference",cur_vec)
#                 # tensor_list.append(gaze_tensor)
#                 # print(gaze_tensor.shape)
#                 if gaze_feature is None:
#                     gaze_feature = gaze_tensor
#                 else:
#                     gaze_feature = torch.cat((gaze_feature,gaze_tensor),0)

#     for vec in tqdm(binary_tensors,desc="Binary Cali Compute"):
        
#         cur_vec = vec.to(device)
#         gazes = calimodel.computeGaze(gaze_feature,cur_vec)
    #                 # print(f'\r[Batch : {j}]', end='')
    #                 #print(f'gazes: {gazes.shape}')
    #                 for k, gaze in enumerate(gazes):
    #                     #print(f'gaze: {gaze}')
    #                     gaze = gaze.cpu().detach()
    #                     count += 1
    #                     acc = dis(gaze, labels[k])
    #                     total += acc
    #                     gaze = [str(u) for u in gaze.numpy()]
    #                     label = [str(u) for u in labels.numpy()[k]]
    #                     log = [name] + gaze + label + [str(acc)]
                        
    #                     outfile.write(",".join(log) + "\n")
    #             loger = f"[{name}] Total Num: {count}, avg: {total/count} \n"
    #             outfile.write(loger)
    #             if(total / count < cur_min_avg):
    #                 min_vec = cur_vec
    #                 cur_min_avg = min(cur_min_avg,total/count)
    #             # print(loger)
    # print("这个人的最好的校准向量是")
    # print(min_vec)
    # print("最好的校准样本校准结果是")
    # print(cur_min_avg)
    # return min_vec




def cali_test_func(root_path, label):
    rank = config.cur_rank
    k = config.k
    cur_id = label.split("/")[-1].split(".")[0]
    cali_folder = os.path.join(config.test_save_path,config.cur_dataset,config.commit, "cali_num_" + str(config.cali_image_num) +"_" + str(config.cali_last_layer) + "_" + str(config.cali_lr) + "_" + str(config.k), cur_id)

    all_label = []
    with open(label, "r") as f:
        all_label = f.readlines()
        all_label.pop(0)

    selected_cali_lines, remaining_lines =  testtools.select_by_quadrants(all_label,int(config.cali_image_num / 4) + 1)
    # 部分用户采集图片很少


    if len(remaining_lines) < 10  or  len(selected_cali_lines) < 10:
        print("该用户数据较少，跳过测试")
        return 
    
    if config.cur_dataset == "GazeCapture":
        all_test_dataset = gc_reader.calitxtload(all_label,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)
        cali_train_dataset = gc_reader.calitxtload(selected_cali_lines,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)
        cali_test_dataset = gc_reader.calitxtload(remaining_lines,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)

    if config.cur_dataset == "MPII":
        all_test_dataset = mpii_reader.calitxtload(all_label,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)
        cali_train_dataset = mpii_reader.calitxtload(selected_cali_lines,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)
        cali_test_dataset = mpii_reader.calitxtload(remaining_lines,os.path.join(root_path,"Image"),config.cali_batch_size,True,8,True)

    test_model_path = config.test_model_path
    calimodel = model(config.hm_size, 12, 25 * 25)
    statedict = torch.load(test_model_path)
    new_state_dict = {}
    for key, value in statedict.items():
    # 如果 key 以 "module." 开头，则去掉这个前缀
        new_key = key[7:]
        new_state_dict[new_key] = value
    calimodel.load_state_dict(new_state_dict)
    # print(calimodel.cali_vectors)

    id_path = "/home/hi/zhuzi/data/GCOutput/Label/model_fineture/train"
    file_list = os.listdir(id_path)
    file_list = [file.split(".")[0] for file in file_list]
    lines = calimodel.cali_vectors
    with open("log.txt", "w") as f:
        f.write("2D Tensor:\n")
        for id in file_list:
            id = int(id)
            line = STE.BinarizeSTE_origin.apply(lines[id])
            # sigmoid_output = torch.sigmoid(lines[id])
            # binary_output = (sigmoid_output > 0.5).float()
            f.write(str(id) + " " + str(line) + '\n')
        # f.write(str())  # 将 Tensor 转为 numpy 数组并写入
        f.write("\n")
    f.close()
    device = torch.device("cuda" + ":" + str(rank))   

    # 首先测试这个校准模型在校准数据集上的未校准性能
    test_func(cur_id,calimodel,cali_test_dataset,cali_folder,rank,False,None)

    if config.cali_vector_type == "float32":
        calimodel,cali_vec = float_cali_func("name",calimodel,cali_train_dataset,cali_folder,rank)
    else:
        pass
        cali_vec = binary_cali_func("name",calimodel,cali_train_dataset,cali_folder,rank)

    # 测试计算好的函数
    print(cali_vec)
    test_func(cur_id,calimodel,cali_test_dataset,cali_folder,rank,True,cali_vec)


# 主函数
if __name__ == "__main__":
    if config.cur_dataset == "GazeCapture":
        root_path = config.GazeCapture_root
    elif config.cur_dataset == "MPII":
        root_path = config.MPIIFaceGaze_root

    test_label_path = os.path.join(root_path,"Label","model_fineture","test")
    label_list = [os.path.join(test_label_path, item) for item in os.listdir(test_label_path)]
    for label in tqdm(label_list):
        res = cali_test_func(root_path, label)
    # binary_cali_func(None,None,None,None,None)

    