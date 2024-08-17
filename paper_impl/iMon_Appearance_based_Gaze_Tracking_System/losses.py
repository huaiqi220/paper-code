import torch
import time
import cv2
import numpy as np
import pandas as pd
from config import sage_model_config as config
import np2torchdata as n2t

def simple_test_euclidean(ddp_model,dots_val,regions_val,rank):
    '''Euclidean test'''
    # t = time.time()
    total_avg_loss = 0.0
    total_samples = 0
    ds = n2t.get_torch_dataset(dots_val, regions_val, shuffle=False,num_workers=8)
    for i, data in enumerate(ds):
        preds = ddp_model(data["left"].to(rank), data["right"].to(rank), data['eyelandmark'].to(rank), data['orientation'].to(rank))
        cur_loss = euclidean_loss(data["label"].to(rank), preds).sum()
        total_avg_loss += cur_loss.item()
        total_samples += len(data["label"])
    print('Total samples:', total_samples, ' Avg Error:', total_avg_loss/total_samples)
    # print('Total time:', time.time() - t)

def full_test_euclidean(ddp_model,dots_val,regions_val,df_info_val,dots_train,regions_train,df_info_train,rank):
    ds = n2t.get_torch_dataset(dots_val, regions_val, shuffle=False,num_workers=8)
    all_outputs = []
    name_list = []
    for i, data in enumerate(ds):
        preds = ddp_model(data["left"].to(rank), data["right"].to(rank), data['eyelandmark'].to(rank), data['orientation'].to(rank))
        all_outputs.append(preds.detach().cpu())
        name_list.append(data["name"].detach())
    all_outputs = torch.cat(all_outputs, dim=0)
    name_list = torch.cat(name_list, dim=0)
    








def euclidean_loss(y_true, y_pred):
    # print(y_true.shape)
    # print(y_pred.shape)
    return torch.sqrt(torch.sum(torch.square(y_pred - y_true), dim=-1, keepdim=True))

def show_heatmap(y_true, y_pred):
    save_img_true = y_true[0,:,:,:]
    save_img_pred = y_pred[0,:,:,:]
    true_np = save_img_true.detach().cpu().numpy()
    true_np = np.transpose(true_np, (1, 2, 0))
    pred_np = save_img_pred.detach().cpu().numpy()
    pred_np = np.transpose(pred_np, (1, 2, 0))
    print(true_np.shape)
    print(pred_np.shape)
    name = str(time.time())
    cv2.imwrite(f"./hm_debug/"+ name + ".png", true_np*255)
    cv2.imwrite(f"./hm_debug/"+ name + "_pred.png", pred_np*255)


def heatmap_loss(y_true, y_pred):
    if torch.sum(y_pred) == 0:
        return torch.tensor(1.0)
    else:
        # y_pred /= torch.sum(y_pred)
        # y_true /= torch.sum(y_true)
        # return torch.sum(torch.abs(y_pred - y_true)) / 2.0
        normalized_y_pred = y_pred / torch.sum(y_pred)
        normalized_y_true = y_true / torch.sum(y_true)
        return torch.sum(torch.abs(normalized_y_pred - normalized_y_true)) / 2.0




def get_pred_scope(df_info_train, regions_train, dots_train, df_info_test):
    # Get scope of each test device
    unique_devices = pd.unique(df_info_test['DeviceName'])
    pred_scopes = []
    for device in unique_devices:  # each device has different dimension measurement
        train_subjectIDs = list(df_info_train.loc[df_info_train['DeviceName'] == device,
                                                  'subjectID'])
        orientations = [1, 3, 4] if config.mobile else [1, 2, 3, 4]
        for ori in orientations:
            train_indices = np.where((regions_train[:, 24] == ori) &
                                     (np.isin(regions_train[:, 0], train_subjectIDs)))[0]
            pred_max = np.max(dots_train[train_indices, -3: -1], axis=0)
            pred_min = np.min(dots_train[train_indices, -3: -1], axis=0)
            pred_scopes.append([device, ori, pred_max, pred_min])

    df_pred_scopes = pd.DataFrame(pred_scopes)
    df_pred_scopes.columns = ['device', 'orientation', 'max', 'min']
    df_pred_scopes = df_pred_scopes.set_index(['device', 'orientation'])

    return df_pred_scopes
if __name__ == "__main__":
    
    test1 = torch.ones(10,2)
    test2 = torch.rand(10,2)
    print(euclidean_loss(test1,test2))

    test3 = torch.rand(10,64,64)
    test4 = torch.rand(10,64,64)
    print(heatmap_loss(test3,test4))