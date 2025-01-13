import os
import torch
from utils.metric import *
import torch.nn.functional as F
import numpy as np
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def get_x(dataset):
    if 'cycDEH' in dataset:
        independent_v = 'n-th DEH time' 
    elif 'isoDEH' in dataset :
        independent_v = '1-st DEH time' 
    elif  'TPD' in dataset:
        independent_v = '1-st DEH temperature'
    return independent_v 

# def test_curve(data_dict_normalized,model,args):
#     model.eval()
#     true_list = []
#     pred_list = []
#     # mae_list  = []
#     time = []
#     xs_normalized,ys_true,orignal_xs = data_dict_normalized['X'],data_dict_normalized['y'],data_dict_normalized['orignal_x']
#     for x_normalized,y_true,orignal_x in zip(xs_normalized,ys_true,orignal_xs):
         
#         if isinstance(orignal_x, (int, float, str)):
#             time+=[orignal_x]
#         else:
#             time.append(orignal_x[-1]) 
#         # y_true = y_true.cpu()
       
#         if args.model in ['KNeighborsRegressor','DecisionTreeRegressor','SVR',
#                         'RandomForestRegressor','GradientBoostingRegressor','BayesianRidge']:
#             y_pred = model.predict(x_normalized)
#         else:
#             x_normalized = torch.tensor(x_normalized).to(args.device)
#             y_pred = model(x_normalized)
#         y_pred = y_pred.cpu().detach().squeeze()       
#         # mae_list.append(  F.mse_loss(y_true, y_pred).item() )
#         pred_list.append(y_pred.item())
#         true_list.append(y_true)
#     # avg_mae =  sum(mae_list) / len(mae_list)
#     if args.data_mode == 'multi-to-one' and args.remove_y == False:
#         time.pop(0)
#         true_list.pop()
#         pred_list.pop()
#     return pred_list,true_list,time

def test_curve(data_dict_normalized, model, args,scaler_y=None):
    model.eval()
    true_list = []
    pred_list = []
    time = []

    xs_normalized = data_dict_normalized['X']
    ys_true = data_dict_normalized['orignal_x_df']['DEH value']
    orignal_xs = data_dict_normalized['orignal_x_df'][args.independent_v]
    
    batch_size = args.batch_size

    # 按批次遍历数据
    for i in range(0, len(xs_normalized), batch_size):
        batch_xs_normalized = xs_normalized[i:i + batch_size]
        batch_ys_true = ys_true[i:i + batch_size]
        batch_orignal_xs = orignal_xs[i:i + batch_size]
        for orignal_x in batch_orignal_xs:
            if isinstance(orignal_x, (int, float, str)):
                time.append(orignal_x)
            else:
                time.append(orignal_x[-1])

        if args.model in ['KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR',
                        'RandomForestRegressor', 'GradientBoostingRegressor', 'BayesianRidge']:
            y_pred = model.predict(batch_xs_normalized)
        else:
            batch_xs_normalized = torch.tensor(batch_xs_normalized).to(args.device)
            y_pred = model(batch_xs_normalized)
        # y_pred = torch.from_numpy(scaler_y.inverse_transform(y_pred.cpu().detach().numpy()))
        # y_pred = y_pred.squeeze().tolist()
        y_pred = y_pred.cpu().detach().squeeze().tolist()
        if isinstance(y_pred , float):
            pred_list.extend([y_pred])
        else:
            pred_list.extend(y_pred)
        true_list.extend(batch_ys_true)
    if args.data_mode == 'multi-to-one' and args.remove_y == False:
        time.pop(0)
        true_list.pop()
        pred_list.pop()
    return pred_list,true_list,time



