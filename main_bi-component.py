import pandas as pd
from model.attention import AttentionModel, TransformerForRegression, TransformerRegressor, bi_Model
from model.sequence_model import SimpleNet
from utils.data_process import data_split,data_split_sequence, generate_test_data, get_test_data, get_types, ordered_set, process_new_type, process_new_type_for_generated_data, split_nan_df
from utils.load_para import parse
import random
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from utils.metric import *
from utils.utils import check_path, get_x,  test_curve
from utils.vis import pie, plot_df,bar, plot_scatter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import logging
from model.regression_model import *
from model.sklearn_model import *

def train_model(dataloader1,dataloader2,model,model_type,optimizer):
    sum_loss = 0
    model.train()
    num_data = 0
    for data1,data2 in zip(dataloader1,dataloader2):        
        X1,y_true,_ = data1
        X2,_,_ = data2
        # if len(y_true) == args.batch_size:
        optimizer.zero_grad()
        y_pred = model(X1,X2)
        criterion = nn.MSELoss()
        # get loss
        y_true =y_true.unsqueeze(-1) 
        loss = criterion(y_pred,y_true)
        if model_type == 'LassoRegressor':
            loss += model.l1_penalty()
        elif model_type == 'RidgeRegressor':
            loss += model.l2_penalty()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        num_data += len(y_true)
    avg_loss = sum_loss/num_data
    return avg_loss,model


def test_model(dataloader1,dataloader2,model):
    model.eval()
    y_pred = torch.tensor([])
    y_true = torch.tensor([])
    
    with torch.no_grad():
        for data1,data2 in zip(dataloader1,dataloader2):        
            X1,y,_ = data1
            X2,_,_ = data2
            y_hat =  model(X1,X2).cpu().detach().squeeze()
            if y_hat.dim() == 0:
                y_hat = y_hat.unsqueeze(0)
            # 将 y_hat 和 y 追加到列表中
            y_pred = torch.cat((y_pred,y_hat))
            y_true = torch.cat((y_true,y.cpu().detach()))
    
    # 将列表中的张量拼接成一个张量
    # y_pred = torch.cat(y_pred, dim=0)
    # y_true = torch.cat(y_true, dim=0)
        
    # combined_y_pred = torch.stack(y_pred) #torch.cat(y_pred, dim=0) 
    # combined_y_true = torch.stack(y_true).squeeze()  #torch.cat(y_true, dim=0)    
    results = all_metric(y_pred,y_true)
    # avg_mae =  sum(mae_list) / len(mae_list)
    return results,y_pred,y_true 


def data_process(data_list,name_map):
    dfs = pd.DataFrame()
    empty_row = pd.Series({}, name='EmptyRow')
    name_list = []
    flag = []
    for sheet, data in data_list.items():
        columns_with_underscore = [col for col in data.columns if '_' in col]
        # data[columns_with_underscore] = 0   # bu
        data= data.drop(columns=columns_with_underscore)
        data.columns = [col.replace('_', '') for col in data.columns] # remove '_' in the name of columns
        data.rename(columns=name_map, inplace=True)
        data = data.append(empty_row,ignore_index=True)
        dfs =  pd.concat([dfs,data], ignore_index=True)
        name_list =  data.columns.tolist()
        flag.append(len(dfs))
    if args.data_mode == 'one-to-one':
        name_list = name_list[:-1]
    types = get_types(dfs)
    # one_hot_encoded = pd.get_dummies(dfs['types'], prefix='types')
    # dfs.drop('types', axis=1, inplace=True)  # del Category colunm
    # clearn_dfs = pd.concat([one_hot_encoded,dfs], axis=1)
     # instead of one-hot
    # if 'bi' in args.dataset:
    #     df_type = pd.read_excel(f'data/bi-types.xlsx')
    # else:
    #     df_type = pd.read_excel(f'data/types.xlsx')
    df_type = pd.read_excel(f'data/types2.xlsx').drop(columns=['classes'])
    last = dfs.columns[-1]
    clearn_dfs = process_new_type(df_type,dfs,last)
    
    args.num_sys = len(clearn_dfs.columns)-len(dfs.columns) # number of system 
    
    # normalization
    scaler = MinMaxScaler()
    df_normalized = clearn_dfs.copy()  
    for column in clearn_dfs.columns[:-1]:
        clearn_dfs[column].replace('\\', 0, inplace=True)
        df_normalized[column] = scaler.fit_transform(clearn_dfs[[column]])
        
    df_normalized1 = df_normalized.iloc[: flag[0]]  # 前 n 行
    df_normalized2 = df_normalized.iloc[ flag[0]:]
    df_normalized2 = df_normalized2.reset_index(drop=True)
    clearn_dfs1 = clearn_dfs.iloc[: flag[0]]  # 前 n 行
    clearn_dfs2 = clearn_dfs.iloc[ flag[0]:]  
    clearn_dfs2 = clearn_dfs2.reset_index(drop=True)  # change index that from 0
    
    return df_normalized1,df_normalized2,clearn_dfs1,clearn_dfs2,args,clearn_dfs,types


def test_data_process(test_data_list,original_data_df ):
    dfs = pd.DataFrame()
    empty_row = pd.Series({}, name='EmptyRow')
    # type_list = []
    flag = []
    
    for sheet, data in test_data_list.items():
        # columns_with_underscore = [col for col in data.columns if '_' in col]
        # data[columns_with_underscore] = 0   # bu
        # data= data.drop(columns=columns_with_underscore)
        # data.columns = [col.replace('_', '') for col in data.columns] # remove '_' in the name of columns
        # data.rename(columns=name_map, inplace=True)
        data = data.append(empty_row,ignore_index=True)
        dfs =  pd.concat([dfs,data], ignore_index=True)
        # type_list = type_list.append(list(data['type'].value))
        flag.append(len(dfs))
    # if args.data_mode == 'one-to-one':
    #     name_list = name_list[:-1]
    types = get_types(dfs)
    
    
    df_type = pd.read_excel('data/types2.xlsx')
    df_type.drop(columns=['classes'], inplace=True)
    last = dfs.columns[-1]
    clean_dfs = process_new_type_for_generated_data(df_type,dfs,last)
    # one_hot_encoded = pd.get_dummies(dfs['types'], prefix='types')
    # dfs.drop('types', axis=1, inplace=True)  # del Category colunm
    # clean_dfs = pd.concat([one_hot_encoded,dfs], axis=1)
    # args.num_sys = len(clean_dfs.columns)-len(dfs.columns) # number of system 
    
    boundary = len( clean_dfs)
    
    temporary_dfs = pd.concat([clean_dfs,original_data_df ])
    # normalization
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame()
    for column in temporary_dfs.columns:
        # clean_dfs[column].replace('\\', 0, inplace=True)
        # df_normalized[column] = scaler.fit_transform(clean_dfs[[column]])
        df_normalized[column] = scaler.fit_transform(temporary_dfs[[column]]).flatten()
        
    df_normalized=  df_normalized.head(boundary)
     
    df_normalized1 = df_normalized.iloc[: flag[0]]  # 前 n 行
    df_normalized2 = df_normalized.iloc[ flag[0]:]
    df_normalized2 = df_normalized2.reset_index(drop=True)
    clearn_dfs1 = clean_dfs.iloc[: flag[0]]  # 前 n 行
    clearn_dfs2 = clean_dfs.iloc[ flag[0]:]  
    clearn_dfs2 = clearn_dfs2.reset_index(drop=True)  # change index that from 0
    
    return df_normalized1,df_normalized2,clearn_dfs1,clearn_dfs2,args,types

def test_curve(data_dict_normalized1,data_dict_normalized2, model, args):
    model.eval()
    true_list = []
    pred_list = []
    time = []

    xs_normalized1 = data_dict_normalized1['X']
    ys_true = data_dict_normalized1['y']
    # orignal_xs = data_dict_normalized1['orignal_x']
    
    xs_normalized2 = data_dict_normalized2['X']
    # ys_true = data_dict_normalized['y']
    orignal_xs = data_dict_normalized2['orignal_x_df'][args.independent_v]
    
    batch_size = args.batch_size

    # 按批次遍历数据
    for i in range(0, len(xs_normalized1), batch_size):
        batch_xs_normalized1 = xs_normalized1[i:i + batch_size]
        batch_xs_normalized2 = xs_normalized2[i:i + batch_size]
        batch_ys_true = ys_true[i:i + batch_size]
        batch_orignal_xs = orignal_xs[i:i + batch_size]
        for orignal_x in batch_orignal_xs:
            if isinstance(orignal_x, (int, float, str)):
                time.append(orignal_x)
            else:
                time.append(orignal_x[-1])

        batch_xs_normalized1 = torch.tensor(batch_xs_normalized1).to(args.device)
        batch_xs_normalized2 = torch.tensor(batch_xs_normalized2).to(args.device)
        y_pred = model(batch_xs_normalized1,batch_xs_normalized2)
        
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


# def only_prediction_test_curve(data_dict_normalized, model, args):
#     model.eval()
#     true_list = []
#     pred_list = []
#     time = []

#     xs_normalized = data_dict_normalized['X']
#     # ys_true = data_dict_normalized['orignal_x_df']['DEH value']
#     orignal_xs = data_dict_normalized['orignal_x_df'][args.independent_v]
    
#     batch_size = args.batch_size

#     # 按批次遍历数据
#     for i in range(0, len(xs_normalized), batch_size):
#         batch_xs_normalized = xs_normalized[i:i + batch_size]
#         # batch_ys_true = ys_true[i:i + batch_size]
#         batch_orignal_xs = orignal_xs[i:i + batch_size]
#         for orignal_x in batch_orignal_xs:
#             if isinstance(orignal_x, (int, float, str)):
#                 time.append(orignal_x)
#             else:
#                 time.append(orignal_x[-1])

#         if args.model in ['KNeighborsRegressor', 'DecisionTreeRegressor', 'SVR',
#                         'RandomForestRegressor', 'GradientBoostingRegressor', 'BayesianRidge']:
#             y_pred = model.predict(batch_xs_normalized)
#         else:
#             batch_xs_normalized = torch.tensor(batch_xs_normalized).to(args.device)
#             y_pred = model(batch_xs_normalized)
#         # y_pred = torch.from_numpy(scaler_y.inverse_transform(y_pred.cpu().detach().numpy()))
#         # y_pred = y_pred.squeeze().tolist()
#         y_pred = y_pred.cpu().detach().squeeze().tolist()
#         if isinstance(y_pred , float):
#             pred_list.extend([y_pred])
#         else:
#             pred_list.extend(y_pred)
#         # true_list.extend(batch_ys_true)
#     if args.data_mode == 'multi-to-one' and args.remove_y == False:
#         time.pop(0)
#         # true_list.pop()
#         pred_list.pop()
#     return pred_list,time


def only_prediction_test_curve(data_dict_normalized1,data_dict_normalized2, model, args):
    model.eval()
    true_list = []
    pred_list = []
    time = []

    xs_normalized1 = data_dict_normalized1['X']
    # ys_true = data_dict_normalized1['y']
    # orignal_xs = data_dict_normalized1['orignal_x']
    
    xs_normalized2 = data_dict_normalized2['X']
    # ys_true = data_dict_normalized['y']
    orignal_xs = data_dict_normalized2['orignal_x_df'][args.independent_v]
    
    batch_size = args.batch_size

    # 按批次遍历数据
    for i in range(0, len(xs_normalized1), batch_size):
        batch_xs_normalized1 = xs_normalized1[i:i + batch_size]
        batch_xs_normalized2 = xs_normalized2[i:i + batch_size]
        # batch_ys_true = ys_true[i:i + batch_size]
        batch_orignal_xs = orignal_xs[i:i + batch_size]
        for orignal_x in batch_orignal_xs:
            if isinstance(orignal_x, (int, float, str)):
                time.append(orignal_x)
            else:
                time.append(orignal_x[-1])

        batch_xs_normalized1 = torch.tensor(batch_xs_normalized1).to(args.device)
        batch_xs_normalized2 = torch.tensor(batch_xs_normalized2).to(args.device)
        y_pred = model(batch_xs_normalized1,batch_xs_normalized2)
        
        y_pred = y_pred.cpu().detach().squeeze().tolist()
        if isinstance(y_pred , float):
            pred_list.extend([y_pred])
        else:
            pred_list.extend(y_pred)
        # true_list.extend(batch_ys_true)
    if args.data_mode == 'multi-to-one' and args.remove_y == False:
        time.pop(0)
        # true_list.pop()
        pred_list.pop()
    return pred_list,time
 
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO
     
if __name__ == '__main__':
    #     # setting type
    # plt.rcParams['font.family'] = 'Times New Roman'
    # plt.rcParams['font.size'] = 21
    
    # load args
    args = parse()
    random.seed(args.seed)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    # check all path
    if args.dataset in ['cycDEH','isoDEH','TPD']:
        args.task_name = 'uni-component' 
    else:
        args.task_name = 'bi-component' 
    args.res_path = os.path.join(args.res_path,f'{args.model}/{args.task_name}_{args.dataset}')    
    args.res_value_scatter_path = os.path.join(args.res_path,'res_value')  # (3 rounds) store .csv and .log  and a gloabl scatter file
    args.res_importance_path = os.path.join(args.res_path,'res_importance') # (1 rounds)
    args.res_curve_path = os.path.join(args.res_path,'res_curve') # (1 rounds) a part of system need store the .csv file
    check_path(args.res_value_scatter_path)
    check_path(args.res_importance_path)
    check_path(args.res_curve_path)
    
        # config .log and .csv 
    log_path = os.path.join(args.res_value_scatter_path,f'errors_{args.seed}.log')
    csv_path = os.path.join(args.res_value_scatter_path,f'scatter_{args.seed}.csv')
    if os.path.exists(log_path):
        os.remove(log_path)
    logging.basicConfig(filename=log_path, level=logging.INFO,  format='%(message)s')    
    
    # setting the x axis
    args.independent_v = get_x(args.dataset)

    # clean dataset
    name_map = {'x1':'type', 'x2':'weight ratio', 'x3':'milling time', 
                'x4':'milling speed', 'x5':'ball-to-powder ratio', 
                'x6':'heating rate', 'x7':'1-st DEH temperature', 
                'x8':'1-st DEH time', 'x9':'REH temperature', 
                'x10': 'REH pressure', 'x11': 'REH time', 'x12': 'n-th DEH temperature', 
                'x13': 'n-th DEH time', 'x14': 'cycle number', 'y':'DEH value'}
    # train 
    data_list = pd.read_excel(f'data/{args.dataset}.xlsx',sheet_name=None)
    
    df_normalized1,df_normalized2,clearn_dfs1,clearn_dfs2,args,clearn_dfs,types = data_process(data_list,name_map)
    
    args,global_data1,global_sequence_data1,dataset1,loader1 = data_split_sequence(df_normalized1,clearn_dfs1,args)
    args,global_data2,global_sequence_data2,dataset2,loader2  = data_split_sequence(df_normalized2,clearn_dfs2,args)
    
     # load model
    if args.model == 'TR':
        backbone = TransformerForRegression\
            (args=args, output_dim=1).to(args.device)
    elif args.model in ['LinearRegressor','LassoRegressor', 'RidgeRegressor'] :
        backbone = eval(args.model)(args).to(args.device)
    elif args.model in ['KNeighborsRegressor','DecisionTreeRegressor','SVR',
                        'RandomForestRegressor','GradientBoostingRegressor','BayesianRidge']:
        backbone = get_model(args)
    elif args.model == 'Attention':
        args.attention_dim = args.fea_dim
        backbone = AttentionModel(args,output_dim= 1).to(args.device)
    elif args.model == 'TransformerRegressor':
        backbone = TransformerRegressor(args).to(args.device)
        
    model = bi_Model(args,backbone,output_dim=1).to(args.device)
        
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        avg_loss,model = train_model(loader1['train'],loader2['train'],model,args.model,optimizer) 
        print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss}')
          
    results,pred_list,true_list= test_model(loader1['test'],loader2['test'],model) 
    
    print(results)
    logging.info(f'original test dataset: {results}')    
    
    Preds = []
    Trues = []
    gloabl_feature_importances = []
    # common_paras = [{'type': 'K2NbF7', 'ratio': 14.84}, {'type': 'K2TiF6', 'ratio': 12.09}, 
    #                 {'type': 'Nb2C', 'ratio': 30.0}, {'type': 'Ti2C', 'ratio': 30.0}, 
    #                 {'type': 'Ti3C2', 'ratio': 30.0}, {'type': 'TiO2', 'ratio': 2.87}]
    for type_id in range(args.num_data):
        pred_list,true_list = [],[]
        if types[type_id] : #['K2TiF6', 'Ti3C2', 'Nb2C', 'TiO2', 'K2NbF7', 'Ti2C']:
            # prediction curve

            pred_list,true_list,time= test_curve(global_sequence_data1[type_id],global_sequence_data2[type_id],model,args) 
            curve_df = pd.DataFrame({f'{args.independent_v}':time,'True DEH Value': true_list, 'Prediction DEH Value': pred_list})
            curve_df.to_csv(f'{args.res_curve_path}/Orignal_Curve_{args.model}_{types[type_id]}.csv', index=False) 
            plot_df(curve_df,args,types[type_id])
            
        # scatter plot
        Preds+=pred_list 
        Trues+=true_list
    scatter_df = pd.DataFrame ({'Preds':Preds,'Trues':Trues}) 
    scatter_df.to_csv(csv_path, index=False) 
    plot_scatter(Preds,Trues,args)
        
    # test data genaration
    args.test_data = 'generated'
    test_data_list = get_test_data(args.dataset)
    df_normalized1,df_normalized2,clearn_dfs1,clearn_dfs2,args,types = test_data_process(test_data_list,clearn_dfs.iloc[:, :-1])
    args,global_data1,global_sequence_data1,dataset1,loader1 = data_split_sequence(df_normalized1,clearn_dfs1,args,splite = 0)
    args,global_data2,global_sequence_data2,dataset2,loader2  = data_split_sequence(df_normalized2,clearn_dfs2,args,splite = 0)
    
    Preds = []
    Trues = []
    gloabl_feature_importances = []
    
    for type_id in range(args.num_data):
        pred_list,true_list = [],[]
        
        if types[type_id] : #['K2TiF6', 'Ti3C2', 'Nb2C', 'TiO2', 'K2NbF7', 'Ti2C']:
            # prediction curve

            pred_list,time= only_prediction_test_curve(global_sequence_data1[type_id],global_sequence_data2[type_id],model,args) 
            curve_df = pd.DataFrame({f'{args.independent_v}':time,'Prediction DEH Value': pred_list})
            curve_df.to_csv(f'{args.res_curve_path}/Curve_{args.model}_{types[type_id]}.csv', index=False) 
            plot_df(curve_df,args,types[type_id])
      
    print('Finish!') 
    torch.cuda.empty_cache() 
    
    
    
    
    
    
        
    
        