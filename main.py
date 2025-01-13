import pandas as pd
from model.attention import AttentionModel, TransformerForRegression, TransformerRegressor
from model.sequence_model import SimpleNet
from utils.data_process import data_split,data_split_sequence, get_types, ordered_set, process_new_type, split_nan_df
from utils.load_para import parse
import random
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from utils.metric import *
from utils.utils import check_path, get_x, test_curve
from utils.vis import pie, plot_df,bar, plot_scatter
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import logging
from model.regression_model import *
from model.sklearn_model import *
import shap
from matplotlib.font_manager import FontProperties

 
def train_model(dataloader,model,model_type,optimizer, criterion):
    sum_loss = 0
    model.train()
    num_data = 0
    for data in dataloader:        
        X,y_true,_ = data
        # if len(y_true) == args.batch_size:
        optimizer.zero_grad()
        y_pred = model(X)
        # get loss
        y_true =y_true.unsqueeze(-1) 
        loss = criterion(y_pred,y_true)
        if model_type == 'LassoRegressor':
            loss += model.l1_penalty()
        elif model_type == 'RidgeRegressor':
            loss += model.l2_penalty()
        loss.backward()
        optimizer.step()
        sum_loss += loss.item() * X.size(0)
        num_data += len(y_true)
    avg_loss = sum_loss/num_data
    return avg_loss,model
 
 
def test_model(model,dataloader):
    model.eval()
    y_pred = torch.tensor([])
    y_true = torch.tensor([])
    
    with torch.no_grad():
        for data in dataloader:
            x, y, _ = data
            y_hat = model(x)
            # y_hat = torch.from_numpy(scaler_y.inverse_transform(y_hat.cpu().detach().numpy()))
            # 将 y_hat 和 y 追加到列表中
            if y_hat.squeeze().dim()==0:
                y_hat = y_hat.view(-1)
                y_pred = torch.cat((y_pred, y_hat.cpu().detach()), dim=0)
            else:
                y_pred = torch.cat((y_pred, y_hat.cpu().detach().squeeze()), dim=0)
            y_true = torch.cat((y_true, y.cpu().detach()), dim=0)
            # y_pred.append(y_hat.cpu().detach().squeeze())
            # y_true.append(y.cpu().detach())
    
    # 将列表中的张量拼接成一个张量
    # y_pred = torch.cat(y_pred, dim=0)
    # y_true = torch.cat(y_true, dim=0)
        
    # combined_y_pred = torch.stack(y_pred) #torch.cat(y_pred, dim=0) 
    # combined_y_true = torch.stack(y_true).squeeze()  #torch.cat(y_true, dim=0)    
    results = all_metric(y_pred,y_true)
    # avg_mae =  sum(mae_list) / len(mae_list)
    return results,y_pred,y_true 


  
if __name__ == '__main__':
    # setting type
    font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
    font_prop = FontProperties(fname=font_path)
    # load args
    args = parse()
    random.seed(args.seed)
    args.device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    print(f"-------------dataset:{args.dataset},model:{args.model}------------------")
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
    import_png_path = os.path.join(args.res_importance_path,f'importance_{args.seed}.png')
    import_csv_path = os.path.join(args.res_importance_path,f'importance_{args.seed}.csv')
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
    
    data_list = pd.read_excel(f'data/{args.dataset}.xlsx',sheet_name=None)
    dfs = pd.DataFrame()
    empty_row = pd.Series({}, name='EmptyRow', dtype='float64')
    name_list = []
    # types_list = []
    for sheet, data in data_list.items():

        columns_with_underscore = [col for col in data.columns if '_' in col]
        # data[columns_with_underscore] = 0   # bu
        data= data.drop(columns=columns_with_underscore)
        data.columns = [col.replace('_', '') for col in data.columns] # remove '_' in the name of columns
        data.rename(columns=name_map, inplace=True)
        # data = pd.concat([data, empty_row], ignore_index=True)
        data = data.append(empty_row,ignore_index=True)
        dfs =  pd.concat([dfs,data], ignore_index=True)
        name_list =  data.columns.tolist()
    if args.data_mode == 'one-to-one':
        name_list = name_list[:-1]
    
    # get all types
    types = get_types(dfs)
    
    # if args.dataset == 'cycDEH':
    #     columns_to_drop = ['weight ratio','milling time','milling speed','1-st DEH temperature','REH time','cycle number']
    # elif args.dataset == 'isoDEH':
    #     columns_to_drop = ['milling time']
    # elif args.dataset == 'TPD':
    #     columns_to_drop = ['milling time','ball-to-powder ratio','heating rate']
    # dfs= dfs.drop(columns=columns_to_drop)
    
    # # one-hot
    # one_hot_encoded = pd.get_dummies(dfs['types'], prefix='types')
    # dfs.drop('types', axis=1, inplace=True)  # del Category colunm
    # clearn_dfs = pd.concat([one_hot_encoded,dfs], axis=1)
     
    # instead of one-hot
    df_type = pd.read_excel(f'data/types.xlsx')
    last = dfs.columns[-1]
    clearn_dfs = process_new_type(df_type,dfs,last)
    
    args.num_sys = len(clearn_dfs.columns)-len(dfs.columns)+1 # number of system 
    
    # normalization
    scaler_x =  StandardScaler()
    df_normalized = clearn_dfs.copy()  
    for column in clearn_dfs.columns[ args.num_sys: -1]:
        df_normalized[column] = scaler_x.fit_transform(clearn_dfs[[column]])
    # scaler_y = StandardScaler()
    # df_normalized['DEH value'] = scaler_y.fit_transform(clearn_dfs[['DEH value']])
        
    # load data
    args,global_data,global_sequence_data,dataset,loader = data_split_sequence(df_normalized,clearn_dfs,args)
    
    # args.win = args.fea_dim
    
    # load model
    if args.model == 'TR':
        model = TransformerForRegression\
            (args=args, output_dim=1).to(args.device)
    elif args.model in ['LinearRegressor','LassoRegressor', 'RidgeRegressor'] :
        model = eval(args.model)(args).to(args.device)
    elif args.model in ['KNeighborsRegressor','DecisionTreeRegressor','SVR',
                        'RandomForestRegressor','GradientBoostingRegressor','BayesianRidge']:
        model = get_model(args)
    elif args.model == 'Attention':
        args.attention_dim = args.fea_dim
        model = AttentionModel(args,output_dim= 1).to(args.device)
    elif args.model == 'TransformerRegressor':
        model = TransformerRegressor(args).to(args.device)
    
    
    # train
    if args.model in ['TR','LinearRegressor','LassoRegressor', 'RidgeRegressor','Attention','TransformerRegressor'] :
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

   
    for epoch in range(args.epochs):
        
        if args.model in ['KNeighborsRegressor','DecisionTreeRegressor','SVR',
                        'RandomForestRegressor','GradientBoostingRegressor','BayesianRidge']:
            X,y_true,_ = dataset['train']
            X,y_true = X.cpu(),y_true.cpu()
            model.fit( X, y_true) 
        else:
            avg_loss,model = train_model(loader['train'],model,args.model,optimizer, criterion) 
            print(f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss}')
    
    # test:  No training data
    if args.model in ['KNeighborsRegressor','DecisionTreeRegressor','SVR',
                        'RandomForestRegressor','GradientBoostingRegressor','BayesianRidge']:
        results,pred_list,true_list= test_sklearn_model(model,dataset['test']) 
    else:
        results,pred_list,true_list= test_model(model,loader['test']) 
    csv_path_test =  os.path.join(args.res_value_scatter_path,f'scatter_{args.seed}_test.csv')   
    scatter_df = pd.DataFrame ({'Preds':pred_list,'Trues':true_list}) 
    scatter_df.to_csv(csv_path_test, index=False) 
    fig_pth = f'{args.res_value_scatter_path}/{args.model}_scatter_test.png'
    plot_scatter(pred_list,true_list,args,fig_pth)  
    print(results)
    logging.info(f'{results}')
    
    # test: Training data may be included
    Preds = []
    Trues = []
    gloabl_feature_importances = []
    # common_paras = [{'type': 'K2NbF7', 'ratio': 14.84}, {'type': 'K2TiF6', 'ratio': 12.09}, 
    #                 {'type': 'Nb2C', 'ratio': 30.0}, {'type': 'Ti2C', 'ratio': 30.0}, 
    #                 {'type': 'Ti3C2', 'ratio': 30.0}, {'type': 'TiO2', 'ratio': 2.87}]
    for type_id in range(args.num_data):
        pred_list,true_list = [],[]
        #['K2TiF6', 'Ti3C2', 'Nb2C', 'TiO2', 'K2NbF7', 'Ti2C']:
            # prediction curve
        if args.model in ['KNeighborsRegressor','DecisionTreeRegressor','SVR',
                        'RandomForestRegressor','GradientBoostingRegressor','BayesianRidge']:
            pred_list,true_list,independent_v = test_sklearn_model_curve(global_sequence_data[type_id],model,args.independent_v)
        else: 
            pred_list,true_list,independent_v= test_curve(global_sequence_data[type_id],model,args) 
            
        # if types[type_id] in common_paras: 
        curve_df = pd.DataFrame({f'{args.independent_v}':independent_v,'True DEH Value': true_list, 'Prediction DEH Value': pred_list})
        curve_df.to_csv(f'{args.res_curve_path}/Curve_{args.model}_{types[type_id]}.csv', index=False) 
        plot_df(curve_df,args,types[type_id])
    
    
        # calculate feature importance 
        if args.model == 'TR' or args.model == 'Attention':
            feature_importance = model.calculate_attention_scores(global_sequence_data[type_id]).cpu().detach().numpy()
            sum_of_first_env = sum(feature_importance[:args.num_sys])/ args.num_sys
            feature_importance =  np.concatenate( ([sum_of_first_env] ,feature_importance[args.num_sys:]))
            # logging.info(f'{type_id}:{feature_importance}')
            pie(types[type_id],feature_importance, labels=name_list,args=args)
            gloabl_feature_importances.append(feature_importance)
            
        
        # scatter plot
        Preds+=pred_list 
        Trues+=true_list
    Preds,Trues = torch.tensor(Preds),torch.tensor(Trues)
    scatter_df = pd.DataFrame ({'Preds':Preds,'Trues':Trues}) 
    scatter_df.to_csv(csv_path, index=False) 
    plot_scatter(Preds,Trues,args)  
    
    
    # if args.model == 'TR':
    #     explainer = shap.GradientExplainer(model, dataset['train'][0])
    #     shap_values = np.squeeze(explainer.shap_values(dataset['test'][0]))
    #     X_test_numpy = dataset['test'][0].cpu().detach().numpy()
    #     shap_values = np.squeeze(shap_values[:, :, 1])
    #     X_test_numpy = np.squeeze( X_test_numpy[:, :, 1])
    #     shap.summary_plot(shap_values, X_test_numpy, feature_names=[f'Feature {i+1}' for i in range(args.fea_dim)])
    #     plt.savefig("test.png")  
    
    # calculate shap    
    if args.model == 'RandomForestRegressor':
        explainer = shap.TreeExplainer(model)
        ## split
        # shap_values = np.squeeze(explainer.shap_values(dataset['test'][0].cpu().numpy()))
        # test_sample = dataset['test'][0].cpu().detach().numpy()  
        # all_name_list = df_type.columns.tolist()[1:]+ name_list[1:] 
        # shap.summary_plot(shap_values , test_sample, feature_names=all_name_list, max_display=50)

        ##integration
        shap_values = np.squeeze(explainer.shap_values(dataset['test'][0].cpu().numpy()))
        test_sample = dataset['test'][0].cpu().detach().numpy()  
        type_shap_values = np.sum(np.abs(shap_values[:, :args.num_sys]), axis=1, keepdims=True)
        shap_values_reduced = np.concatenate([type_shap_values, shap_values[:, args.num_sys:]], axis=1)
        type_feature = np.mean(test_sample[:, :args.num_sys], axis=1, keepdims=True)
        test_sample_reduced = np.concatenate([type_feature, test_sample[:, args.num_sys:]], axis=1)
        
        all_name_list = name_list
        shap.summary_plot(shap_values_reduced , test_sample_reduced, feature_names=all_name_list, max_display=50)
               
        # plot
        fig = plt.gcf()

        # 设置字体为 Times New Roman，字号为 21
        for ax in fig.axes:
            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontproperties(font_prop)
                label.set_fontsize(21)
            ax.title.set_fontproperties(font_prop)
            ax.title.set_fontsize(21)
            ax.xaxis.label.set_fontproperties(font_prop)
            ax.xaxis.label.set_fontsize(21)
            ax.yaxis.label.set_fontproperties(font_prop)
            ax.yaxis.label.set_fontsize(21)
        plt.tight_layout()
        plt.savefig(import_png_path)
        
        # calculate importance
        type_shap_values = np.sum(np.abs(shap_values[:, :args.num_sys]), axis=1, keepdims=True)
        shap_values_reduced = np.concatenate([type_shap_values, shap_values[:, args.num_sys:]], axis=1)
        # type_feature = np.mean(test_sample[:, :args.num_sys], axis=1, keepdims=True)
        # test_sample_reduced = np.concatenate([type_feature, test_sample[:, args.num_sys:]], axis=1)
        feature_importance = np.mean(np.abs(shap_values_reduced), axis=0)
        feature_importance_normalized = feature_importance / np.sum(feature_importance)
        importance_df = pd.DataFrame({
            'feature': name_list,
            'importance': feature_importance_normalized
        })
        print(importance_df)
        importance_df.to_csv(import_csv_path, index=False) 
    
    if args.model == 'TR' or args.model == 'Attention':
    # global feature importance
        matrix_np = np.array(gloabl_feature_importances).T
        average_gloabl_feature_importance = np.mean(matrix_np, axis=1)
        pie('Global',average_gloabl_feature_importance, labels=name_list,args=args)
    print(f'dataset:{args.dataset}, model:{args.model}, lr:{args.lr}, batch size:{args.batch_size}. Finish!')    

    
        
    
        