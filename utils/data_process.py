import os
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch.utils.data import Dataset
import numpy as np
import math  
from sklearn.preprocessing import MinMaxScaler
from itertools import product

class CustomDataset(Dataset):
    def __init__(self, X, y,orignal_x):
        self.X = X
        self.y = y
        self.orignal_x = orignal_x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx]),torch.tensor(self.orignal_x[idx])

def data_split(data,args):
    # del NAN raws
    data = data.dropna(axis=0, how='any')
    # get_data_feature
    data = data.astype(float)
    if args.shuffle == True:
        data = data.sample(frac=1).reset_index(drop=True)
    # split 
    data_list = data.values
    # #781,20
    args.input_size = len(data_list[0])-1
    train_size = int(0.7 * len(data_list))
    val_size = int(0.2 * len(data_list))
    train_dataset= torch.tensor(data_list[0:train_size], dtype=torch.float32)
    val_dataset = torch.tensor(data_list[train_size:val_size], dtype=torch.float32)
    test_dataset =  torch.tensor(data_list[val_size:], dtype=torch.float32)
    
    train_dataset1 = CustomDataset(train_dataset)
    val_dataset1 = CustomDataset(val_dataset)
    test_dataset1 = CustomDataset(test_dataset)
    
    # batch_size = 2
    train_dataloader = DataLoader(train_dataset1, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset1, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False)
    # for i, data in enumerate( train_dataloader):
    #     y = data[:,-1]

    return train_dataset,val_dataset,test_dataset,train_dataloader,val_dataloader, test_dataloader,args

def split_nan_df(df):
    # find the position of the NAN raw
    # empty_rows = df[df.isnull().all(axis=1)].index
    empty_rows = df[df.isnull().any(axis=1)].index
    dfs = []
    start = 0
    for end in empty_rows:
        new_df = df.iloc[start:end]
        if not new_df.empty:
            # dfs = pd.concat([dfs, new_df], ignore_index=True)
            dfs.append(new_df)
        start = end + 1

    # the last dataframe
    last_df = df.iloc[start:]
    if not last_df.empty:
        # dfs = pd.concat([dfs, last_df], ignore_index=True)
        dfs.append(last_df)
    return dfs


def sliding_window(df,df_orignal,args):
    # Calculate the starting and ending indices for the sliding window
    remove_y, win,step = args.remove_y, args.win, args.win_step
    start = 0
    end = win
    X = []
    y = []
    orignal_x = []
    # print(args.data_mode)
    # print(remove_y)
    if args.data_mode == 'multi-to-one':
        if remove_y:  
            while end <= len(df):  
            # Get the values within the sliding window as X
                x_window = df.iloc[start:end, :-1].values.T.tolist()  # not include y
                X.append(x_window)
                # Get the value of the last row in the sliding window as y
                y_value = df.iloc[end-1, -1]
                y.append(y_value)
                # get orignal X axis value
                subset = df_orignal.loc[:, args.independent_v].iloc[start:end].values  
                orignal_x_window =subset.T.tolist()
                # Update the starting and ending indices               
                start += step
                end += step  
        else:
            while end <= len(df)-1:
            # Get the values within the sliding window as X
                x_window = df.iloc[start:end, :].values.T.tolist()  # not include y
                X.append(x_window)
                # Get the value of the last row in the sliding window as y
                y_value = df.iloc[end, -1]
                y.append(y_value)
                 # get orignal X axis value
                subset = df_orignal.loc[:, args.independent_v].iloc[start:end].values  
                orignal_x_window =subset.T.tolist() 
                orignal_x.append(orignal_x_window)
                # Update the starting and ending indices
                start += step
                end += step
            # print(orignal_x)
    elif args.test_data == 'generated':
        for (index, row), (index2, row2) in zip(df.iterrows(), df_orignal.iterrows()):
            x =  row.tolist()
            orignal_x_window = row2.tolist()
            orignal_x.append(orignal_x_window)
            X.append(x)
        
            
    else: #one-to-one
        
       for (index, row), (index2, row2) in zip(df.iterrows(), df_orignal.iterrows()):
            x =  row.tolist()[:-1]
            y_value =  row.tolist()[-1]
            # orignal_x_window = row2[args.independent_v]
            orignal_x_window = row2.tolist()[:-1]
            orignal_x.append(orignal_x_window)
            X.append(x)
            y.append(y_value)  
    
    
    return X, y,orignal_x

def data_split_sequence(dfs,dfs_orignal,args,splite = 0.7): 
    dfs = split_nan_df(dfs)
    dfs_orignal = split_nan_df(dfs_orignal)
    
    args.num_data = len(dfs)
      
    feas = []  # 730,3,20
    labels = [] # 730,
    orignal_xs = []
    global_sequence_data = dict()
    
    for i in range(len(dfs)):
        # get_data_feature
        df = dfs[i]
        df = df.astype(float)
        # get original x axis
        df_orignal = dfs_orignal[i]
        # data_list = df.values
        X,y,orignal_x = sliding_window(df,df_orignal,args)
        feas+=X
        labels+=y
        orignal_xs += orignal_x
        global_sequence_data[i]={'X':X,'y':y,'orignal_x':orignal_x,'orignal_x_df':df_orignal}
       
    if args.shuffle:
        # random.seed(args.seed)
        random_indices = list(range(len(feas)))
        random.shuffle(random_indices)
        if args.test_data != 'generated':
            labels = [labels[i] for i in random_indices]
        feas = [feas[i] for i in random_indices]
        
        # print(np.array(orignal_xs).shape)
        orignal_xs = [orignal_xs[i] for i in random_indices]
        
    args.fea_dim = len(feas[0])
    len_d = len(feas)
    train_size = int(splite * len_d )
    global_data = ( feas,  labels, orignal_xs)
    
    # split into train,test
    train_feas= torch.tensor(feas[0:train_size], dtype=torch.float32).to(args.device)
    train_labels =  torch.tensor(labels[0:train_size], dtype=torch.float32).to(args.device)
    train_orignal_x =  torch.tensor(orignal_xs[0:train_size], dtype=torch.float32).to(args.device)
    test_feas = torch.tensor(feas[train_size:], dtype=torch.float32).to(args.device)
    test_labels =  torch.tensor(labels[train_size:], dtype=torch.float32).to(args.device)
    test_orignal_x =  torch.tensor(orignal_xs[train_size:], dtype=torch.float32).to(args.device)

    args.input_size = test_feas.shape[-1]
    
    train_dataset = (train_feas, train_labels,train_orignal_x)
    test_dataset = (test_feas, test_labels,test_orignal_x)
    
    train_customdataset = CustomDataset(train_feas,train_labels,train_orignal_x)
    test_customdataset = CustomDataset(test_feas,test_labels,test_orignal_x)
    
    # batch_size = 2
    train_dataloader = DataLoader(train_customdataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_customdataset, batch_size=args.batch_size, shuffle=False)
    # for i, train_data in enumerate( train_dataloader):
    #     # test = data
    #     x= train_data[0]  # 4,20,3
    #     y = train_data[1]
        # print(np.array(y).shape)
    dataset = {'train':train_dataset,'test':test_dataset}
    loader = {'train':train_dataloader,'test':test_dataloader}
    return args,global_data, global_sequence_data,dataset,loader


def ordered_set(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]

def get_types(dfs):
    dfs = split_nan_df(dfs)
    type = []
    for df in dfs:
        if 'type' in df.columns:
            type_ = df['type'].iloc[0]
            ratio_ = df['weight ratio'].iloc[0]  if 'weight ratio' in df.columns and not df['weight ratio'].empty else 0
            temp_ = df['1-st DEH temperature'].iloc[0] if '1-st DEH temperature' in df.columns and not df['1-st DEH temperature'].empty else 0
            cyc_ = df['cycle number'].iloc[0] if 'cycle number' in df.columns and not df['cycle number'].empty else 0
        else:
            type_ = df['x1'].iloc[0]if 'x1' in df.columns else 0
            ratio_ = df['x2'].iloc[0] if 'x2' in df.columns else 0
            temp_ = df['x7'].iloc[0] if 'x7' in df.columns else 0
            cyc_ = df['x14'].iloc[0] if 'x14' in df.columns else 0
            
        # types.extend([{'type':type_,'ratio':ratio_}])
        type.extend([{'type':type_,'ratio':ratio_,'temp':temp_,'cyc':cyc_}])
    return type


def process_new_type(df_type,df,last):
    # # Example DataFrame A with 11 columns, where the 0th column is 'type'
    # df_type = pd.DataFrame({
    #     'type': ['apple', 'banana', 'cherry'],
    #     'feature1': [10, 20, 30],
    #     'feature2': [100, 200, 300],
    #     # Add other columns...
    #     'feature9': [1000, 2000, 3000],
    #     'feature10': [10000, 20000, 30000]
    # })

    # # Example DataFrame df
    # df = pd.DataFrame({
    #     'type': ['apple', 'banana', 'banana', 'cherry', 'apple'],
    #     'other_column': [1, 2, 3, 4, 5]
    # })
    # get the last column name
  
    
    # Set the 'type' column of A as the index to facilitate mapping
    df_type_indexed = df_type.set_index('type')

    # Join A's columns to df on the 'type' column and drop the original 'type' column in df
    df = df.join(df_type_indexed, on='type').drop(columns=['type'])

    # Adjust the column order to insert new columns after the original 'type' column
    cols = df.columns.tolist()
    type_idx = cols.index(last)  # Find the insertion point, i.e., the original 'type' column's position
    type_cols = cols[type_idx+1:]  # Get the newly added columns
    orignal_cols =  cols[:type_idx+1] 
    type_col_df = df[type_cols]
    orignal_col_df = df[orignal_cols]
    
    # normalized
    scaler = MinMaxScaler()
    type_df_normalized = pd.DataFrame(scaler.fit_transform(type_col_df), columns=type_col_df.columns)

   
    merged_df = pd.concat([type_df_normalized, orignal_col_df ], axis=1)
    
    # cols = new_cols + cols[:type_idx+1]  # Place the new columns right after the original 'type' column
    # new_df = df[cols]
    # Print the result
    return merged_df

def merge_and_compute_sum(df_A: pd.DataFrame, df_B: pd.DataFrame) -> pd.DataFrame:
    A_feature_cols = [col for col in df_A.columns if col != 'type']
    type_split = df_A['type'].str.split('-', expand=True)
    df_A[['type1', 'type2']] = type_split

    df_A = df_A.merge(df_B, how='left', left_on='type1', right_on='type', suffixes=('', '_type1'))
    
    feature_cols = [col for col in df_B.columns if col != 'type']
    type1_features = {col: f"{col}_type1" for col in feature_cols}
    df_A.rename(columns=type1_features, inplace=True)
    df_A.drop('type_type1', axis=1, inplace=True)

    df_A = df_A.merge(df_B, how='left', left_on='type2', right_on='type', suffixes=('', '_type2'))
    type2_features = {col: f"{col}_type2" for col in feature_cols}
    df_A.rename(columns=type2_features, inplace=True)
    df_A.drop('type_type2', axis=1, inplace=True)

    for feature in feature_cols:
        type1_col = f"{feature}_type1"
        type2_col = f"{feature}_type2"
        df_A[feature] = df_A[type1_col].fillna(0) + df_A[type2_col].fillna(0)

    feature_columns = feature_cols.copy()
    df_final = df_A[feature_columns+A_feature_cols].copy()
    return df_final

def process_new_type_for_generated_data(df_type,df,last):
    
    # df_type = pd.DataFrame({
    #     'type': ['a', 'b', 'c','d','e'],
    #     'feature1': [0,0,0,1,1],
    #     'feature2': [0,1,0,0,1],
    #     'feature3': [1,0,0,1,0],
    #     'feature4': [0,2,0,0,0]
    # })

    
    type_cols = df_type.columns.tolist() [1:]  # Get the newly added columns
    orignal_cols = df.columns.tolist()[1:] 
    new_df = merge_and_compute_sum(df, df_type)

    # df_type_indexed = df_type.set_index('type')
    # new_df  = df.join(df_type_indexed, on='types').drop(columns=['types'])

    # cols = new_df .columns.tolist()

    type_col_df = new_df [type_cols]
    orignal_col_df = new_df [orignal_cols]
    
    # normalized
    scaler = MinMaxScaler()
    type_df_normalized = pd.DataFrame(scaler.fit_transform(type_col_df), columns=type_col_df.columns)

   
    merged_df = pd.concat([type_df_normalized, orignal_col_df ], axis=1)
    
    # cols = new_cols + cols[:type_idx+1]  # Place the new columns right after the original 'type' column
    # new_df = df[cols]
    # Print the result
    return merged_df



# def process_new_type_for_generated_data(df_type,df):
#     # # Example DataFrame A with 11 columns, where the 0th column is 'type'
#     # df_type = pd.DataFrame({
#     #     'type': ['apple', 'banana', 'cherry'],
#     #     'feature1': [10, 20, 30],
#     #     'feature2': [100, 200, 300],
#     #     # Add other columns...
#     #     'feature9': [1000, 2000, 3000],
#     #     'feature10': [10000, 20000, 30000]
#     # })

#     # # Example DataFrame df
#     # df = pd.DataFrame({
#     #     'type': ['apple', 'banana', 'banana', 'cherry', 'apple'],
#     #     'other_column': [1, 2, 3, 4, 5]
#     # })
#     # get the last column name
  
    
#     # Set the 'type' column of A as the index to facilitate mapping
#     df_type_indexed = df_type.set_index('type')

#     # Join A's columns to df on the 'type' column and drop the original 'type' column in df
#     df = df.join(df_type_indexed, on='types').drop(columns=['types'])

#     # Adjust the column order to insert new columns after the original 'type' column
#     cols = df.columns.tolist()
#     type_idx = cols.index(last)  # Find the insertion point, i.e., the original 'type' column's position
#     type_cols = cols[type_idx+1:]  # Get the newly added columns
#     orignal_cols =  cols[:type_idx+1] 
#     type_col_df = df[type_cols]
#     orignal_col_df = df[orignal_cols]
    
#     # normalized
#     scaler = MinMaxScaler()
#     type_df_normalized = pd.DataFrame(scaler.fit_transform(type_col_df), columns=type_col_df.columns)

   
#     merged_df = pd.concat([type_df_normalized, orignal_col_df ], axis=1)
    
#     # cols = new_cols + cols[:type_idx+1]  # Place the new columns right after the original 'type' column
#     # new_df = df[cols]
#     # Print the result
#     return merged_df

def add_empty_entry(data_dict, empty_value=None):
    return {key: value + [empty_value] for key, value in data_dict.items()}

def generate_test_data():
    """
    name_map = {'x1':'types', 'x2':'weight ratio', 'x3':'milling time', 
                'x4':'milling speed', 'x5':'ball-to-powder ratio', 
                'x6':'heating rate', 'x7':'1-st DEH temperature', 
                'x8':'1-st DEH time', 'x9':'REH temperature', 
                'x10': 'REH pressure', 'x11': 'REH time', 'x12': 'n-th DEH temperature', 
                'x13': 'n-th DEH time', 'x14': 'cycle number', 'y':'DEH value'}
    
    """
    type_df = pd.read_excel(f'data/types2.xlsx')
    # classes = type_df['classes']
    types_dict = type_df.groupby('classes')['type'].apply(list).to_dict()
    A_types = types_dict['A']
    B_types = types_dict['B']
    # A_types = ['AlH3','LiH','NaBH4','NaAlH4','LiBH4','LiAlH4']
    # B_types = ['NdF3','TiF3','Ti2C','V2C','FGi']
    pairs = [(a, b) for a, b in product(A_types, B_types ) if a != b]
    A_pairs, B_pairs = zip(*pairs)
    
    for dataset in ['bi-TPD','bi-isoDEH','bi-cycDEH']:
        data_dict_A = {}
        data_dict_B = {}
        for type_a,type_b in zip(A_pairs, B_pairs):
            if  dataset == 'bi-cycDEH' :
                # """
                # var_dict = {'weight ratio':30,'milling time':2,'milling speed':400,
                #                         'ball-to-powder ratio':120,'heating rate':15,
                #                         '1-st DEH temperature':260, 'REH temperature':300,
                #                         'REH pressure':10, 'REH time':8,'n-th DEH temperature':260, 
                #                         'cycle number':2}
                # """
                length = 360 # a point/per min
                data_dict_A['type'] = data_dict_A['type']+ [type_a+"-"+type_b]*length if 'type' in data_dict_A.keys() else [type_a+"-"+type_b]*length
                data_dict_B['type'] = data_dict_B['type']+ [type_a+"-"+type_b]*length if 'type' in data_dict_B.keys() else [type_a+"-"+type_b]*length
                    
                vars= ['weight ratio', 'milling time', 'milling speed', 'ball-to-powder ratio', 'heating rate', '1-st DEH temperature', 
                       'REH temperature', 'REH pressure', 'REH time', 'n-th DEH temperature', 'n-th DEH time','cycle number']
                vals = [30, 2, 400, 120, 15, 260, 300, 10, 8, 260, 0, 2]
                for var,val in zip(vars,vals):  
                    if var == 'n-th DEH time':
                        data_dict_A[var] = data_dict_A[var]+ list(range(length)) if var in data_dict_A.keys() else list(range(length))
                        data_dict_B[var] = data_dict_B[var]+ list(range(length)) if var in data_dict_B.keys() else list(range(length))
                    else:
                        data_dict_A[var]  = data_dict_A[var]+ [val]*length if var in data_dict_A.keys() else [val]*length
                        data_dict_B[var]  = data_dict_B[var]+ [val]*length if var in data_dict_B.keys() else [val]*length
            
            
            elif  dataset == 'bi-isoDEH' :
                length = 360 # a point/per min
                data_dict_A['type'] = data_dict_A['type']+ [type_a+"-"+type_b]*length if 'type' in data_dict_A.keys() else [type_a+"-"+type_b]*length
                data_dict_B['type'] = data_dict_B['type']+ [type_a+"-"+type_b]*length if 'type' in data_dict_B.keys() else [type_a+"-"+type_b]*length
                vars= ['weight ratio', 'milling time', 'milling speed', 'ball-to-powder ratio', 'heating rate', '1-st DEH temperature','1-st DEH time']
                vals = [30, 2, 400, 120, 15, 260, 0]
                for var,val in zip(vars,vals):
                    
                    if var == '1-st DEH time':
                        data_dict_A[var] = data_dict_A[var]+ list(range(length)) if var in data_dict_A.keys() else list(range(length))
                        data_dict_B[var] = data_dict_B[var]+ list(range(length)) if var in data_dict_B.keys() else list(range(length))
                    else:
                        data_dict_A[var]  = data_dict_A[var]+ [val]*length if var in data_dict_A.keys() else [val]*length
                        data_dict_B[var]  = data_dict_B[var]+ [val]*length if var in data_dict_B.keys() else [val]*length
            
            
            elif dataset == 'bi-TPD' :
                length = 500
                data_dict_A['type'] = data_dict_A['type']+ [type_a+"-"+type_b]*length if 'type' in data_dict_A.keys() else [type_a+"-"+type_b]*length
                data_dict_B['type'] = data_dict_B['type']+ [type_a+"-"+type_b]*length if 'type' in data_dict_B.keys() else [type_a+"-"+type_b]*length
                vars= ['weight ratio', 'milling time', 'milling speed', 'ball-to-powder ratio', 'heating rate','1-st DEH temperature']
                vals = [30, 2, 400, 120, 5,0]
                for var,val in zip(vars,vals):
                    if var == '1-st DEH temperature':
                        data_dict_A[var] = data_dict_A[var]+ list(range(length)) if var in data_dict_A.keys() else list(range(length))
                        data_dict_B[var] = data_dict_B[var]+ list(range(length)) if var in data_dict_B.keys() else list(range(length))
                    else:
                        data_dict_A[var]  = data_dict_A[var]+ [val]*length if var in data_dict_A.keys() else [val]*length
                        data_dict_B[var]  = data_dict_B[var]+ [val]*length if var in data_dict_B.keys() else [val]*length
                
            data_dict_A = add_empty_entry(data_dict_A, empty_value=None)
            data_dict_B = add_empty_entry(data_dict_B, empty_value=None)        
        data_df_A = pd.DataFrame(data_dict_A)
        data_df_B = pd.DataFrame(data_dict_B)
        excel_path = f'data/test_{dataset}.xlsx'

        # 使用 ExcelWriter 来写入多个工作表
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            data_df_A.to_excel(writer, sheet_name='sheet1', index=False)
            data_df_B.to_excel(writer, sheet_name='sheet2', index=False)


def get_test_data(dataset_name) :
    path = f'data/test_{dataset_name}.xlsx'
    if os.path.exists(path):
        dataset  = pd.read_excel(path,sheet_name=None) 
    else:
        generate_test_data()
        dataset  = pd.read_excel(path,sheet_name=None) 
    return dataset

if __name__ == '__main__':
    generate_test_data()
        
                
            
            
            
            
        
   
