import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
import torch

from utils.metric import all_metric


def get_model(args):
    model_name = args.model
    # 多项式回归（为了适应多维数据，我们使用每个特征的二次项）
    if model_name  == 'KNeighborsRegressor':
        model = KNeighborsRegressor(n_neighbors=5)
    elif model_name  == 'DecisionTreeRegressor':
        model = DecisionTreeRegressor()
    elif model_name == 'SVR':
        model= SVR(kernel='rbf')
    elif model_name == 'RandomForestRegressor':
        model = RandomForestRegressor(n_estimators=10, random_state=args.seed)
    elif model_name == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(n_estimators=10, random_state=args.seed)
    elif model_name == 'BayesianRidge':
        model =  BayesianRidge()
    return model
    
    
def train_sklearn_model(model,data):
    X,y_true,_ = data
    model.fit( X, y_true) 
    return model    


def test_sklearn_model(model,data):
    X,y_true,_ = data
    X,y_true = X.cpu(),y_true.cpu()
    y_pred = model.predict(X)
    y_pred= torch.from_numpy(y_pred)
    # y_true = torch.from_numpy(y_true)
    results = all_metric(y_pred,y_true)    
    return results, y_pred, y_true 

def test_sklearn_model_curve(data_dict_normalized,model,independent_v):
    # X,y_true,_ = data
    # X,y_true = X.cpu(),y_true.cpu()
    orignal_x = data_dict_normalized['orignal_x_df'][independent_v]
    xs_normalized,ys_true,orignal_xs = data_dict_normalized['X'],data_dict_normalized['y'],orignal_x 
    # xs_normalized,ys_true,orignal_xs = xs_normalized.cpu(),ys_true.cpu(),orignal_xs.cpu()
    y_pred = model.predict(xs_normalized)
    # time = orignal_xs[:,-1]
    return list(y_pred) ,list(ys_true),list(orignal_xs)
    
    