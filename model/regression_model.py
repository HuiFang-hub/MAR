import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# 定义 PyTorch 模型
class LinearRegressor(nn.Module):
    def __init__(self,args ):
        super(LinearRegressor, self).__init__()
        input_size, hidden_size, output_size = args.input_size, args.hidden_size, args.output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def compute_feature_importance(self, loss_func, input_data):
        # model.eval()
        x = input_data[:,:-1]
        # y = input_data[:,-1]
        x.requires_grad_()
        x = self.fc1(x)
        x = self.relu(x)
        output = self.fc2(x)
        loss = loss_func(output, torch.zeros_like(output))  # Assuming zero target for feature importance
        loss.backward()
        feature_importance = x.grad.abs().mean(dim=0)  # Compute mean of absolute gradients
        return feature_importance  


class LassoRegressor(nn.Module):
    def __init__(self, args, l1_lambda=0.01):
        super(LassoRegressor, self).__init__()
        self.linear = nn.Linear(args.input_size, 1)
        self.l1_lambda = l1_lambda
        
    def forward(self, x):
        return self.linear(x)
    
    def l1_penalty(self):
        l1_norm = sum(p.abs().sum() for p in self.linear.parameters())
        return self.l1_lambda * l1_norm

class RidgeRegressor(nn.Module):
    def __init__(self, args, l2_lambda=0.01):
        super(RidgeRegressor, self).__init__()
        self.linear = nn.Linear(args.input_size, 1)
        self.l2_lambda = l2_lambda
        
    def forward(self, x):
        return self.linear(x)
    
    def l2_penalty(self):
        l2_norm = sum(p.pow(2.0).sum() for p in self.linear.parameters())
        return self.l2_lambda * l2_norm

