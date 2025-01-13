import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

class SimpleNet(nn.Module):
    def __init__(self,args ):
        super(SimpleNet, self).__init__()
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

    
    
if __name__ == '__main__':
    input_size = 10
    hidden_size = 20
    output_size = 1
    num_layers = 1
