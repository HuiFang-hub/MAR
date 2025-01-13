import torch
import torch.nn.functional as F


def all_metric(y_pred,y_true):
    mse = F.mse_loss(y_true, y_pred)
    rmse = round(torch.sqrt(mse).item(),4)
    mae = round(F.l1_loss(y_true, y_pred).item(),4)
    
    y_mean = torch.mean(y_true)
    sse = torch.sum((y_true - y_pred) ** 2)
    sst = torch.sum((y_true - y_mean) ** 2)
    r_squared = round(1 - (sse / sst).item(),4)
    
    explained_variance = round(1 - (sse / len(y_true) / torch.var(y_true)).item(),4)
    
    mse = round(mse.item() ,4)
    
    res = {'MSE':str(mse), 'RMSE':str(rmse), 'MAE':str(mae),
           'R2':str(r_squared),'EVS': str(explained_variance) }
    return res


if __name__ == '__main__':
    # Example data for true values and predicted values as PyTorch tensors
    y_true = torch.tensor([3, -0.5, 2, 7], dtype=torch.float32)
    y_pred = torch.tensor([2.5, 0.0, 2.1, 7.8], dtype=torch.float32)

    res = all_metric(y_pred,y_true)
    print(res)
    
