
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import numpy as np
import shap

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=1000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]
    
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
        
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # [1, max_len, d_model]
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         # x shape: [batch_size, seq_length, d_model]
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)

# class Attention(nn.Module):
#     def __init__(self, input_dim):
#         super(Attention, self).__init__()
#         self.query = nn.Linear(input_dim, input_dim)
#         self.key = nn.Linear(input_dim, input_dim)
#         self.value = nn.Linear(input_dim, input_dim)

#     def forward(self, x):
#         q = self.query(x)
#         k = self.key(x)
#         attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(x.size(-1)).float())
#         attn_weights = F.softmax(attn_scores, dim=-1)
#         attn_output = torch.matmul(attn_weights.transpose(-2, -1), x)
#         return attn_output.squeeze(-2), attn_weights.squeeze(-2)

class Attention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(Attention, self).__init__()
        self.compression = nn.Linear(input_dim, attention_dim)
        self.attention_weights= nn.Linear(attention_dim, attention_dim, bias=False)
        # nn.init.normal_(self.compression.weight, mean=0.0, std=0.01)
        init.xavier_uniform_(self.compression.weight)
        init.constant_(self.compression.bias, 0)
        # init.xavier_uniform_(self.attention_weights.weight)
        # init.constant_(self.attention_weights.bias, 0)
        
    def forward(self, x):
        # 计算注意力权重
        x = self.compression(x)
        attention_scores = self.attention_weights(x)
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 计算注意力加权平均
        context_vector = attention_weights * x
        # context_vector = torch.sum(context_vector, dim=1)
        
        return context_vector, attention_weights
   
class AttentionModel(nn.Module):
    def __init__(self, args,output_dim):
        super(AttentionModel, self).__init__()
        
        self.attention = Attention(args.fea_dim, args.attention_dim)
        self.fc1 = nn.Linear(args.fea_dim, output_dim)
        # self.fc1 = nn.Linear(args.fea_dim, args.hidden_size)
        # self.fc2 = nn.Linear(args.hidden_size, output_dim)
        # init.xavier_uniform_(self.fc1.weight)
        # init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.01)
        self.args = args
    
    def forward(self, x):
        context_vector, attention_weights = self.attention(x)
        out = self.fc1(context_vector)
        # out = torch.relu(self.fc1(context_vector))
        # out = self.fc2(out)
        # return out, attention_weights
        # out = torch.relu(context_vector)
        return out
    
    def calculate_attention_scores(self,data_dict):
        # optimizer.zero_grad()
        feas,labels = data_dict['X'],data_dict['y']
        # y = data[1]
        atts =  torch.tensor([])
        # for x,y_true in zip(feas,labels):
        for i in range(0, len(feas), self.args.batch_size):
            batch_xs =  torch.tensor(feas[i:i + self.args.batch_size]).to(self.args.device)            
            _, attention_weights = self.attention(batch_xs)
            att = attention_weights.cpu().detach()
            atts = torch.cat((atts, att), dim=0)
        attentions_final = torch.mean(atts, dim=0) 
        return attentions_final 
    
class bi_Model(nn.Module):
    def __init__(self, args,model,output_dim):
        super(bi_Model, self).__init__()
        
        self.model = model
        self.fc1 = nn.Linear(2, output_dim)
        # self.fc2 = nn.Linear(args.hidden_size, output_dim)
        self.args = args
    
    def forward(self, x1,x2):
        x1_out =  self.model(x1)
        x2_out=  self.model(x2)
        x_concat = torch.cat((x1_out,x2_out),dim = 1)
        # context_vector, attention_weights = self.attention(x)
        out = self.fc1(x_concat)
        out = torch.relu(out) 
        # out = torch.relu(self.fc1(context_vector))
        # out = self.fc2(out)
        # return out, attention_weights
        # out = torch.relu(context_vector)
        return out
    
    # def calculate_attention_scores(self,data_dict):
    #     # optimizer.zero_grad()
    #     feas,labels = data_dict['X'],data_dict['y']
    #     # y = data[1]
    #     atts =  torch.tensor([])
    #     # for x,y_true in zip(feas,labels):
    #     for i in range(0, len(feas), self.args.batch_size):
    #         batch_xs =  torch.tensor(feas[i:i + self.args.batch_size]).to(self.args.device)            
    #         _, attention_weights = self.attention(batch_xs)
    #         att = attention_weights.cpu().detach()
    #         atts = torch.cat((atts, att), dim=0)
    #     attentions_final = torch.mean(atts, dim=0) 
    #     return attentions_final 
    
    
class TransformerForRegression(nn.Module):
    def __init__(self, args, output_dim):
        super(TransformerForRegression, self).__init__()

        # self.positional_encoding = PositionalEncoding(input_dim)
        # self.attention = Attention(input_dim)
        self.multihead_attn = nn.MultiheadAttention(args.win, args.num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(args.win, nhead=args.num_heads), num_layers=args.num_layers)
        # self.fc1 = nn.Linear(args.fea_dim*args.win, args.hidden_size)
        # self.fc2 = nn.Linear(args.hidden_size, output_dim)
        self.fc1 = nn.Linear(args.fea_dim,output_dim)
        # self.fc2 = nn.Linear(args.hidden_dim, output_dim)
        self.device = args.device
        # self.batch_size = args.batch_size
        self.fea_dim = args.fea_dim
        self.args = args
             # 参数初始化
        self.init_weights()

    def init_weights(self):
        # 对线性层进行参数初始化
        init.xavier_uniform_(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        # init.xavier_uniform_(self.fc2.weight)
        # init.constant_(self.fc2.bias, 0)

    def forward(self, x,src_mask = None):
        if x.dim() < 3:
            # adjust dimension (3 D)
            # x = x.t()
            x = x.unsqueeze(-1)
        
        # x = self.positional_encoding(x)
        # self_attn_output, _ = self.attention(x)
        self_attn_output, attn_output_weights = self.multihead_attn(x, x, x, attn_mask=src_mask)
        # encoded_X = self.transformer_encoder(self_attn_output)
        # pooled_encoded_X = torch.mean(encoded_X, dim=-1)
        pooled_encoded_X = self_attn_output.squeeze(-1)
        if pooled_encoded_X.shape[-1] != 1:
            pooled_encoded_X = torch.mean(self_attn_output, dim=-1)
        # output = self.fc1(pooled_encoded_X)
        # output = self.fc2(output)
        output = self.fc1(pooled_encoded_X)
        return output
    
    def calculate_attention_scores(self,data_dict):
        # optimizer.zero_grad()
        feas,labels = data_dict['X'],data_dict['y']
        # y = data[1]
        atts =  torch.tensor([])
        for i in range(0, len(feas), self.args.batch_size):
            batch_xs =  torch.tensor(feas[i:i + self.args.batch_size]).to(self.args.device)
            if batch_xs.dim() < 3:
                # adjust dimension (3 D)
                batch_xs = batch_xs.unsqueeze(-1)
            _, attn_output_weights = self.multihead_attn(batch_xs, batch_xs, batch_xs, attn_mask=None)
            att = attn_output_weights[-1,:] 
            att = att.cpu().squeeze(-1).detach()
            atts = torch.cat((atts, att), dim=0)
        # for x,y_true in zip(feas,labels):
        #     x = torch.tensor(x).to(self.device)
        #     if x.dim() < 3:
        #         # adjust dimension (3 D)
        #         x = x.unsqueeze(-1)
        #     _, attn_output_weights = self.multihead_attn(x, x, x, attn_mask=None)
        #     att = attn_output_weights[-1,:] 
        #     att = att.cpu().squeeze(-1).detach().tolist()
        #     atts.append(att)
        attentions_final = torch.mean(atts, dim=0) 
        return attentions_final
    
    # def calculate_shap_scores(self,data_dict):
def calculate_shap_scores(args,model,data_loader) :
    explainer = shap.DeepExplainer(model, data_loader['train'])
    shap_values = explainer.shap_values(x_test)
    return 

#     feas,labels = data_dict['X'],data_dict['y']
#     shap_list = []  
#     for i in range(0, len(feas), args.batch_size):
#         batch_xs =  torch.tensor(feas[i:i + args.batch_size]).to(args.device)
#         if batch_xs.dim() < 3:
#             # adjust dimension (3 D)
#             batch_xs = batch_xs.unsqueeze(-1)

# 位置编码类（可选，因特征无序）
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 定义Transformer回归模型
class TransformerRegressor(nn.Module):
    # def __init__(self, input_dim, model_dim=64, num_heads=4, num_encoder_layers=2, dropout=0.1):
    def __init__(self, args, output_dim=1, dropout=0.1):
     
    
        super(TransformerRegressor, self).__init__()
        self.model_dim = args.hidden_size
        self.input_proj = nn.Linear(args.input_size, args.hidden_size)
        self.pos_encoder = PositionalEncoding(args.hidden_size, dropout)  # 可选
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=args.num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.num_layers)
        self.regressor = nn.Linear(args.hidden_size, output_dim)

    def forward(self, src):
        # src shape: [batch_size, input_dim]
        src = self.input_proj(src)  # [batch_size, model_dim]
        src = src.unsqueeze(1)  # 转换为 [batch_size, seq_length=1, model_dim]
        src = self.pos_encoder(src)  # 添加位置编码
        src = src.permute(1, 0, 2)  # Transformer expects [seq_length, batch_size, model_dim]
        memory = self.transformer_encoder(src)  # [seq_length, batch_size, model_dim]
        memory = memory.permute(1, 0, 2)  # [batch_size, seq_length, model_dim]
        out = memory.squeeze(1)  # [batch_size, model_dim]
        out = self.regressor(out)  # [batch_size, 1]
        return out





if __name__ == '__main__':
    # 定义模型参数
    input_dim = 20
    num_layers = 6
    num_heads = 2
    output_dim = 1  # 因为是回归任务，输出维度为1
    win= 6
     # 输入数据
    batch_size = 21
    # 创建模型实例
    model = TransformerForRegression(win, num_layers, num_heads, output_dim)
  
    
    X = torch.randn( batch_size,input_dim, win )  # 输入序列
    y = torch.randn(batch_size, output_dim)           # 输出目标

    # 前向传播
    output = model(X)

    # 计算回归损失
    criterion = nn.MSELoss()
    loss = criterion(output, y)

    print("Loss:", loss.item())
