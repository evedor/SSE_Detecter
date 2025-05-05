import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from dataload import CustomDataset


#################################################
# timestep = 1  # 时间步长，就是利用多少时间窗口
batch_size = 32  # 批次大小
input_dim = 3  # 每个步长对应的特征数量，就是使用每天的14个特征
hidden_dim = 64  # 隐层大小
output_dim = 1  # 由于是分类任务，最终输出层大小为1
num_layers = 3  # LSTM的层数
epochs = 10
best_loss = 10
train_ratio = 0.7  # 70% 用于训练，20% 用于验证
eval_ratio = 0.2
model_name = 'bilstm_attention'
save_path = './{}.pth'.format(model_name)


###################################
dataset = CustomDataset('/home/wj/fakenet/cascadia/train_data3')
# 计算划分的数量
train_size = int(len(dataset) * train_ratio)
eval_size = int(len(dataset) * eval_ratio)
test_size = len(dataset) - train_size - eval_size

# 划分数据集
train_dataset, eval_dataset, test_dataset = random_split(dataset, [train_size, eval_size, test_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



###############################################
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim  # 隐层大小
        self.num_layers = num_layers  # LSTM层数
        # embed_dim为每个时间步对应的特征数
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=3)
        # input_dim为特征维度，就是每个时间点对应的特征数量，这里为3
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, query, key, value):
#         print(query.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        attention_output, attn_output_weights = self.attention(query, key, value)
#         print(attention_output.shape) # torch.Size([16, 1, 4]) batch_size, time_step, input_dim
        output, (h_n, c_n) = self.lstm(attention_output)

        # print("output.shape：",output.shape) # torch.Size([32, 4748, 64]) batch_size, time_step, hidden_dim
        output = self.fc(output)
        # print("output.shape：",output.shape)
        # batch_size, timestep, hidden_dim = output.shape
        # output = output.reshape(-1, hidden_dim)
        # output = self.fc(output)
        # print("output.shape：",output.shape)
        # output = output.reshape(timestep, batch_size, -1)
        return output


model = LSTM(input_dim, hidden_dim, num_layers, output_dim)  # 定义LSTM网络
loss_function = nn.MSELoss()  # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 定义优化器







#######################################################
for epoch in range(epochs):
    model.train()
    running_loss = 0
    train_bar = tqdm(train_dataloader)  # 形成进度条
    for data in train_bar:
        x_train, y_train = data  # 解包迭代器中的X和Y
        optimizer.zero_grad()
        y_train_pred = model(x_train, x_train, x_train)
        # print(y_train_pred.shape)
        # print(y_train.shape)
        loss = loss_function(y_train_pred, y_train[:,:,np.newaxis])
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)

    # 模型验证
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        eval_bar = tqdm(eval_dataloader)
        for data in eval_bar:
            x_eval, y_eval = data
            y_eval_pred = model(x_eval, x_eval, x_eval)
            eval_loss = loss_function(y_eval_pred, y_eval[:,:,np.newaxis])

torch.save(model.state_dict(), save_path)
print('Finished Training')


# #############################################################
model.eval()
tp = 0
fp = 0
tn = 0
fn = 0
for feature, labels in test_dataloader:
    predictions = model(feature, feature, feature)
    predicted_classes = predictions >0.5
    tp += ((predicted_classes == 1) & (labels == 1)).sum().item()
    fp += ((predicted_classes == 1) & (labels == 0)).sum().item()
    tn += ((predicted_classes == 0) & (labels == 0)).sum().item()
    fn += ((predicted_classes == 0) & (labels == 1)).sum().item()
accuracy = (tp+tn)/(tp+tn+fp+fn)
precision = tp/(tp+fp)
recall = tp/(tp+fn)
f1_score = 2*(precision*recall)/(precision+recall)

print("accuracy:",accuracy)
print("precision:",precision)
print("recall:",recall)
print("f1_score:",f1_score)
