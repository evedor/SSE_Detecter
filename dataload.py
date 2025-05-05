import os
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scipy import signal
from torch.masked import masked_tensor


class TrainDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.file_list = os.listdir(self.folder_path)
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        data = pd.read_csv(file_path,skiprows=1,header=None)
        

        features = data.iloc[:, 1:4].values  # Extracting columns 2 to 4 as features
        label = data.iloc[:, 5].values  # Extracting column 5 as label
        # [m,n] = features.shape
        # outlier = self.make_outliers(m,5,0.005)
        # features = features
        # print(features)

        features = self.scaler.fit_transform(features)
        features = torch.tensor(features, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        # elif self.model =='v2':
        #     subsig1 = features[:,0]
        #     subsig2 = features[:,1]
        #     subsig3 = features[:,2]
        #     _,_,stftsig1 = signal.stft(subsig1,fs=1,nperseg=31,noverlap=30)
        #     _,_,stftsig2 = signal.stft(subsig2,fs=1,nperseg=31,noverlap=30)
        #     _,_,stftsig3 = signal.stft(subsig3,fs=1,nperseg=31,noverlap=30)
        #     # print('xinhao:',subsig1.shape)
        #     feature=np.empty((16,1000,6))
        #     feature[:,:,0] = np.real(stftsig1)
        #     feature[:,:,1] = np.imag(stftsig1)
        #     feature[:,:,2] = np.real(stftsig2)
        #     feature[:,:,3] = np.imag(stftsig2)
        #     feature[:,:,4] = np.real(stftsig3)
        #     feature[:,:,5] = np.imag(stftsig3)
        #     feature =feature.reshape(-1,16*6)
        #     feature = self.scaler.fit_transform(feature)
        #     feature = torch.tensor(feature, dtype=torch.float32)
        #     label = torch.tensor(label, dtype=torch.float32)
        return features,label
        

    # def make_outliers(self,length,max_size,prob):
    #     a = np.random.rand(length)<prob
    #     b = np.random.rand(length)<prob
    #     c = np.random.rand(length)<prob
    #     d1 = a*max_size*(np.random.randn(length)-0.5)*2
    #     d2 = b*max_size*(np.random.randn(length)-0.5)*2
    #     d3 = c*max_size*(np.random.randn(length)-0.5)*2
    #     d = np.vstack((d1,d2,d3)).T
    #     return d

class TestDataset(Dataset):
    def __init__(self, folder_path,lack_path):
        self.folder_path = folder_path
        self.lack_path = lack_path
        self.file_list = np.sort(os.listdir(self.folder_path))
        self.scaler = MinMaxScaler()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.folder_path, file_name)
        data = pd.read_csv(file_path,header=None,sep='\t')

        lack_file_path = os.path.join(self.lack_path,file_name)
        lack_data = pd.read_csv(lack_file_path,header=None,sep='\t')
        lack_datas = lack_data.iloc[:,1:4].values
        mask =torch.tensor(~np.isnan(lack_datas))

        time = data.iloc[:,0].values
        features = data.iloc[:, 1:4].values  # Extracting columns 2 to 4 as features
        features = self.scaler.fit_transform(features)
        features = torch.tensor(features, dtype=torch.float32)
        # features = masked_tensor(features,mask)
        # label = torch.tensor(label, dtype=torch.float32)
        return time,features,mask,file_name
    

# # # 用法示例
# dataset = CustomDataset('/home/wj/fakenet/cascadia/train_data3','v2')
# print(dataset)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# for features, labels in dataloader:
#     print("Batch of features:", features.shape)
#     print("Batch of labels:", labels)

# test_dataset = TestDataset('/home/wj/fakenet/cascadia/Data','/home/wj/fakenet/cascadia/Data_fulltime')
# print(test_dataset)
# print("a")
# # 创建数据加载器
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
# a=next(iter(test_dataloader))
# print(test_dataloader)

