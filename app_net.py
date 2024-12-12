from torch.utils.data import Dataset,DataLoader,random_split
import torch
import torchkeras
import torchmetrics
import torch.nn as nn
import numpy as np
import pandas as pd
import d2l.torch as d2l
class app_net_data(Dataset):
    def __init__(self):
        with open(r'E:\code\pretraining\app_net_bytes.txt','r') as f:
            lines = f.readlines()
        flowBytes = [line.strip().split('\t')[0].split(' ') for line in lines]

        self.vocab = d2l.Vocab(flowBytes, min_freq=1)
        self.data_c = [torch.tensor(self.vocab[flow_byte], dtype=torch.long) for flow_byte in flowBytes]
        data_lstm = pd.read_csv(r'E:\code\pretraining\app_net_length.csv', header = None)
        features_lstm = data_lstm.iloc[:,2:]
        labels = data_lstm.iloc[:,1]
        label_map = {label: index for index, label in enumerate(set(labels))}
        self.labels = list(map(lambda x: label_map[x], labels))
        # print(self.label)
        self.data_l = torch.tensor(np.array(features_lstm),dtype=torch.float)
        #print(len(self.data_l))
        #print(len(self.data_c))

    def getClassNum(self):
        return len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.data_c[index],self.data_l[index]),self.labels[index]

class app_net(nn.Module):
    def __init__(self,classNum):
        super().__init__()
        self.embed_cnn = nn.Embedding(260,256)
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=7,stride=1),
            nn.ReLU()
        )
        self.pool_1 = nn.MaxPool1d(kernel_size=3,stride=1)
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=256,out_channels=256,kernel_size=7,stride=1),
            nn.ReLU()
        )
        self.pool_2 = nn.MaxPool1d(kernel_size=3,stride=3)
        self.blstm_1 = nn.LSTM(input_size=1,hidden_size=128,batch_first=True,bidirectional=True,num_layers=2)
        self.liner = nn.Sequential(
            nn.Linear(in_features=91136,out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.liner_2 = nn.Sequential(
            nn.Linear(in_features=1024,out_features=classNum),
            nn.ReLU()

        )

    def forward(self,x):
        x_cnn,x_lstm = x[0],x[1]
        y_cnn = self.embed_cnn(x_cnn)
        y_cnn = torch.transpose(y_cnn, 1, 2)

        y_cnn = self.conv_1(y_cnn)
        y_cnn = self.pool_1(y_cnn)
        y_cnn = self.conv_2(y_cnn)
        y_cnn = self.pool_2(y_cnn)
        y_cnn = y_cnn.view(-1,256*336)
        # y_lstm = self.blstm_1(x_lstm)
        # print(x_lstm.shape)
        x_lstm = x_lstm.view(-1,20,1)
        y_lstm = self.blstm_1(x_lstm)[0]

        # print(y_lstm.shape)
        y_lstm = y_lstm.contiguous().view(-1,20*256)
        y = torch.cat((y_cnn,y_lstm),-1)
        # print(y.shape)
        y = self.liner(y)
        y = self.liner_2(y)
        return y

mydataSet = app_net_data()
classNum = mydataSet.getClassNum()
#print(classNum)
tran_size,test_size = int(0.8 * len(mydataSet)),len(mydataSet)-int(0.8 * len(mydataSet))
train_dataset, test_dataset = random_split(mydataSet,[tran_size,test_size])
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=True)

cnn_1d = app_net(classNum)
model = torchkeras.KerasModel(cnn_1d,loss_fn=nn.CrossEntropyLoss(),
                              metrics_dict = {"acc":torchmetrics.Accuracy(task='multiclass',num_classes=classNum),
                                              "recall":torchmetrics.Recall(task='multiclass',num_classes=classNum),
                                              "f1":torchmetrics.F1Score(task='multiclass',num_classes=classNum)
                                              })
dfhistory=model.fit(train_data=train_dataloader,
                    val_data=test_dataloader,
                    epochs=3,
                    patience=100,
                    # ckpt_path='checkpoint.pt',
                    monitor="val_acc",
                    mode="max")
print(dfhistory)

