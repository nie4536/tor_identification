from torch.utils.data import Dataset,DataLoader,random_split
import torch
import torchkeras
import torchmetrics
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class cnn_1d_Data(Dataset):
    def __init__(self):
        data = pd.read_csv(r'E:\code\pretraining\1d-cnn_input_flow.csv', header=None)
        features = data.iloc[:,2:]      #取所有行和第二列到最后一列的所有数据
        labels = data.iloc[:,1]         #取所有行和第一列的交叉数据
        label_map = {label: index for index, label in enumerate(set(labels))}
        self.labels = list(map(lambda x: label_map[x], labels))
        # print(self.label)
        self.data = torch.tensor(np.array(features)/255,dtype=torch.float)

    def getClassNum(self):
        return len(set(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index],self.labels[index]

class cnn_1d_wangwei(nn.Module):
    def __init__(self,classNum):
        super().__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=25,stride=1,padding='same'),
            nn.ReLU()
        )
        self.pool_1 = nn.MaxPool1d(kernel_size=3,stride=3)
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=25,stride=1,padding='same'),
            nn.ReLU()
        )
        self.pool_2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.fc_1 = nn.Sequential(
            nn.Linear(87*64,1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.fc_2 = nn.Linear(1024, classNum)

    def forward(self,x):
        x = x.view(-1,1,784)             #(64,1,784)
        y = self.conv_1(x)
        y = self.pool_1(y)
        y = self.conv_2(y)
        y = self.pool_2(y)
        y = y.view(-1,87*64)
        y = self.fc_1(y)
        y = self.fc_2(y)                  #(64,24)
        return y


mydataSet = cnn_1d_Data()
classNum = mydataSet.getClassNum()
print(classNum)
tran_size,test_size = int(0.8 * len(mydataSet)),len(mydataSet)-int(0.8 * len(mydataSet))
train_dataset, test_dataset = random_split(mydataSet,[tran_size,test_size])
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=64,shuffle=True)

cnn_1d = cnn_1d_wangwei(classNum)
cnn_1d=cnn_1d.to(device)
# cnn_1d(torch.randint(0,1,(64,784),dtype=torch.float))
# summary = torchkeras.summary(cnn_1d,input_shape=(1,784))
model = torchkeras.KerasModel(cnn_1d,loss_fn=nn.CrossEntropyLoss(),
                              metrics_dict = {"acc":torchmetrics.Accuracy(task='multiclass',num_classes=classNum),
                                              "recall":torchmetrics.Recall(task='multiclass',num_classes=classNum),
                                              "f1":torchmetrics.F1Score(task='multiclass',num_classes=classNum)
                                              })
model.to(device)
dfhistory=model.fit(train_data=train_dataloader,
                    val_data=test_dataloader,
                    epochs=3,
                    patience=3,
                    # ckpt_path='checkpoint.pt',
                    monitor="val_acc",
                    mode="max")
print(dfhistory)
# y_pred = []
# y_true = []
# for inputs, labels in test_dataloader:
#     cnn_1d=cnn_1d.to(device)
#     output = cnn_1d(inputs.to(device))  # Feed Network
#
#     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#     y_pred.extend(output)  # Save Prediction
#
#     labels = labels.data.cpu().numpy()
#     y_true.extend(labels)
#
# cf_matrix = confusion_matrix(y_true, y_pred)
# print(cf_matrix)
# df_cm = pd.DataFrame(cf_matrix , index = [i for i in range(classNum)],
#                      columns = [i for i in range(classNum)])
# plt.figure(figsize = (12,10))
# sn.heatmap(df_cm, annot=False,cmap='Blues', vmin=100,vmax=1300)
# plt.savefig('output.png')