import torch
from torch import nn
import model

# from torchsummary import summary
from d2l.torch import d2l
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score,precision_score
import torchkeras
import torchmetrics
from thop import profile,clever_format
from ptflops import get_model_complexity_info
import csv
packet_num = 1
bytes_num = 60
class flowDataset(Dataset):
    def __init__(self):
        global bytes_num
        with open('E:/code/pretraining/fine_tuning_data.txt','r') as f:
        # with open('./ISCX_data.txt', 'r') as f:
            lines = f.readlines()
            # print(len(lines))
        self.labels = [line.strip().split('\t')[-1] for line in lines if len(line.strip().split('\t')[0].split(' '))>=bytes_num]
        label_map = {label: index for index, label in enumerate(set(self.labels))}
        #print(label_map)
        self.labels = list(map(lambda x: label_map[x], self.labels))
        # print(lines)
        flowBytes = [line.strip().split('\t')[0].split(' ')[0:bytes_num] for line in lines if len(line.strip().split('\t')[0].split(' '))>=bytes_num]
        # print(len(self.labels))
        self.vocab = d2l.Vocab(flowBytes, min_freq=1)
        # print(self.vocab)
        # print(len(self.vocab))
        # print(flowBytes)
        self.bytes = [torch.tensor(self.vocab[flow_byte],dtype=torch.long) for flow_byte in flowBytes]
        print(self.bytes[0].shape)

    def getClassNum(self):
        return len(set(self.labels))

    def getVocabLen(self):
        return len(self.vocab)

    def __getitem__(self, index):
        return self.bytes[index],self.labels[index]

    def __len__(self):
        return len(self.labels)

class PB_Bert(nn.Module):
    def __init__(self,classNum,vocabLen):
        self.channels = 128

        super(PB_Bert, self).__init__()
        self.embed = nn.Embedding(257,self.channels)
        h_num =128
        self.pretain_model = model.BERTModel(258, num_hiddens=h_num , norm_shape=h_num ,
                        ffn_num_input=h_num , ffn_num_hiddens=256, num_heads=4,
                        num_layers=2, dropout=0.2, key_size=h_num ,query_size=h_num ,
                        value_size=h_num , hid_in_features=h_num , mlm_in_features=h_num ,max_len=300
                        )
        self.liner_1 = nn.Linear(self.channels,128)
        self.liner_2 = nn.Linear(bytes_num * self.channels,classNum)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.cnn_pool = nn.Sequential(
            nn.Conv1d(in_channels=self.channels,out_channels=self.channels * 2 ,kernel_size=3,stride=1,padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2,stride=2)
        )

    def forward(self, X):
        #print("X1:", X.shape)
        X = self.embed(X)
        X = torch.transpose(X,1,2)
       # print("X2:",X.shape)
        Y =self.cnn_pool(X)
        #print("Y1:",Y.shape)

        Y = Y.view(-1, bytes_num * 128)
        Y = self.relu(Y)
       # print("Y2:",Y.shape)
        Y = self.dropout(Y)
       # print("Y3:",Y.shape)
        Y = self.liner_2(Y)
       # print("Y4:",Y.shape)
        return Y


mydataSet = flowDataset()
classNum = mydataSet.getClassNum()
tran_size,test_size = int(0.7 * len(mydataSet)),len(mydataSet)-int(0.7 * len(mydataSet))
train_dataset, test_dataset = random_split(mydataSet,[tran_size,test_size])
train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=128,shuffle=True)

#---------------------------------------------
#--------------model inhert-------------------

PB_model = PB_Bert(classNum,vocabLen=mydataSet.getVocabLen())
model_dict = PB_model.state_dict()
#print("model_dict:",model_dict)
# pretrained_dict = torch.load('pretain-bert_1.pt')
#pretrained_dict = torch.load('E:\code\Bert\pretain-bert.pt')
pretrained_dict = torch.load('pretrain-1.pt')
h_num =128
temp_model = model.BERTModel(259, num_hiddens=h_num , norm_shape=h_num ,
                    ffn_num_input=h_num , ffn_num_hiddens=256, num_heads=4,
                    num_layers=2, dropout=0.2, key_size=h_num ,query_size=h_num ,
                    value_size=h_num , hid_in_features=h_num , mlm_in_features=h_num ,max_len=60
                    )
temp_model.load_state_dict(pretrained_dict)
temp_model_dict = temp_model.state_dict()
new_model_dict = PB_model.state_dict()
new_dict = {k: v for k, v in temp_model_dict.items() if k in new_model_dict.keys()}
new_model_dict.update(new_dict)
#print("new_model_dict:",new_model_dict)
PB_model.load_state_dict(new_model_dict)


print("模型参数量：",len(PB_model.state_dict()))


# print(PB_model)
model_ = torchkeras.KerasModel(PB_model,loss_fn=nn.CrossEntropyLoss(),
                               metrics_dict = {"acc":torchmetrics.Accuracy(task='multiclass',num_classes=classNum),
                                               "recall": torchmetrics.Recall(task='multiclass',num_classes=classNum),
                                               "f1": torchmetrics.F1Score(task='multiclass', num_classes=classNum)
                                               }
                               )
dfhistory=model_.fit(train_data=train_dataloader,
                    val_data=test_dataloader,
                    epochs=2,
                    patience=100,
                    # ckpt_path='checkpoint.pt',
                    monitor="val_acc",
                    mode="max")
print(dfhistory)

## FLOPS
with torch.cuda.device(0):
    input = torch.randint(1, 100,(1,60),dtype=torch.long)
    flops, params = profile(PB_model, (input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops,params)

# y_pred = []
# y_true = []
# # iterate over test data
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# for inputs, labels in test_dataloader:
#     output = PB_model(inputs.to(device))  # Feed Network
#
#     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#     y_pred.extend(output)  # Save Prediction
#
#     labels = labels.data.cpu().numpy()
#     y_true.extend(labels)
# ac =classification_report(y_pred , y_true,digits=4)
# print(ac)
# print(precision_score(y_pred , y_true,average='macro'))

# total = sum([param.nelement() for param in PB_model.parameters()])
# print("Number of parameter: %.2fM" % (total/1e6))
# print(type(ac))

#### confusion_matrix
# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd
# import matplotlib.pyplot as plt
#
# cf_matrix = confusion_matrix(y_true, y_pred)
# #/ np.sum(cf_matrix, axis=1)
# print(cf_matrix)
# df_cm = pd.DataFrame(cf_matrix , index = [i for i in range(classNum)],
#                      columns = [i for i in range(classNum)])
# plt.figure(figsize = (12,10))
# sn.heatmap(df_cm, annot=False,cmap='Blues', vmax=1300,vmin=100)
# plt.savefig('output.png')