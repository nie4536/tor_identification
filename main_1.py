import torch
from torch import nn
import model
# from torchsummary import summary
from d2l.torch import d2l
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, accuracy_score
import torchkeras
import torchmetrics
import csv
from thop import profile,clever_format
torch.cuda.device_count()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dateset(Dataset):
    def __init__(self):
        global bytes_num,start_byte
        with open('E:/code/pretraining/ET-BERT_data.txt','r') as f:
            lines = f.readlines()
            # print(len(lines))
        self.labels = [line.strip().split('\t')[-1] for line in lines if len(line.strip().split('\t')[0].split(' '))>=start_byte+bytes_num]
        label_map = {label: index for index, label in enumerate(set(self.labels))}
        self.labels = list(map(lambda x: label_map[x], self.labels))
        # print(lines)
        flowBytes = [line.strip().split('\t')[0].split(' ')[start_byte:start_byte+bytes_num] for line in lines if len(line.strip().split('\t')[0].split(' '))>=start_byte+bytes_num]
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

bytes_num = 0
start_byte = 0
class flowDataset(Dataset):
    def __init__(self):
        global bytes_num,start_byte
        with open('E:/code/pretraining/fine_tuning_data.txt','r') as f:
            lines = f.readlines()
            # print(len(lines))
        self.labels = [line.strip().split('\t')[-1] for line in lines if len(line.strip().split('\t')[0].split(' '))>=start_byte+bytes_num]
        label_map = {label: index for index, label in enumerate(set(self.labels))}
        self.labels = list(map(lambda x: label_map[x], self.labels))
        # print(lines)
        flowBytes = [line.strip().split('\t')[0].split(' ')[start_byte:start_byte+bytes_num] for line in lines if len(line.strip().split('\t')[0].split(' '))>=start_byte+bytes_num]
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


class ET_Bert(nn.Module):
    def __init__(self,classNum,vocabLen):
        self.channels = 128
        self.classNum = classNum
        super(ET_Bert, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.embed = nn.Embedding(256,self.channels)
        h_num =128
        self.pretain_model = model.BERTModel(258, num_hiddens=h_num , norm_shape=h_num ,
                        ffn_num_input=h_num , ffn_num_hiddens=256, num_heads=4,
                        num_layers=2, dropout=0.2, key_size=h_num ,query_size=h_num ,
                        value_size=h_num , hid_in_features=h_num , mlm_in_features=h_num ,max_len=80
                        )
        self.liner_1 = nn.Linear(self.channels,128)
        # num_features = torch.numel(self.pool(torch.randn(1, *x.shape[1:])))
        num_features = int(bytes_num/2)* self.channels *2
        # print(num_features)
        self.liner_2 = nn.Linear(num_features,classNum)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU(inplace=True)
        self.cnn_pool = nn.Sequential(
            nn.Conv1d(in_channels=self.channels, out_channels=self.channels * 2, kernel_size=3, stride=1,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # self.bilstm = nn.LSTM(input_size=self.channels,hidden_size=self.channels,batch_first=True,bidirectional=True)
        # self.Encoders = d2l.TransformerEncoder(vocab_size=self.channels, key_size=self.channels, query_size=self.channels, value_size=self.channels,
        #          num_hiddens=self.channels, norm_shape=self.channels, ffn_num_input=self.channels, ffn_num_hiddens=self.channels,
        #          num_heads=8, num_layers=2, dropout=0.1, use_bias=True)

    def forward(self, X):
        # X = self.embed(X)
        # print(X.shape)
        X = self.pretain_model(X)
        X = torch.transpose(X, 1, 2)
        Y = self.cnn_pool(X)
        # Y = self.cnn_pool_2(Y)
        # print(Y.shape)
        # print(Y.shape)

        Y = Y.view(-1, Y.shape[-2] * Y.shape[-1])
        # print(Y.shape)
        Y = self.dropout(Y)
        # self.liner_2 = nn.Linear(Y.shape[-1], self.classNum ,device=self.device)
        Y = self.liner_2(Y)
        # Y = nn.Linear(Y.shape[1], self.classNum,device=self.device)(Y)
        return Y

# file = open(r'bytes_num_t1.csv',mode='a+',encoding='utf-8',newline='')
# swriter = csv.writer(file)
# for n in range(50,100,5):
bytes_num = 60
mydataSet = flowDataset()
classNum = mydataSet.getClassNum()
tran_size,test_size = int(0.8 * len(mydataSet)),len(mydataSet)-int(0.8 * len(mydataSet))
train_dataset, test_dataset = random_split(mydataSet,[tran_size,test_size])
train_dataloader = DataLoader(train_dataset,batch_size=128,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=128,shuffle=False)

#---------------------------------------------
#--------------model inhert-------------------

ET_model = ET_Bert(classNum,vocabLen=mydataSet.getVocabLen()).to(device)
model_dict = ET_model.state_dict()
#print("model_dict:",model_dict)
# pretrained_dict = torch.load('pretain-bert_1.pt',map_location=device)
pretrained_dict = torch.load('pretrain-1.pt',map_location=device)
h_num =128
temp_model = model.BERTModel(259, num_hiddens=h_num , norm_shape=h_num ,
                    ffn_num_input=h_num , ffn_num_hiddens=256, num_heads=4,
                    num_layers=2, dropout=0.2, key_size=h_num ,query_size=h_num ,
                    value_size=h_num , hid_in_features=h_num , mlm_in_features=h_num ,max_len=60
                    )
temp_model.load_state_dict(pretrained_dict)
temp_model_dict = temp_model.state_dict()
new_model_dict = ET_model.state_dict()
new_dict = {k: v for k, v in temp_model_dict.items() if k in new_model_dict.keys()}
new_model_dict.update(new_dict)
ET_model.load_state_dict(new_model_dict)
#print("PB_model_dict:",PB_model.state_dict())


model_ = torchkeras.KerasModel(ET_model,loss_fn=nn.CrossEntropyLoss(),
                               metrics_dict = {"acc":torchmetrics.Accuracy(task='multiclass',num_classes=classNum),
                                               "recall": torchmetrics.Recall(task='multiclass', num_classes=classNum),
                                               "f1": torchmetrics.F1Score(task='multiclass', num_classes=classNum)
                                               })
dfhistory=model_.fit(train_data=train_dataloader,
                    val_data=test_dataloader,
                    epochs=2,
                    patience=100,
                    # ckpt_path='checkpoint.pt',
                    monitor="val_acc",
                    mode="max")
print(dfhistory)

#-------------------------------------------------------------------
with torch.cuda.device(0):
    input = torch.randint(1, 100,(1,60),dtype=torch.long)
    flops, params = profile(ET_model, (input, ))
    flops, params = clever_format([flops, params], "%.3f")
    print("10.821G 133.829M")
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