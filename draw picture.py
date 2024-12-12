import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import csv
file=r"bytes_num.csv"
df=pd.read_csv(file)
#防止中文乱码
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
x1=df.iloc[0:15,0]
#x2=df.iloc[15:,0]
y1=df.iloc[0:15,1]   #另一个是20
#y2=df.iloc[15:,1]
plt.plot(x1,y1,label='Byte',color='blue', linewidth=2,marker='o')
#plt.plot(x2,y2,label='T1',color='red', linewidth=2,marker='^')
#plt.title('截断长度')
plt.xlabel('字节窗口')
plt.ylabel('准确率')
plt.xlim(0,160)
plt.ylim(0.2,1)
plt.legend()
plt.savefig("bytewindows.png")