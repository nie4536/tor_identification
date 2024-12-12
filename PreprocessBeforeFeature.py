import pathlib
import subprocess
import shutil
import os

# 使用splitcap将原始pcap包分流
# SplitCap -r %%i -o %flow%
splitCap = r"E:\Flow\SplitCap.exe"

def ExtractFlow(dirnameInput, dirnameOutput):
    dirnameInput = pathlib.Path(dirnameInput)
    dirnameOutput = pathlib.Path(dirnameOutput)
    numb = 0
    pool = []
    poolLength = 0
    for date in dirnameInput.iterdir():
        if date.is_dir():
            numb = numb + 1
            # if numb > 20 :
            #     break
            tempInput = dirnameInput /date.name
            tempOutput = dirnameOutput /date.name
            # print(str(tempInput))
            if not tempOutput.exists(): # 目录不存在，则递归创建目录;若存在，则认为已经对这个目录的流量运行过splitcap了。
                tempOutput.mkdir(parents=True)
                # while(True):
                cmd = "{} -r {} -recursive -o {}".format(splitCap,str(tempInput),str(tempOutput))
                # os.popen(cmd)
                p = subprocess.Popen(cmd,shell=True)
                pool.append([numb,p])
                poolLength = poolLength + 1
                if poolLength >= 5 :
                    for i in pool:
                        i[1].wait()
                    print("\tSplitCap任务 {} 完成".format(pool[-1][0]))
                    del pool
                    pool = []
                    poolLength = 0
    if poolLength > 0 :
        for i in pool:
            i[1].wait()

        print("\tSplitCap任务 {} 完成".format(pool[-1][0]))
                
# 从多个流中选择最大的作为对应网页的流
def SelectMaxFlow(dirnameInput, dirnameOutput):
    dirnameInput = pathlib.Path(dirnameInput)
    dirnameOutput = pathlib.Path(dirnameOutput)
    numb = 0
    # for ip in dirnameInput.iterdir():
    #     if ip.is_dir():
    for date in dirnameInput.iterdir():
        if date.is_dir():
            numb = numb + 1
            # if numb > 5:
            #     break
            webIDs = {}
            for traffic in date.iterdir():
                #webID = (traffic.name).split("+")[0]
                webID = (traffic.name).split(".")[0]
                if webIDs.get(webID, 0) == 0:
                    webIDs[webID] = []

                    webIDs[webID].append(str(traffic))
                    webIDs[webID].append(traffic.stat().st_size)
                else:
                    if traffic.stat().st_size > webIDs[webID][1] :
                        webIDs[webID][0] = str(traffic)
                        webIDs[webID][1] = traffic.stat().st_size
            tempOutput = dirnameOutput / date.name
            if not tempOutput.exists(): # 目录不存在，则递归创建目录
                tempOutput.mkdir(parents=True)
            for i in webIDs:
                isExist = False
                for j in tempOutput.glob("{}+*.pcap".format(webIDs[i][0].split("\\")[-1].split("+")[0])):
                    isExist = True
                    break
                if not isExist:
                    shutil.move(str(webIDs[i][0]), str(tempOutput) + "\\")
            if numb % 5 == 0 :
                print("\t任务 {} 完成".format(numb))
    print("\t任务 {} 完成".format(numb))

# 检查上一步后，是不是每个网页对应一个pcap文件
def CheckSinglePcap(dirname):
    dirname = pathlib.Path(dirname)
    for ip in dirname.iterdir():
        if ip.is_dir():
            for date in ip.iterdir():
                if date.is_dir():
                    webIDs = []
                    res = []
                    for traffic in date.iterdir():
                        webID = traffic.name.split("+")[0]
                        if webID in webIDs:
                            res.append(webID)
                        else:
                            webIDs.append(webID)
                    if len(res) > 0:
                        print("\t{}:".format(str(date).split(dirname)[-1]))
                        for i in res:
                            print("\t\t{}".format(i.name))
                    