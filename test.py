import pathlib
import PreprocessBeforeFeature

if __name__ == '__main__':
    #1. 使用splitcap从原始流量中提取单个的流

    dirnameInput = pathlib.Path(r"E:\流量\常规Tor表面网")
    dirnameOutput = pathlib.Path(r"E:\处理过的流量")

    #dirnameInput = pathlib.Path(r"E:\测试\未分流")
    #dirnameOutput = pathlib.Path(r"E:\测试\分流后")

    print("\n2.使用splitcap将原始流量分割成单个的流")
    print("\t原始流量样本的根目录：{}".format(str(dirnameInput)))
    print("\t分割后流量样本的根目录：{}".format(str(dirnameOutput)))
    # time_s = time.time()
    PreprocessBeforeFeature.ExtractFlow(str(dirnameInput),str(dirnameOutput))
    # time_e = time.time()
    # print("\t运行时长 = {}".format(time_e - time_s))
    
    #2. 从多个流中选择最大的作为对应网页的流
    # 同一个网页的原始流量中，可能包含多个流
    
    #dirnameInput = pathlib.Path(r"F:\处理过的流量\Non-Tor")
    #dirnameOutput = pathlib.Path(r"F:\maxFlow\Non-Tor")

    dirnameInput = pathlib.Path(r"E:\处理过的流量")
    dirnameOutput = pathlib.Path(r"E:\maxFlow")

    # dirnameInput = pathlib.Path(r"E:\测试\分流后")
    # dirnameOutput = pathlib.Path(r"E:\测试\最大流量")

    print("\n3.从多个流中选择最大的作为对应网页的流")
    print("\tFlow的根目录：{}".format(str(dirnameInput)))
    print("\tFlowSingle的根目录：{}".format(str(dirnameOutput)))
    PreprocessBeforeFeature.SelectMaxFlow(dirnameInput, dirnameOutput)

    PreprocessBeforeFeature.CheckSinglePcap(dirnameOutput)


    