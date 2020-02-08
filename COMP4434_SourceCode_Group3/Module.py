import os
import matplotlib.pyplot as plt
import h5py
import numpy as np
import time
import datetime
from cnn import *
testrate = 0.2
grid = 32

def GetFile(datafile, metefile):
    FileData=dict()
    f1=h5py.File(datafile, "r")
    for key, value in f1.items():
        FileData[key]=value[:]   # data (7220, 2, 32, 32) date (7220, )
    f2=h5py.File(metefile, "r")
    for key, value in f2.items():  
        FileData[key]=value[:]  # Temperature (7220,) Weather (7220, 17)  WindSpeed (7220,)
    f1.close()
    f2.close()
    return FileData

def lastdate(today):
    today=str(today, encoding = "utf-8")
    today=datetime.datetime.strptime(today,'%Y%m%d')
    yesterday=today-datetime.timedelta(days=1)
    yesterday=yesterday.strftime("%Y%m%d")
    return bytes(yesterday, encoding = "utf8")

def GetFeatures_cnn(Data):
    data=Data["data"]
    shapes=data.shape
    date=list(Data["date"])
    todelete=[]
    Temperature=Data["Temperature"].repeat(grid*grid).reshape((shapes[0],1,grid,grid))
    WindSpeed=Data["WindSpeed"].repeat(grid*grid).reshape((shapes[0],1,grid,grid))
    Lastdays=[]
    for i in range(3,shapes[0]):
        yesterday=lastdate(date[i][:-2])+date[i][-2:]
        if(yesterday not in date):
            todelete.append(i-3)
            Lastdays.append(np.ones([2,32,32],dtype=int))
        else:
            Lastdays.append(data[date.index(yesterday)])
    Lastdays=np.array(Lastdays)
    features=np.hstack((data[:-3], data[1:-2], data[2:-1], Lastdays, Temperature[3:],WindSpeed[3:])) 
    flows=data[3:]
    #delete the pairs of features which have missing values.
    for i in range(3,shapes[0]):
        now=int(date[i][-2:])
        last1=int(date[i-1][-2:])
        last2=int(date[i-2][-2:])
        last3=int(date[i-3][-2:])
        if (now-last1)%48 == 1 and (now-last2)%48 == 2 and (now-last3)%48 == 3:
            continue
        todelete.append(i-3)
    todelete=sorted(set(todelete), key=todelete.index)
    flows=np.delete(flows,todelete,axis=0)
    features=np.delete(features,todelete,axis=0)
    return features, flows, flows.shape[0]

def TrainAndTest(m,Input,Output):
    testnum=int(m*testrate)
    train={"input":Input[:m-testnum], "output":Output[:m-testnum]}
    test={"input":Input[m-testnum:], "output":Output[m-testnum:]}
    return train, test, m-testnum, testnum

def Module(Data, m):
    Input=Data["input"]
    Output=Data["output"]
    module=cnnModule(Input, Output, m)
    return module
        
def TestModule(Data, m, module):
    Input=Data["input"]
    Output=Data["output"]
    inflows, outflows, cdu = Test(Input, Output, m, module)
    if cdu:
        inflows=inflows.cpu().detach().numpy()
        outflows=outflows.cpu().detach().numpy()
    else:
        inflows=inflows.detach().numpy()
        outflows=outflows.detach().numpy()
    return inflows, outflows

def ResultVisual(MSElist, L1list):
    epoch=list(range(1,len(MSElist)+1))

    plt.figure(figsize=(15,12))
    plt.bar(epoch, L1list, alpha=0.7)
    plt.ylabel('L1loss')
    plt.xlabel('epoch times',fontsize=20)
    plt.grid(True)
    plt.title("L1Loss changes along epoch times")
    plt.savefig("L1Loss.png")

    plt.figure(figsize=(15,12))
    plt.bar(epoch, MSElist, alpha=0.7)
    plt.ylabel('MSEloss')
    plt.xlabel('epoch times',fontsize=20)
    plt.grid(True)
    plt.title("MSELoss changes along epoch times")
    plt.savefig("MSELoss.png")

def DataVisual(Data, name, visual=False):
    Data = Data.flatten()
    maxValue = max(Data)
    mean = np.mean(Data)
    middle = np.median(Data)
    information = name+": Max value: {:.1f}, Mean: {:.1f}, Median: {:.1f}".format(maxValue, mean, middle)
    print(information)
    if visual == True:
        plt.figure(figsize=(15,12))
        plt.hist(Data, bins=80, alpha=0.7)
        plt.xlabel("Flows")
        plt.ylabel("Number of timeplots")
        plt.title(name)
        plt.grid(True)
        plt.savefig(name+".png")

def CompareVisual(Data, PData, name):
    Data = Data.flatten()
    PData = PData.flatten()
    plt.grid(True)
    plt.figure(figsize=(15,12))
    plt.hist(Data, bins=80, color="#FF0000", alpha=0.9, label="observed")
    plt.hist(PData, bins=80, color="#C1F320", alpha=0.5, label="predicted")
    plt.xlabel("Flows")
    plt.ylabel("Number of timeplots")
    plt.title(name)
    plt.legend(loc="upper right")
    plt.savefig(name+".png")

def main():
    datafile="BJ16_M32x32_T30_InOut.h5"
    metefile="BJ_Meteorology.h5"
    while(True):
        file=input("Please enter files to train the model (in the format of \"FlowFile MeteorologyFile\", enter n to use the default input files)\n>")
        if file=='n':
            break
        else:
            files = file.split()
            if len(files) != 2:
                print("Invalid number of files.")
                continue
            datafile = files[0]
            metefile = files[1]
            if os.path.exists(datafile) and os.path.exists(metefile):
                break
    print()
    FileData=GetFile(datafile, metefile)
    features,flows,m=GetFeatures_cnn(FileData)
    train,test,trainnum,testnum=TrainAndTest(m,features,flows)
    DataVisual(train["output"][:,0], "Training Inflows Distribution", True)
    DataVisual(train["output"][:,1], "Training Outflows Distribution", True)
    print()
    module, MSElist, L1list=Module(train, trainnum)
    ResultVisual(MSElist, L1list)
    DataVisual(test["output"][:,0], "Test Inflows Distribution")
    DataVisual(test["output"][:,1], "Test Outflows Distribution")
    print()
    inflows, outflows = TestModule(test, testnum, module)
    DataVisual(inflows, "Test Predicted Inflows Distribution")
    DataVisual(outflows, "Test Predicted Outflows Distribution")
    print()
    CompareVisual(test["output"][:,0], inflows, "Observed Inflows vs Predicted Inflows")
    CompareVisual(test["output"][:,1], outflows, "Observed Outflows vs Predicted Outflows")
    print("                  -Bye-")

if __name__ == "__main__":
    main()
