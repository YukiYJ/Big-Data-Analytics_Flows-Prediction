import math
import torch
import torch.utils.data as Data
from torch import nn,optim
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import time

num_epoches = 50
learning_rate = 0.001
batch_size = 64

torch.set_default_tensor_type('torch.DoubleTensor')

class CNN(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(CNN,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim,256,kernel_size=3,stride=1,padding=1),   
            nn.ReLU(True),      #激活函数
            #nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            #nn.ReLU(True),
            nn.Conv2d(256,out_dim,kernel_size=3,stride=1,padding=1),
            nn.ReLU(True)
        )
        
    def forward(self, x):
        out = self.conv(x)
        self.dropout = nn.Dropout(0.5) 
        return out

def cnnModule(Input, Output, m):
    Input, Output = torch.from_numpy(Input), torch.from_numpy(Output)
    dataset = Data.TensorDataset(Input, Output)
    loader = Data.DataLoader(
        dataset=dataset,      # torch TensorDataset format
        batch_size=batch_size,      # mini batch size
        shuffle=True,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )
    print("CNN regression Module training:\n")
    cnn=CNN(10,2)

    if torch.cuda.is_available():       #是否可用GPU计算
        cnn=cnn.cuda()           #转换成可用GPU计算的模型
    
    MSEcriterion=nn.MSELoss() #cost function
    L1criterion=nn.L1Loss()
    optimizer = optim.Adam(cnn.parameters(), lr=learning_rate)

    MSElist = []
    L1list = []
    for epoch in range(num_epoches):
        start = time.time()
        MSErunning_loss = 0.0
        L1running_loss = 0.0
        total = 0.0
        for step, (features, flows) in enumerate(loader):
            if torch.cuda.is_available():
                features = Variable(features).cuda()
                flows = Variable(flows).cuda()
            else:
                features = Variable(features)
                flows = Variable(flows)
            out = cnn(features)
            MSEloss = MSEcriterion(out,flows)
            L1loss = L1criterion(out, flows)
            MSErunning_loss += MSEloss.item()*features.shape[0]
            L1running_loss += L1loss.item()*features.shape[0]
            optimizer.zero_grad()
            MSEloss.backward()
            optimizer.step()
            total += features.shape[0]
        stop = time.time()
        print('epoch{}'.format(epoch+1))
        print('* '*20)
        print('Finish {} epoch, MSELoss:{:.6f}, RMSELoss:{:.6f}, L1Loss:{:.6f}, with {:.3f} seconds'.format(epoch+1,MSErunning_loss/m, math.sqrt(MSErunning_loss/m), L1running_loss/m, stop-start))
        print()
        MSElist.append(MSErunning_loss/m)
        L1list.append(L1running_loss/m)

    return cnn, MSElist, L1list

def Test(Input, Output, m, cnn):
    print("CNN regression module test:")
    Input, Output = torch.from_numpy(Input), torch.from_numpy(Output)
    dataset = Data.TensorDataset(Input, Output)
    loader = Data.DataLoader(
        dataset=dataset,      # torch TensorDataset format
        batch_size=batch_size,      # mini batch size
        shuffle=False,               # 要不要打乱数据 (打乱比较好)
        num_workers=2,              # 多线程来读数据
    )
    if torch.cuda.is_available():
        cnn=cnn.cuda() 
        
    MSEcriterion=nn.MSELoss() #cost function
    L1criterion=nn.L1Loss()
    
    MSEloss = 0.0
    L1loss = 0.0

    inflows = []
    outflows = []
    start = time.time()
    for step, (features, flows) in enumerate(loader):
        if torch.cuda.is_available():
            features = Variable(features).cuda()
            flows = Variable(flows).cuda()
        else:
            features = Variable(features)
            flows = Variable(flows)
        out = cnn(features)
        MSEloss += MSEcriterion(out,flows).item()*features.shape[0]
        L1loss += L1criterion(out,flows).item()*features.shape[0]
        inflows.append(out[:,0])
        outflows.append(out[:,1])
    stop = time.time()
    inflows = torch.cat(tuple(inflows), 0)
    outflows = torch.cat(tuple(outflows), 0)
    print('Test Result:')
    print('MSELoss:{:.6f}, RMSELoss:{:.6f}, L1Loss:{:.6f}, with {:.3f} seconds'.format(MSEloss/m, math.sqrt(MSEloss/m), L1loss/m, stop-start))
    return inflows, outflows, torch.cuda.is_available()
