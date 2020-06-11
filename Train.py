from scipy.io import loadmat
import scipy.io as spio
import numpy as np
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision import transforms
import os
import sys
import time
sys.path.append(".")
import GrNet as gr


base_path = "D:\File\GrNet_experiment\AFEW_gr_data\grface_400_inter_histeq"
dataset_path = r"D:\File\GrNet_experiment\AFEW_gr_data"

def evaluate_accuracy(data_iter,net,device=None):
    if device is None and isinstance(net,torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum , n = 0.0 ,0
    with torch.no_grad():
        for i in data_iter:
            X,y = i["data"],i["label"]
            if isinstance(net,torch.nn.Module):
                net.eval()
                acc_sum+=(net(X.to(device)).argmax(dim=1)==y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if("is_training" in net.__code__.co_varnames):
                    acc_sum+=(net(X,is_training=False).argmax(dim=1)==y).float().sum().item()
                else:
                    acc_sum+=(net(X).argmax(dim=1)==y).float().sum().item()
            n+=y.shape[0]
    return acc_sum/n


def train(net,train_iter,test_iter,batch_size,optimizer,num_epoch,device="cpu",loss=torch.nn.CrossEntropyLoss()):
    net=net.to(device)
    print("training on ",device)
    for epoch in range(num_epoch):
        train_l_s ,train_acc_sum , n ,batch_count , start = 0.0,0.0,0,0,time.time()
        for i in train_iter:
            X,y = i["data"],i["label"]
            X=X.to(device)
            y=y.to(device)
            y_hat = net(X)
            l=loss(y_hat,y)
            optimizer.zero_grad()
            #deal with FR.w.grad
            #for param in Gr.ManiBlock.parameters():
                #if param.grad is not None:
                    #param.grad.data.zero_()
            l.backward()
            optimizer.step()
            #deal with FR.w.grad
            #for param in Gr.ManiBlock.parameters():
            #  param.data-=LR * param.grad.data
            
            
            train_l_s+=l.cpu().item()
            train_acc_sum+=(y_hat.argmax(dim=1)==y).sum().cpu().item()
            n+=y.shape[0]
            batch_count+=1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_s / batch_count, train_acc_sum / n, test_acc, time.time() - start))

batch, kernel_num, embeddim, pool_size = 32,16,10,4
lr = 0.001

def loadmat_(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict   
def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class AfewDataset(Dataset):
    def __init__(self, shuffle=True, train=False, augmentation=False):
        super(AfewDataset, self).__init__()

        self.train = train
        self.base_path = "D:\File\GrNet_experiment\AFEW_gr_data"
        self.dataset_path = os.path.join(self.base_path, "grdb_afew_train_gr400_10_int_histeq.mat")

        dataset = loadmat_(self.dataset_path)
        self.gr_path = [path.replace('\\', '/') for path in dataset["gr_train"]["gr"]["name"]]
        self.labels = dataset["gr_train"]["gr"]["label"]
        
        gr_set = dataset["gr_train"]["gr"]['set']
        if train:
            self.data_index = np.argwhere(gr_set == 1).squeeze()
        else:
            self.data_index = np.argwhere(gr_set == 2).squeeze()

        if shuffle:
            random.shuffle(self.data_index)

        self.nSamples = len(self.data_index) 
        print("dataset size", self.nSamples)
        self.nClasses = 7
        self.augmentation = augmentation
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, idx):
        index = self.data_index[idx]
        data_path = os.path.join(self.base_path,'grface_400_inter_histeq', self.gr_path[index])
        data = loadmat_(data_path)
        data = torch.from_numpy(data['Y1']).unsqueeze(0)
        label = np.asarray([self.labels[index] - 1])

        sample = {'data': data, 'label': torch.from_numpy(label).squeeze().long()}
        return sample

train_dataset = AfewDataset(train=True)
test_dataset = AfewDataset(train=False)
train_data = DataLoader(train_dataset, batch_size=32,
                    shuffle=True, num_workers=0)
test_data = DataLoader(test_dataset, batch_size=32,
                    shuffle=True, num_workers=0)

class GrNet(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.ManiBlock = gr.ManiBlock(channel)
        self.fc = nn.Linear(8000,7)
        self.softmax = nn.Softmax(1)
    def forward(self, x):
        x = self.ManiBlock(x)
        x = self.fc(x.reshape(x.shape[0],-1))
        x = self.softmax(x)
        return x

Gr = GrNet(1)
optimizer = torch.optim.SGD(Gr.parameters(), lr=lr)
train(Gr,train_data,test_data,batch,optimizer,1)