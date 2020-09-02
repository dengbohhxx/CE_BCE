#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn.functional as F
import torch.nn as nn
class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pred,label):
        pred=F.softmax(pred,dim=1)
        one_hot_label=F.one_hot(label,pred.size(1)).float()
        loss=-1*(one_hot_label*torch.log(pred))
        loss=loss.sum(axis=1,keepdim=False)
        loss=loss.mean()
        return loss
class BCEloss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,pred,label):
        loss=-1*(label*torch.log(pred)+(1-label)*torch.log(1-pred))
        return loss.mean()

if __name__=='__main__':
    torch.manual_seed(32)
    pred=torch.randn(3,3)
    print(pred)
    label=torch.randint(0,3,(3,))
    print(label)
    CEloss=torch.nn.CrossEntropyLoss()
    loss=CEloss(pred,label)
    print(f'CrossEntropyLoss:{loss}')
    celoss=CELoss()
    loss1=celoss(pred,label)
    print(f'class CELoss:{loss1}')
    sig=nn.Sigmoid()
    bceloss=nn.BCELoss()
    loss=bceloss(sig(pred),F.one_hot(label,pred.size(1)).float())
    print(f'BCELoss:{loss}')
    bceloss1=BCEloss()
    loss=bceloss1(sig(pred),F.one_hot(label,pred.size(1)).float())
    print(f'BCEloss:{loss}')
