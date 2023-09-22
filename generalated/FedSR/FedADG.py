import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as dist

import copy
import numpy as np
from collections import defaultdict, OrderedDict

from generalated.FedSR.base import *
from generalated.FedSR.util import *


class Model(Base):
    def __init__(self, args):
        self.probabilistic = False
        super(Model, self).__init__(args)
        self.noise_dim = 10
        self.G = nn.Sequential(
                nn.Linear(self.noise_dim,self.z_dim//8),
                nn.BatchNorm1d(self.z_dim//8),
                nn.ReLU(),
                nn.Linear(self.z_dim//8,self.z_dim//4),
                nn.BatchNorm1d(self.z_dim//4),
                nn.ReLU(),
                nn.Linear(self.z_dim//4,self.z_dim//2),
                nn.BatchNorm1d(self.z_dim//2),
                nn.ReLU(),
                nn.Linear(self.z_dim//2,self.z_dim),
                )
        self.optim.add_param_group({'params':self.G.parameters(),'lr':self.lr,'momentum':0.9})

        self.D = nn.Sequential(
                #nn.Linear(self.z_dim,self.z_dim//2),
                #nn.BatchNorm1d(self.z_dim//2),
                #nn.ReLU(),
                #nn.Linear(self.z_dim//2,self.z_dim//4),
                #nn.BatchNorm1d(self.z_dim//4),
                #nn.ReLU(),
                nn.Linear(self.z_dim,self.z_dim//8),
                nn.BatchNorm1d(self.z_dim//8),
                nn.ReLU(),
                nn.Linear(self.z_dim//8,1),
                nn.Sigmoid(),
                )
        if args.optim == 'SGD':
            self.D_optim = torch.optim.SGD( 
                self.D.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay)
        elif args.optim == 'Adam':
            self.D_optim = torch.optim.Adam(
                self.D.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def train_client(self,loader,steps=1):
        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        DlossMeter = AverageMeter()
        DaccMeter = AverageMeter()
        for step in range(steps):
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.featurize(x)
            logits = self.cls(z)
            loss = F.cross_entropy(logits,y)

            noise = torch.rand([x.shape[0],self.noise_dim]).to(self.device)
            z_fake = self.G(noise)

            D_inp = torch.cat([z_fake,z])
            D_target = torch.cat([torch.zeros([x.shape[0],1]),torch.ones([x.shape[0],1])]).to(self.device)
            
            # Train D
            D_out = self.D(D_inp.detach())
            D_loss = ((D_target-D_out)**2).mean()

            self.D_optim.zero_grad()
            D_loss.backward()
            self.D_optim.step()

            # Train Net
            D_out = self.D(D_inp)
            #D_loss_g = ((1-D_out)**2).mean()
            D_loss_g = -((D_target-D_out)**2).mean()
            obj = loss + self.D_beta * D_loss_g

            self.optim.zero_grad()
            obj.backward()
            self.optim.step()


            acc = (logits.argmax(1)==y).float().mean()
            D_acc = ((D_out>0.5).long() == D_target).float().mean()
            lossMeter.update(loss.data,x.shape[0])
            accMeter.update(acc.data,x.shape[0])
            DlossMeter.update(D_loss.data,x.shape[0])
            DaccMeter.update(D_acc.data,x.shape[0])

        return {'acc': accMeter.average(), 'loss': lossMeter.average(), 'Dacc': DaccMeter.average(), 'Dloss': DlossMeter.average()}

