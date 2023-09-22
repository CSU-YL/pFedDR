import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision.models as models
import torch.distributions as distributions
import numpy as np
from collections import defaultdict, OrderedDict

from generalated.FedSR.util import *

class SqueezeLastTwo(nn.Module):
    """A module which squeezes the last two dimensions, ordinary squeeze can be a problem for batch size 1"""
    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])

class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        for name in args.__dict__:
            setattr(self,name,getattr(args,name))

        out_dim = 2*args.z_dim if self.probabilistic else args.z_dim
        
        if args.back_bone == 'smallcnn':
            class Flatten(nn.Module):
                def forward(self,x):
                    return x.view(-1, 1024)

            net = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
                #nn.Conv2d(1, 32, kernel_size=5, stride=1, bias=True), nn.ReLU(), nn.MaxPool2d(2, 2),
                #nn.Conv2d(32, 64, kernel_size=5, stride=1, bias=True), nn.ReLU(), nn.MaxPool2d(2, 2),
                Flatten(),
                nn.Linear(1024, out_dim))
            torch.nn.init.xavier_uniform_(net[0].weight)
            torch.nn.init.xavier_uniform_(net[4].weight)
            torch.nn.init.xavier_uniform_(net[4].weight)
            net[9].bias.data.zero_()

        elif args.back_bone == 'mediumcnn':
            net = nn.Sequential(
                    nn.Conv2d(1, 64, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(8, 64),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(8, 128),
                    nn.Conv2d(128, 128, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(8, 128),
                    nn.Conv2d(128, out_dim, 3, 1, padding=1),
                    nn.ReLU(),
                    nn.GroupNorm(8, out_dim),
                    nn.AdaptiveAvgPool2d((1,1)),
                    SqueezeLastTwo(),
                    )
        elif args.back_bone == 'alexnet':
            net = models.alexnet(pretrained=True)
            net.classifier[6] = nn.Linear(net.classifier[6].in_features,out_dim)
        elif args.back_bone == 'resnet18':
            net = models.resnet18(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features,out_dim)
        elif args.back_bone == 'resnet50':
            net = models.resnet50(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features,out_dim)
        else:
            raise NotImplementedError

        self.net = net

        self.cls = nn.Linear(args.z_dim,args.num_classes)

        self.net.to(args.device)
        self.cls.to(args.device)
        self.model = nn.Sequential(self.net,self.cls)

        if args.optim == 'SGD':
            self.optim = torch.optim.SGD( 
                self.model.parameters(),
                lr=self.lr,
                momentum=0.9,
                weight_decay=self.weight_decay)
        elif args.optim == 'Adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def communicate(self,master):
        state_dict = self.model.state_dict()
        for key in state_dict:
            p = state_dict[key]
            ps = all_gather(p,stack=False)
            
            # exclude params of master node:
            ps = ps[:master] + ps[master+1:]
            p_avg = sum(ps)/len(ps)
            #p_avg = torch.stack(ps).mean(0)
            state_dict[key] = p_avg
        self.model.load_state_dict(state_dict)

    def featurize(self,x,num_samples=1,return_dist=False):
        if not self.probabilistic:
            return self.net(x)
        else:
            z_params = self.net(x)
            z_mu = z_params[:,:self.z_dim]
            z_sigma = F.softplus(z_params[:,self.z_dim:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
            z = z_dist.rsample([num_samples]).view([-1,self.z_dim])
            
            if return_dist:
                return z, (z_mu,z_sigma)
            else:
                return z

    def forward(self, x):
        if not self.probabilistic:
            return self.model(x)
        else:
            if self.training:
                z = self.featurize(x)
                return self.cls(z)
            else:
                z = self.featurize(x,num_samples=self.num_samples)
                preds = torch.softmax(self.cls(z),dim=1)
                preds = preds.view([self.num_samples,-1,self.num_classes]).mean(0)
                return torch.log(preds)

    def state_dict(self):
        state_dict = {'model_state_dict':self.model.state_dict(),
                        'optim_state_dict':self.optim.state_dict()}
        return state_dict
    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optim.load_state_dict(state_dict['optim_state_dict'])

