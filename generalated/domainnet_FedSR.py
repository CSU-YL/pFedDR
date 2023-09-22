"""
federated learning with different aggregation strategy on domainnet dataset
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions

import pickle as pkl
from utils.data_utils import DomainNetDataset
from nets.models_multitask_dbn import AlexNet
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np

class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        class Flatten(nn.Module):
            def forward(self, x):
                return torch.flatten(x, 1)

        out_dim = 2 * args.z_dim if self.probabilistic else args.z_dim

        net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.AdaptiveAvgPool2d((6, 6)),
            Flatten(),
            nn.Linear(256 * 6 * 6, 4096), nn.BatchNorm1d(4096), nn.ReLU(inplace=True),
            nn.Linear(4096, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True),
        )

        self.net = net

        self.cls = nn.Linear(args.z_dim, args.num_classes)

        self.net.to(args.device)
        self.cls.to(args.device)
        self.model = nn.Sequential(self.net, self.cls)

        if args.optim == 'SGD':
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr)
            # self.optim = torch.optim.SGD(
            #     self.model.parameters(),
            #     lr=self.lr,
            #     momentum=0.9,
            #     weight_decay=self.weight_decay)
        elif args.optim == 'Adam':
            self.optim = torch.optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def featurize(self, x, num_samples=1, return_dist=False):
        if not self.probabilistic:
            return self.net(x)
        else:
            z_params = self.net(x)
            z_mu = z_params[:, :self.z_dim]
            z_sigma = F.softplus(z_params[:, self.z_dim:])
            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

            if return_dist:
                return z, (z_mu, z_sigma)
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
                z = self.featurize(x, num_samples=self.num_samples)
                preds = torch.softmax(self.cls(z), dim=1)
                preds = preds.view([self.num_samples, -1, self.num_classes]).mean(0)
                return torch.log(preds)

    def state_dict(self):
        state_dict = {'model_state_dict': self.model.state_dict(),
                      'optim_state_dict': self.optim.state_dict()}
        return state_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.optim.load_state_dict(state_dict['optim_state_dict'])


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0

    def update(self, val, n=1):
        self.count += n
        self.sum += val * n

    def average(self):
        return self.sum / self.count

    def __repr__(self):
        r = self.sum / self.count
        if r < 1e-3:
            return '{:.2e}'.format(r)
        else:
            return '%.4f' % (r)


class FedAVG(Base):
    def __init__(self, args):
        self.probabilistic = False
        super(FedAVG, self).__init__(args)

    def train_client(self, loader, steps=1):
        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        train_iter = iter(loader)
        for step in range(len(train_iter)):
            x, y = next(train_iter)
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = F.cross_entropy(logits, y)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            acc = (logits.argmax(1) == y).float().mean()
            lossMeter.update(loss.data, x.shape[0])
            accMeter.update(acc.data, x.shape[0])
        # print("", {'acc': accMeter.average(), 'loss': lossMeter.average()})
        return {'acc': accMeter.average(), 'loss': lossMeter.average()}


class FedSR(Base):
    def __init__(self, args):
        self.probabilistic = True
        super(FedSR, self).__init__(args)
        self.r_mu = nn.Parameter(torch.zeros(args.num_classes, args.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(args.num_classes, args.z_dim))
        self.C = nn.Parameter(torch.ones([]))
        self.optim.add_param_group({'params': [self.r_mu, self.r_sigma, self.C], 'lr': self.lr, 'momentum': 0.9})

    def train_client(self, loader, steps=1):

        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        regL2RMeter = AverageMeter()
        regCMIMeter = AverageMeter()
        regNegEntMeter = AverageMeter()

        train_iter = iter(loader)
        for step in range(len(train_iter)):
            x, y = next(train_iter)
            x, y = x.to(self.device), y.to(self.device)
            z, (z_mu, z_sigma) = self.featurize(x, return_dist=True)
            logits = self.cls(z)
            loss = F.cross_entropy(logits, y)

            obj = loss
            regL2R = torch.zeros_like(obj)
            regCMI = torch.zeros_like(obj)
            regNegEnt = torch.zeros_like(obj)
            if self.L2R_coeff != 0.0:
                regL2R = z.norm(dim=1).mean()
                obj = obj + self.L2R_coeff * regL2R
            if self.CMI_coeff != 0.0:
                r_sigma_softplus = F.softplus(self.r_sigma)
                r_mu = self.r_mu[y]
                r_sigma = r_sigma_softplus[y]
                z_mu_scaled = z_mu * self.C
                z_sigma_scaled = z_sigma * self.C
                regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + \
                         (z_sigma_scaled ** 2 + (z_mu_scaled - r_mu) ** 2) / (2 * r_sigma ** 2) - 0.5
                regCMI = regCMI.sum(1).mean()
                obj = obj + self.CMI_coeff * regCMI

            z_dist = distributions.Independent(distributions.normal.Normal(z_mu, z_sigma), 1)
            mix_coeff = distributions.categorical.Categorical(x.new_ones(x.shape[0]))
            mixture = distributions.mixture_same_family.MixtureSameFamily(mix_coeff, z_dist)
            log_prob = mixture.log_prob(z)
            regNegEnt = log_prob.mean()

            self.optim.zero_grad()
            # obj.backward()
            loss.backward()
            self.optim.step()

            acc = (logits.argmax(1) == y).float().mean()
            lossMeter.update(loss.data, x.shape[0])
            accMeter.update(acc.data, x.shape[0])
            regL2RMeter.update(regL2R.data, x.shape[0])
            regCMIMeter.update(regCMI.data, x.shape[0])
            regNegEntMeter.update(regNegEnt.data, x.shape[0])

        # print("result:", {'acc': accMeter.average(), 'loss': lossMeter.average(), 'regL2R': regL2RMeter.average(),
        #                   'regCMI': regCMIMeter.average(), 'regNegEnt': regNegEntMeter.average()})
        return {'acc': accMeter.average(), 'loss': lossMeter.average(), 'regL2R': regL2RMeter.average(),
                'regCMI': regCMIMeter.average(), 'regNegEnt': regNegEntMeter.average()}


class FedADG(Base):
    def __init__(self, args):
        self.probabilistic = False
        super(FedADG, self).__init__(args)
        self.noise_dim = 10
        self.G = nn.Sequential(
            nn.Linear(self.noise_dim, self.z_dim // 8),
            nn.BatchNorm1d(self.z_dim // 8),
            nn.ReLU(),
            nn.Linear(self.z_dim // 8, self.z_dim // 4),
            nn.BatchNorm1d(self.z_dim // 4),
            nn.ReLU(),
            nn.Linear(self.z_dim // 4, self.z_dim // 2),
            nn.BatchNorm1d(self.z_dim // 2),
            nn.ReLU(),
            nn.Linear(self.z_dim // 2, self.z_dim),
        )
        # self.optim.add_param_group({'params': self.G.parameters(), 'lr': self.lr, 'momentum': 0.9})
        self.optim.add_param_group({'params': self.G.parameters(), 'lr': self.lr})

        self.D = nn.Sequential(
            # nn.Linear(self.z_dim,self.z_dim//2),
            # nn.BatchNorm1d(self.z_dim//2),
            # nn.ReLU(),
            # nn.Linear(self.z_dim//2,self.z_dim//4),
            # nn.BatchNorm1d(self.z_dim//4),
            # nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim // 8),
            nn.BatchNorm1d(self.z_dim // 8),
            nn.ReLU(),
            nn.Linear(self.z_dim // 8, 1),
            nn.Sigmoid(),
        )
        if args.optim == 'SGD':
            # self.D_optim = torch.optim.SGD(
            #     self.D.parameters(),
            #     lr=self.lr,
            #     momentum=0.9,
            #     weight_decay=self.weight_decay)
            self.D_optim = torch.optim.SGD(
                self.D.parameters(),
                lr=self.lr)
        elif args.optim == 'Adam':
            self.D_optim = torch.optim.Adam(
                self.D.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def train_client(self, loader, steps=1):
        self.train()
        lossMeter = AverageMeter()
        accMeter = AverageMeter()
        DlossMeter = AverageMeter()
        DaccMeter = AverageMeter()
        train_iter = iter(loader)
        for step in range(len(train_iter)):
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.featurize(x)
            logits = self.cls(z)
            loss = F.cross_entropy(logits, y)

            noise = torch.rand([x.shape[0], self.noise_dim]).to(self.device)
            z_fake = self.G(noise)

            D_inp = torch.cat([z_fake, z])
            D_target = torch.cat([torch.zeros([x.shape[0], 1]), torch.ones([x.shape[0], 1])]).to(self.device)

            # Train D
            D_out = self.D(D_inp.detach())
            D_loss = ((D_target - D_out) ** 2).mean()

            self.D_optim.zero_grad()
            D_loss.backward()
            self.D_optim.step()

            # Train Net
            D_out = self.D(D_inp)
            # D_loss_g = ((1-D_out)**2).mean()
            D_loss_g = -((D_target - D_out) ** 2).mean()
            obj = loss + self.D_beta * D_loss_g

            self.optim.zero_grad()
            obj.backward()
            self.optim.step()

            acc = (logits.argmax(1) == y).float().mean()
            D_acc = ((D_out > 0.5).long() == D_target).float().mean()
            lossMeter.update(loss.data, x.shape[0])
            accMeter.update(acc.data, x.shape[0])
            DlossMeter.update(D_loss.data, x.shape[0])
            DaccMeter.update(D_acc.data, x.shape[0])

        return {'acc': accMeter.average(), 'loss': lossMeter.average(), 'Dacc': DaccMeter.average(),
                'Dloss': DlossMeter.average()}



def prepare_data(args):
    data_base_path = '/home/liuyuan/datasets/fedbn_datasets/'
    transform_train = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
    ])

    # clipart
    clipart_trainset = DomainNetDataset(data_base_path, 'clipart', transform=transform_train)
    clipart_testset = DomainNetDataset(data_base_path, 'clipart', transform=transform_test, train=False)
    # infograph
    infograph_trainset = DomainNetDataset(data_base_path, 'infograph', transform=transform_train)
    infograph_testset = DomainNetDataset(data_base_path, 'infograph', transform=transform_test, train=False)
    # painting
    painting_trainset = DomainNetDataset(data_base_path, 'painting', transform=transform_train)
    painting_testset = DomainNetDataset(data_base_path, 'painting', transform=transform_test, train=False)
    # quickdraw
    quickdraw_trainset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_train)
    quickdraw_testset = DomainNetDataset(data_base_path, 'quickdraw', transform=transform_test, train=False)
    # real
    real_trainset = DomainNetDataset(data_base_path, 'real', transform=transform_train)
    real_testset = DomainNetDataset(data_base_path, 'real', transform=transform_test, train=False)
    # sketch
    sketch_trainset = DomainNetDataset(data_base_path, 'sketch', transform=transform_train)
    sketch_testset = DomainNetDataset(data_base_path, 'sketch', transform=transform_test, train=False)

    min_data_len = min(len(clipart_trainset), len(infograph_trainset), len(painting_trainset), len(quickdraw_trainset),
                       len(real_trainset), len(sketch_trainset))
    val_len = int(min_data_len * 0.05)
    min_data_len = int(min_data_len * 0.05)

    clipart_valset = torch.utils.data.Subset(clipart_trainset, list(range(len(clipart_trainset)))[-val_len:])
    clipart_trainset = torch.utils.data.Subset(clipart_trainset, list(range(min_data_len)))

    infograph_valset = torch.utils.data.Subset(infograph_trainset, list(range(len(infograph_trainset)))[-val_len:])
    infograph_trainset = torch.utils.data.Subset(infograph_trainset, list(range(min_data_len)))

    painting_valset = torch.utils.data.Subset(painting_trainset, list(range(len(painting_trainset)))[-val_len:])
    painting_trainset = torch.utils.data.Subset(painting_trainset, list(range(min_data_len)))

    quickdraw_valset = torch.utils.data.Subset(quickdraw_trainset, list(range(len(quickdraw_trainset)))[-val_len:])
    quickdraw_trainset = torch.utils.data.Subset(quickdraw_trainset, list(range(min_data_len)))

    real_valset = torch.utils.data.Subset(real_trainset, list(range(len(real_trainset)))[-val_len:])
    real_trainset = torch.utils.data.Subset(real_trainset, list(range(min_data_len)))

    sketch_valset = torch.utils.data.Subset(sketch_trainset, list(range(len(sketch_trainset)))[-val_len:])
    sketch_trainset = torch.utils.data.Subset(sketch_trainset, list(range(min_data_len)))

    clipart_train_loader = torch.utils.data.DataLoader(clipart_trainset, batch_size=32, shuffle=True)
    clipart_val_loader = torch.utils.data.DataLoader(clipart_valset, batch_size=32, shuffle=False)
    clipart_test_loader = torch.utils.data.DataLoader(clipart_testset, batch_size=32, shuffle=False)

    infograph_train_loader = torch.utils.data.DataLoader(infograph_trainset, batch_size=32, shuffle=True)
    infograph_val_loader = torch.utils.data.DataLoader(infograph_valset, batch_size=32, shuffle=False)
    infograph_test_loader = torch.utils.data.DataLoader(infograph_testset, batch_size=32, shuffle=False)

    painting_train_loader = torch.utils.data.DataLoader(painting_trainset, batch_size=32, shuffle=True)
    painting_val_loader = torch.utils.data.DataLoader(painting_valset, batch_size=32, shuffle=False)
    painting_test_loader = torch.utils.data.DataLoader(painting_testset, batch_size=32, shuffle=False)

    quickdraw_train_loader = torch.utils.data.DataLoader(quickdraw_trainset, batch_size=32, shuffle=True)
    quickdraw_val_loader = torch.utils.data.DataLoader(quickdraw_valset, batch_size=32, shuffle=False)
    quickdraw_test_loader = torch.utils.data.DataLoader(quickdraw_testset, batch_size=32, shuffle=False)

    real_train_loader = torch.utils.data.DataLoader(real_trainset, batch_size=32, shuffle=True)
    real_val_loader = torch.utils.data.DataLoader(real_valset, batch_size=32, shuffle=False)
    real_test_loader = torch.utils.data.DataLoader(real_testset, batch_size=32, shuffle=False)

    sketch_train_loader = torch.utils.data.DataLoader(sketch_trainset, batch_size=32, shuffle=True)
    sketch_val_loader = torch.utils.data.DataLoader(sketch_valset, batch_size=32, shuffle=False)
    sketch_test_loader = torch.utils.data.DataLoader(sketch_testset, batch_size=32, shuffle=False)

    train_loaders = [clipart_train_loader, infograph_train_loader, painting_train_loader, quickdraw_train_loader,
                     real_train_loader, sketch_train_loader]
    val_loaders = [clipart_val_loader, infograph_val_loader, painting_val_loader, quickdraw_val_loader, real_val_loader,
                   sketch_val_loader]
    test_loaders = [clipart_test_loader, infograph_test_loader, painting_test_loader, quickdraw_test_loader,
                    real_test_loader, sketch_test_loader]

    return train_loaders, val_loaders, test_loaders


# for dapfl
def train_multitask(model, data_loader, optimizer, loss_fun, device):
    def multitaskLossFn(output_local, output_global, y, alpha):
        """
        Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
        Return: loss
        """
        if (alpha > 0):
            loss = loss_fun(output_local, y) * alpha + loss_fun(output_global, y) * (1 - alpha)
        else:
            loss = loss_fun(output_global, y)
        return loss

    model.train()
    loss_all = 0
    total = 0
    correct = 0

    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        # output = model(data)
        output_global = model(data, mode='global')
        output_local = model(data, mode='local')

        # loss = loss_fun(output, target)
        loss = multitaskLossFn(output_local, output_global, target, alpha=0.5)
        loss_all += loss.item()
        total += target.size(0)
        pred = output_global.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct / total


def train(model, data_loader, optimizer, loss_fun, device, mode="global"):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct / total


def train_prox(args, model, data_loader, optimizer, loss_fun, device):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for step, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += args.mu / 2. * w_diff

        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


def test(model, data_loader, loss_fun, device, mode='global'):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


################# fedavg ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        for key in server_model.model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                server_model.model.state_dict()[key].data.copy_(models[0].model.state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].model.state_dict()[key]
                server_model.model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].model.state_dict()[key].data.copy_(server_model.model.state_dict()[key])
    return server_model, models


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=300, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mu', type=float, default=1e-3, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/domainnet_dapfl',
                        help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

    parser.add_argument('--mode', type=str, default='FedSR', help='FedAVG | FedSR | FedADG')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--gpu', type=str, default="3", help='gpu')
    parser.add_argument('--unseen_client', type=int, default=5, help='unseen_client')
    #  python dapfl_domainnet.py --log --mode dapfl --unseen_client 3 --gpu 2

    # fedsr
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--z_dim', type=int, default=4096)
    # parser.add_argument('--L2R_coeff', type=float,
    #                     default=HparamsGen('L2R_coeff', 1e-2, lambda r: 10 ** r.uniform(-5, -3)))
    # parser.add_argument('--CMI_coeff', type=float,
    #                     default=HparamsGen('CMI_coeff', 5e-4, lambda r: 10 ** r.uniform(-5, -3)))
    parser.add_argument('--L2R_coeff', type=float, default=0.01)
    parser.add_argument('--CMI_coeff', type=float, default=0.0005)
    parser.add_argument('--D_beta', type=float, default=1)
    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_samples', type=int, default=20)

    args = parser.parse_args()

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    args.num_classes = 10  # digits
    args.device = device  # digits

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    print('Device:', gpu)
    print('Seed:', seed)

    exp_folder = "generalated"

    args.save_path = os.path.join(args.save_path, exp_folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}_seed{}_unseen{}'.format(args.mode, seed, args.unseen_client))

    log = args.log
    if log:
        log_path = os.path.join('../logs/domainnet_dapfl', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, f'{args.mode}-seed{seed}-unseen{args.unseen_client}.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('seed: {}\n'.format(seed))
        logfile.write('args: {}\n'.format(args))
        print('args: {}\n'.format(args))

    # prepare the data
    train_loaders, val_loaders, test_loaders = prepare_data(args)

    # name of each datasets
    datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']

    # pop unseen client
    unseen_dataset = datasets.pop(args.unseen_client)
    print("unseen client:", unseen_dataset)
    print(f"federated seen clients:{datasets}")
    del train_loaders[args.unseen_client]
    del val_loaders[args.unseen_client]
    unseen_test_loader = test_loaders.pop(args.unseen_client)

    # setup model
    # server_model = AlexNet().to(device)
    if args.mode == 'FedSR':
        server_model = FedSR(args).to(device)
    elif args.mode == 'FedAVG':
        server_model = FedAVG(args).to(device)
    elif args.mode == 'FedADG':
        server_model = FedADG(args).to(device)
    print("server_model.state_dict().keys():", server_model.model.state_dict().keys())

    loss_fun = nn.CrossEntropyLoss()
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]
    # each local client model
    pmodels = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/domainnet/{}'.format(args.mode.lower()))
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                pmodels[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                pmodels[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(pmodels[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))

        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                pmodels[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                pmodels[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers = [optim.SGD(params=pmodels[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):

            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx, model in enumerate(pmodels):
                pmodel, train_loader, optimizer = pmodels[client_idx], train_loaders[client_idx], optimizers[client_idx]

                # train(pmodel, train_loader, optimizer, loss_fun, client_num, device)
                pmodel.train_client(train_loader, steps=1)

        # aggregation
        server_model, pmodels = communication(args, server_model, pmodels, client_weights)

        # Report loss after aggregation
        for client_idx, model in enumerate(pmodels):
            train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)

            print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                 train_acc))
            if args.log:
                logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                               train_loss,
                                                                                               train_acc))

        # Validation
        val_acc_list = [None for j in range(client_num)]
        for client_idx, model in enumerate(pmodels):
            val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
            # val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
            val_acc_list[client_idx] = val_acc
            print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss,
                                                                               val_acc), flush=True)
            if args.log:
                logfile.write(
                    ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                   val_loss,
                                                                                   val_acc))

        # Record best
        if np.mean(val_acc_list) > np.mean(best_acc):
            for client_idx in range(client_num):
                best_acc[client_idx] = val_acc_list[client_idx]
                best_epoch = a_iter
                best_changed = True
                print(' Best site-{:<10s}| Epoch:{} | Val Acc: {:.4f}'.format(datasets[client_idx], best_epoch,
                                                                              best_acc[client_idx]))
                if args.log:
                    logfile.write(
                        ' Best site-{:<10s} | Epoch:{} | Val Acc: {:.4f}\n'.format(datasets[client_idx], best_epoch,
                                                                                   best_acc[client_idx]))

        if best_changed:
            print(' Saving the local and server checkpoint to {}...'.format(SAVE_PATH))
            logfile.write(' Saving the local and server checkpoint to {}...\n'.format(SAVE_PATH))

            torch.save({
                'server_model': server_model.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'a_iter': a_iter
            }, SAVE_PATH)
            best_changed = False

            # test on seen datasets
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(server_model, test_loaders[client_idx], loss_fun, device)
                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                if args.log:
                    logfile.write(
                        ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch,
                                                                                   test_acc))
            # test on unseen datasets
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(server_model, unseen_test_loader, loss_fun, device)
                print(
                    'Unseen site-{:<10s} |Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(unseen_dataset,
                                                                                                 datasite,
                                                                                                 best_epoch,
                                                                                                 test_acc))
                if args.log:
                    logfile.write('Unseen site-{:<10s} |Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(
                        unseen_dataset,
                        datasite,
                        best_epoch,
                        test_acc))

    if log:
        logfile.flush()
        logfile.close()

#  python dapfl_domainnet.py --log --mode dapfl --unseen_client 5 --gpu 4