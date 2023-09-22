"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
# from torch import nn, optim
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions

import time
import copy
# from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
import hashlib


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        for name in args.__dict__:
            setattr(self, name, getattr(args, name))

        class Flatten(nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1)

        out_dim = 2 * args.z_dim if self.probabilistic else args.z_dim

        net = nn.Sequential(
            nn.Conv2d(3, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 5, 1, 2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 5, 1, 2), nn.BatchNorm2d(128), nn.ReLU(),
            Flatten(),
            nn.Linear(6272, 2048), nn.BatchNorm1d(2048), nn.ReLU(),
            nn.Linear(2048, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
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


# hyper paramters
# can use the HparamsGen to auto-generate hyper parameters when performning random search.
# However, if a hparam is specified when running the program, it will always use that value
class HparamsGen(object):
    def __init__(self, name, default, gen_fn=None):
        self.name = name
        self.default = default
        self.gen_fn = gen_fn

    def __call__(self, hparams_gen_seed=0):
        if hparams_gen_seed == 0 or self.gen_fn is None:
            return self.default
        else:
            s = f"{hparams_gen_seed}_{self.name}"
            seed = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (2 ** 31)
            return self.gen_fn(np.random.RandomState(seed))


def prepare_data(args):
    # Prepare data
    transform_mnist = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_svhn = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_usps = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_synth = transforms.Compose([
        transforms.Resize([28, 28]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_mnistm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # MNIST
    mnist_trainset = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=True,
                                              transform=transform_mnist)
    mnist_testset = data_utils.DigitsDataset(data_path="../data/MNIST", channels=1, percent=args.percent, train=False,
                                             transform=transform_mnist)

    # SVHN
    svhn_trainset = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent, train=True,
                                             transform=transform_svhn)
    svhn_testset = data_utils.DigitsDataset(data_path='../data/SVHN', channels=3, percent=args.percent, train=False,
                                            transform=transform_svhn)

    # USPS
    usps_trainset = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent, train=True,
                                             transform=transform_usps)
    usps_testset = data_utils.DigitsDataset(data_path='../data/USPS', channels=1, percent=args.percent, train=False,
                                            transform=transform_usps)

    # Synth Digits
    synth_trainset = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,
                                              train=True, transform=transform_synth)
    synth_testset = data_utils.DigitsDataset(data_path='../data/SynthDigits/', channels=3, percent=args.percent,
                                             train=False, transform=transform_synth)

    # MNIST-M
    mnistm_trainset = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,
                                               train=True, transform=transform_mnistm)
    mnistm_testset = data_utils.DigitsDataset(data_path='../data/MNIST_M/', channels=3, percent=args.percent,
                                              train=False, transform=transform_mnistm)

    mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=args.batch, shuffle=True)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=args.batch, shuffle=False)
    svhn_train_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=args.batch, shuffle=True)
    svhn_test_loader = torch.utils.data.DataLoader(svhn_testset, batch_size=args.batch, shuffle=False)
    usps_train_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=args.batch, shuffle=True)
    usps_test_loader = torch.utils.data.DataLoader(usps_testset, batch_size=args.batch, shuffle=False)
    synth_train_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=args.batch, shuffle=True)
    synth_test_loader = torch.utils.data.DataLoader(synth_testset, batch_size=args.batch, shuffle=False)
    mnistm_train_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=args.batch, shuffle=True)
    mnistm_test_loader = torch.utils.data.DataLoader(mnistm_testset, batch_size=args.batch, shuffle=False)

    train_loaders = [mnist_train_loader, svhn_train_loader, usps_train_loader, synth_train_loader, mnistm_train_loader]
    test_loaders = [mnist_test_loader, svhn_test_loader, usps_test_loader, synth_test_loader, mnistm_test_loader]

    return train_loaders, test_loaders


def train(model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device):
    model.train()
    num_data = 0
    correct = 0
    loss_all = 0
    train_iter = iter(train_loader)
    for step in range(len(train_iter)):
        optimizer.zero_grad()
        x, y = next(train_iter)
        num_data += y.size(0)
        x = x.to(device).float()
        y = y.to(device).long()
        output = model(x)

        loss = loss_fun(output, y)

        #########################we implement FedProx Here###########################
        # referring to https://github.com/IBM/FedMA/blob/4b586a5a22002dc955d025b890bc632daa3c01c7/main.py#L819
        if step > 0:
            w_diff = torch.tensor(0., device=device)
            for w, w_t in zip(server_model.parameters(), model.parameters()):
                w_diff += torch.pow(torch.norm(w - w_t), 2)
            loss += args.mu / 2. * w_diff
        #############################################################################

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def test(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    # targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        # targets.append(target.detach().cpu().numpy())

        output = model(data)

        test_loss += loss_fun(output, target).item()
        # acc = (torch.argmax(output, 1) == target).float().mean()

        # print(acc)
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


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
    gpu = "7"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', gpu)
    print('Seed:', seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=120, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mode', type=str, default='FedAVG', help='FedAVG | FedSR | FedADG')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/digits_generalated/',
                        help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    parser.add_argument('--unseen_client', type=int, default=4, help='unseen_client')

    # fedsr
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--z_dim', type=int, default=512)
    # parser.add_argument('--L2R_coeff', type=float,
    #                     default=HparamsGen('L2R_coeff', 1e-2, lambda r: 10 ** r.uniform(-5, -3)))
    # parser.add_argument('--CMI_coeff', type=float,
    #                     default=HparamsGen('CMI_coeff', 5e-4, lambda r: 10 ** r.uniform(-5, -3)))
    parser.add_argument('--L2R_coeff', type=float, default=0.1)
    parser.add_argument('--CMI_coeff', type=float, default=0.3)
    parser.add_argument('--D_beta', type=float, default=1)
    # parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--num_samples', type=int, default=20)

    args = parser.parse_args()
    args.num_classes = 10  # digits
    args.device = device  # digits
    print(args)

    exp_folder = 'generalated'

    args.save_path = os.path.join(args.save_path, exp_folder)

    log = args.log
    if log:
        log_path = os.path.join('../logs/digits_generalated/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, f'test{args.mode}-seed{seed}-unseen{args.unseen_client}.log'),
                       'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('seed: {}\n'.format(seed))
        logfile.write('args: {}\n'.format(args))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}_seed{}_unseen{}'.format(args.mode, seed, args.unseen_client))

    # setup model
    # server_model = FedSR(args).to(device)

    if args.mode == 'FedSR':
        server_model = FedSR(args).to(device)
    elif args.mode == 'FedAVG':
        server_model = FedAVG(args).to(device)
        def getModelSize(model):
            param_size = 0
            param_sum = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
                param_sum += param.nelement()
            buffer_size = 0
            buffer_sum = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
                buffer_sum += buffer.nelement()
            all_size = (param_size + buffer_size) / 1024 / 1024
            print('模型总大小为：{:.3f}MB'.format(all_size))
            return (param_size, param_sum, buffer_size, buffer_sum, all_size)
        getModelSize(server_model.model)

    elif args.mode == 'FedADG':
        server_model = FedADG(args).to(device)
    print("server_model.state_dict().keys()", server_model.model.state_dict().keys())
    loss_fun = nn.CrossEntropyLoss()

    # prepare the data
    train_loaders, test_loaders = prepare_data(args)

    # name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    unseen_dataset = datasets.pop(args.unseen_client)
    print("unseen client:", unseen_dataset)
    # set unseen client
    del train_loaders[args.unseen_client]
    unseen_test_loader = test_loaders.pop(args.unseen_client)

    # federated setting
    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]

    pmodels = [copy.deepcopy(server_model).to(device) for idx in range(client_num)]

    if args.test:
        print('Loading snapshots...')
        # checkpoint = torch.load('../snapshots/digits/{}'.format(args.mode.lower()))
        checkpoint = torch.load(f'{args.save_path}/{args.mode.lower()}')
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                pmodels[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(pmodels[test_idx], test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
                if args.log:
                    logfile.write(' {:<11s}| Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_acc))
        else:
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(server_model, test_loader, loss_fun, device)
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
                if args.log:
                    logfile.write(' {:<11s}| Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_acc))

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
        resume_iter = int(checkpoint['a_iter']) + 1
        print('Resume training from epoch {}'.format(resume_iter))
    else:
        resume_iter = 0

    # start training
    for a_iter in range(resume_iter, args.iters):
        optimizers = [optim.SGD(params=pmodels[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):

            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                pmodel, train_loader, optimizer = pmodels[client_idx], train_loaders[client_idx], optimizers[client_idx]

                # train(pmodel, train_loader, optimizer, loss_fun, client_num, device)
                pmodel.train_client(train_loader, steps=1)

        # aggregation
        server_model, pmodels = communication(args, server_model, pmodels, client_weights)

        # report after aggregation
        for client_idx in range(client_num):
            pmodel, train_loader, optimizer = pmodels[client_idx], train_loaders[client_idx], optimizers[client_idx]

            train_loss, train_acc = test(pmodel, train_loader, loss_fun, device)

            print(
                ' {:<11s}| Pmodel | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                   train_acc))
            if args.log:
                logfile.write(
                    ' {:<11s}| Pmodel | Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                         train_loss,
                                                                                         train_acc))

        # start testing
        if (a_iter + 1) > 0:
        # if (a_iter + 1) % 10 == 0 or a_iter > 90:
            # test on seen datasets
            for test_idx, test_loader in enumerate(test_loaders):
                test_loss, test_acc = test(pmodels[test_idx], test_loader, loss_fun, device)

                print(
                    ' {:<11s}| Pmodel |Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss,
                                                                                      test_acc))
                if args.log:
                    logfile.write(
                        ' {:<11s}| Pmodel |Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx],
                                                                                            test_loss,
                                                                                            test_acc))
            # test on unseen datasets
            for test_idx in range(len(pmodels)):

                test_loss, test_acc = test(pmodels[test_idx], unseen_test_loader, loss_fun, device)

                print(
                    ' {:<11s}| Pmodel: {:<11s}  |Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(unseen_dataset,
                                                                                                datasets[test_idx],
                                                                                                test_loss,
                                                                                                test_acc))
                if args.log:
                    logfile.write(
                        ' {:<11s}| Pmodel: {:<11s}  |Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(
                            unseen_dataset, datasets[test_idx],
                            test_loss,
                            test_acc))
                # break

    # Save checkpoint
    print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    torch.save({
        'model_0': pmodels[0].state_dict(),
        'model_1': pmodels[1].state_dict(),
        'model_2': pmodels[2].state_dict(),
        'model_3': pmodels[3].state_dict(),
        # 'model_4': Pmodels[4].state_dict(),
        'server_model': server_model.state_dict(),
    }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()
