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

from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from generalated.FedRod.data_utils import read_client_data
from nets.models import DigitModel

# split an original model into a base and a head
class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()

        self.base = base
        self.head = head

    def forward(self, x):
        out = self.base(x)
        out = self.head(out)

        return out

class serverROD(nn.Module):
    def __init__(self, args, model, **kwargs):
        super(serverROD, self).__init__()

        head = copy.deepcopy(model.fc3)
        model.fc3 = nn.Identity()
        model = BaseHeadSplit(model, head)

        self.model = copy.deepcopy(model)


class clientROD(nn.Module):
    def __init__(self, args, model, trainloader, **kwargs):
        super(clientROD, self).__init__()
        # args.model= model

        head = copy.deepcopy(model.fc3)
        model.fc3 = nn.Identity()
        model = BaseHeadSplit(model, head)

        self.model = copy.deepcopy(model)

        self.device = args.device
        self.num_classes = args.num_classes
        self.learning_rate = args.lr

        self.head = copy.deepcopy(self.model.head)
        self.opt_head = torch.optim.SGD(self.head.parameters(), lr=self.learning_rate)

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.learning_rate_scheduler_head = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.opt_head,
        #     gamma=args.learning_rate_decay_gamma
        # )
        # self.learning_rate_decay = args.learning_rate_decay
        self.sample_per_class = torch.zeros(self.num_classes)

        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        for x, y in trainloader:
            for yy in y:
                self.sample_per_class[yy.item()] += 1


    def train_client(self, trainloader):

        start_time = time.time()

        # self.model.to(self.device)
        self.model.train()

        # max_local_steps = self.local_steps
        # if self.train_slow:
        #     max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(1):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.optimizer.zero_grad()
                loss_bsm.backward()
                self.optimizer.step()

                out_p = self.head(rep.detach())
                loss = self.loss(out_g.detach() + out_p, y)
                self.opt_head.zero_grad()
                loss.backward()
                self.opt_head.step()

        # self.model.cpu()

        # if self.learning_rate_decay:
        #     self.learning_rate_scheduler.step()
        #     self.learning_rate_scheduler_head.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def test_metrics(self,testloader, model=None):
        # testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc


    def test_global_metrics(self,testloader, model=None):
        # testloader = self.load_test_data()
        if model == None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                output = out_g

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(F.softmax(output).detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc


    def train_metrics(self, trainloader):
        # trainloader = self.load_train_data()

        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                # rep = self.model(x, rep=True)
                rep = self.model.base(x)
                out_g = self.model.head(rep)
                out_p = self.head(rep.detach())
                output = out_g.detach() + out_p
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num


# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


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


def test_g(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    # targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        # targets.append(target.detach().cpu().numpy())

        rep = model.model.base(data)
        out_g = model.model.head(rep)
        out_p = model.head(rep.detach())
        output = out_g.detach() + out_p
        # output = model(data)

        test_loss += loss_fun(output, target).item()
        # acc = (torch.argmax(output, 1) == target).float().mean()

        # print(acc)
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


def test_p(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0
    # targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        # targets.append(target.detach().cpu().numpy())

        rep = model.model.base(data)
        out_g = model.model.head(rep)
        output = out_g
        # output = model(data)

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
    gpu = "2"
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
    parser.add_argument('--mode', type=str, default='fedavg', help='fedavg | FedSR')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')
    parser.add_argument('--save_path', type=str, default='../checkpoint/digits_generalated/',
                        help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')
    parser.add_argument('--unseen_client', type=int, default=4, help='unseen_client')

    args = parser.parse_args()
    args.num_classes = 10  # digits
    args.device = device  # digits
    print(args)

    exp_folder = 'generalated_rod'

    args.save_path = os.path.join(args.save_path, exp_folder)

    log = args.log
    if log:
        log_path = os.path.join('../logs/digits_generalated/', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, f'{args.mode}-seed{seed}-unseen{args.unseen_client}.log'),
                       'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('seed: {}\n'.format(seed))
        logfile.write('args: {}\n'.format(args))

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}_seed{}_unseen{}'.format(args.mode, seed, args.unseen_client))

    # setup model

    server_model=serverROD(args,DigitModel()).to(device)

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

    pmodels = [clientROD(args,DigitModel(),train_loader).to(device) for train_loader in train_loaders]

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
        # optimizers = [optim.SGD(params=pmodels[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):

            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx , train_loader in enumerate(train_loaders):
                # train(pmodel, train_loader, optimizer, loss_fun, client_num, device)
                pmodels[client_idx].train_client(train_loader)

        # aggregation
        server_model, pmodels = communication(args, server_model, pmodels, client_weights)

        # report after aggregation
        for client_idx , train_loader in enumerate(train_loaders):
            train_loss, train_acc =  pmodels[client_idx].train_metrics(train_loader)

            print(
                ' {:<11s}| Pmodel | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                   train_acc))
            if args.log:
                logfile.write(
                    ' {:<11s}| Pmodel | Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                         train_loss,
                                                                                         train_acc))

        # start testing
        # if (a_iter + 1) > 0:
        if (a_iter + 1) % 10 == 0 or a_iter > 90:
            # test on seen datasets
            loss_fun = nn.CrossEntropyLoss()
            for test_idx, test_loader in enumerate(test_loaders):
                # test_acc ,test_loss ,_ = pmodels[test_idx].test_metrics(test_loader)
                test_loss, test_acc = test_p(pmodels[test_idx],test_loader,loss_fun,device)

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
                # test_acc, test_loss, _ = pmodels[test_idx].test_global_metrics(unseen_test_loader)
                test_loss, test_acc = test_g(pmodels[test_idx], unseen_test_loader, loss_fun, device)

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
