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

import pickle as pkl
from utils.data_utils import DomainNetDataset
from nets.models_multitask_dbn import AlexNet
import argparse
import time
import copy
import torchvision.transforms as transforms
import random
import numpy as np


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


def train(model, data_loader, optimizer, loss_fun, device, mode="global"):
    model.train()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        optimizer.zero_grad()

        data = data.to(device)
        target = target.to(device)
        output = model(data, mode=mode)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

        loss.backward()
        optimizer.step()

    return loss_all / len(data_loader), correct / total


def test(model, data_loader, loss_fun, device, mode='global'):
    model.eval()
    loss_all = 0
    total = 0
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data, mode=mode)
        loss = loss_fun(output, target)
        loss_all += loss.item()
        total += target.size(0)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.view(-1)).sum().item()

    return loss_all / len(data_loader), correct / total


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

    parser.add_argument('--mode', type=str, default='single', help='single')

    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--gpu', type=str, default="0", help='gpu')
    parser.add_argument('--unseen_client', type=int, default=5, help='unseen_client')
    #  python dapfl_domainnet.py --log --mode dapfl --unseen_client 3 --gpu 2

    args = parser.parse_args()

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    print('Device:', gpu)
    print('Seed:', seed)

    exp_folder = args.mode
    args.save_path = os.path.join(args.save_path, exp_folder)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    SAVE_PATH = os.path.join(args.save_path, '{}_seed{}_seen{}'.format(args.mode, seed, args.unseen_client))

    log = args.log
    if log:
        log_path = os.path.join('../logs/domainnet_dapfl', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, f'{args.mode}-seed{seed}-seen{args.unseen_client}.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('seed: {}\n'.format(seed))
        logfile.write('args: {}\n'.format(args))
        print('args: {}\n'.format(args))

    # prepare the data
    train_loaders, val_loaders, test_loaders = prepare_data(args)

    # name of each datasets
    datasets = ['Clipart', 'Infograph', 'Painting', 'Quickdraw', 'Real', 'Sketch']
    args.data = datasets[args.unseen_client]

    train_loader = train_loaders[args.unseen_client]
    val_loader = val_loaders[args.unseen_client]

    # setup model
    model = AlexNet().to(device)
    loss_fun = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)

    best_acc = 0
    best_epoch = 0
    start_epoch = 0
    N_EPOCHS = args.iters

    for epoch in range(start_epoch, start_epoch + N_EPOCHS):
        train_loss, train_acc = train(model, train_loader, optimizer, loss_fun, device)
        print('Epoch: [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(epoch, N_EPOCHS, train_loss, train_acc))
        if log:
            logfile.write(
                'Epoch: [{}/{}] | Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(epoch, N_EPOCHS, train_loss,
                                                                                   train_acc))

        val_loss, val_acc = test(model, val_loader, loss_fun, device)
        print('Val site-{} | Val Loss: {:.4f} | Val Acc: {:.4f}'.format(args.data, val_loss, val_acc))
        if log:
            logfile.write('Val site-{} | Val Loss: {:.4f} | Val Acc: {:.4f}\n'.format(args.data, val_loss, val_acc))
            logfile.flush()

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            print(' Saving the best checkpoint to {}...'.format(SAVE_PATH))
            torch.save({
                'model': model.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'epoch': epoch
            }, SAVE_PATH)
            print('Best site-{} | Epoch:{} | Test Acc: {:.4f}'.format(args.data, best_epoch, best_acc))
            if log:
                logfile.write('Best site-{} | Epoch:{} | Test Acc: {:.4f}\n'.format(args.data, best_epoch, best_acc))

            for index in range(len(datasets)):
                _, test_acc = test(model, test_loaders[index], loss_fun, device)
                print('Test site-{} | Epoch:{} | Test Acc: {:.4f}'.format(datasets[index], best_epoch, test_acc))
                if log:
                    logfile.write(
                        'Test site-{} | Epoch:{} | Test Acc: {:.4f}\n'.format(datasets[index], best_epoch, test_acc))

    if log:
        logfile.flush()
        logfile.close()
