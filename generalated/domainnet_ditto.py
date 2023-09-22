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

import collections


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


def aggregate(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
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
    parser.add_argument('--save_path', type=str, default='../checkpoint/domainnet_dapfl',
                        help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

    parser.add_argument('--mode', type=str, default='ditto', help='for log with ditto')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--d_lambda', type=float, default=0.1, help='The hyper parameter for Ditto')

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
    SAVE_PATH = os.path.join(args.save_path, '{}-d_lambda{}_seed{}_unseen{}'.format(args.mode, args.d_lambda, seed, args.unseen_client))

    log = args.log
    if log:
        log_path = os.path.join('../logs/domainnet_dapfl', exp_folder)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path, f'{args.mode}-d_lambda{args.d_lambda}-seed{seed}-unseen{args.unseen_client}.log'), 'a')
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

    client_num = len(datasets)
    client_weights = [1 / client_num for i in range(client_num)]

    loss_fun = nn.CrossEntropyLoss()

    # setup model
    global_model = AlexNet().to(device)
    # print(global_model.state_dict().keys())

    # each local client model
    g_models = [copy.deepcopy(global_model).to(device) for idx in range(client_num)]
    v_models = [copy.deepcopy(global_model).to(device) for idx in range(client_num)]

    best_changed = False

    if args.test:
        print('Loading snapshots...')
        checkpoint = torch.load('../snapshots/domainnet/{}'.format(args.mode.lower()))
        global_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                v_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                v_models[client_idx].load_state_dict(checkpoint['server_model'])
        for test_idx, test_loader in enumerate(test_loaders):
            _, test_acc = test(v_models[test_idx], test_loader, loss_fun, device)
            print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))

        exit(0)

    if args.resume:
        checkpoint = torch.load(SAVE_PATH)
        global_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'fedbn':
            for client_idx in range(client_num):
                v_models[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
        else:
            for client_idx in range(client_num):
                v_models[client_idx].load_state_dict(checkpoint['server_model'])
        best_epoch, best_acc = checkpoint['best_epoch'], checkpoint['best_acc']
        start_iter = int(checkpoint['a_iter']) + 1

        print('Resume training from epoch {}'.format(start_iter))
    else:
        # log the best for each model on all datasets
        best_epoch = 0
        best_acc = [0. for j in range(client_num)]
        start_iter = 0

    client_grads = [collections.defaultdict(list) for _ in range(client_num)]
    # Start training
    for a_iter in range(start_iter, args.iters):
        optimizers_g = [optim.SGD(params=g_models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        optimizers_v = [optim.SGD(params=v_models[idx].parameters(), lr=args.lr) for idx in range(client_num)]
        for wi in range(args.wk_iters):

            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for k in range(client_num):
                g_model, v_model, train_loader, optimizer_g, optimizer_v = g_models[k], v_models[k], \
                                                                           train_loaders[k], \
                                                                           optimizers_g[k], optimizers_v[k]
                # UPDATE GLOBAL(wt, ∇Fk(wt)),得到wtk，局部更新的全局模型参数
                train(g_model, train_loader, optimizer_g, loss_fun, device)

                ################# ditto ########################
                # 保存vk该轮训练前参数
                v_k = copy.deepcopy(v_model.state_dict())
                train(v_model, train_loader, optimizer_v, loss_fun, device)
                nabla_F_k = collections.defaultdict(list)
                # 通过参数差值计算得到该迭代轮次的η∇Fk(vt)
                for key in v_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    else:
                        nabla_F_k[key] = v_k[key] - v_model.state_dict()[key]
                # Ditto
                for key in v_model.state_dict().keys():
                    if 'num_batches_tracked' in key:
                        pass
                    else:
                        # v_model.state_dict()[key] = v_k[key] - F_k[key] - args.lr * args.d_lambda * (
                        #         v_k[key] - global_model.state_dict()[key])
                        # ditto原算法，d_lambda还需要乘以lr
                        v_model.state_dict()[key] = v_k[key] - nabla_F_k[key] - args.lr * args.d_lambda * (
                                v_k[key] - global_model.state_dict()[key])

        # aggregation
        global_model, g_models = aggregate(args, global_model, g_models, client_weights)

        # Report loss after aggregation
        # v_models train
        for client_idx, model in enumerate(v_models):
            train_loss, train_acc = test(model, train_loaders[client_idx], loss_fun, device)

            print(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss,
                                                                                 train_acc))
            if args.log:
                logfile.write(' Site-{:<10s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                               train_loss,
                                                                                               train_acc))
        # Validation
        val_acc_list = [None for j in range(client_num)]
        for client_idx, model in enumerate(v_models):
            val_loss, val_acc = test(model, val_loaders[client_idx], loss_fun, device)
            val_acc_list[client_idx] = val_acc
            print(' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}'.format(datasets[client_idx], val_loss,
                                                                               val_acc), flush=True)
            if args.log:
                logfile.write(
                    ' Site-{:<10s}| Val  Loss: {:.4f} | Val  Acc: {:.4f}\n'.format(datasets[client_idx],
                                                                                   val_loss,
                                                                                   val_acc))

        # Record validation best
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
                'model_0': v_models[0].state_dict(),
                'model_1': v_models[1].state_dict(),
                'model_2': v_models[2].state_dict(),
                'model_3': v_models[3].state_dict(),
                'model_4': v_models[4].state_dict(),
                # 'model_5': pmodels[5].state_dict(),
                'server_model': global_model.state_dict(),
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'a_iter': a_iter
            }, SAVE_PATH)
            best_changed = False

            # Test v_models
            # test on seen datasets
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(v_models[client_idx], test_loaders[client_idx], loss_fun, device)
                print(' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch, test_acc))
                if args.log:
                    logfile.write(
                        ' Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite, best_epoch,
                                                                                   test_acc))
            # test on unseen datasets
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(v_models[client_idx], unseen_test_loader, loss_fun, device)
                print(
                    'Unseen site-{:<10s} |Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(unseen_dataset,
                                                                                                 datasite,
                                                                                                 best_epoch,
                                                                                                 test_acc))
                if args.log:
                    logfile.write(
                        'Unseen site-{:<10s} |Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(
                            unseen_dataset,
                            datasite,
                            best_epoch,
                            test_acc))

            # Test global_model
            # on seen datasets
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(global_model, test_loaders[client_idx], loss_fun, device)
                print(
                    ' Test site-{:<10s}|global_model| Epoch:{} | Test Acc: {:.4f}'.format(datasite, best_epoch,
                                                                                          test_acc))
                if args.log:
                    logfile.write(
                        ' Test site-{:<10s}|global_model| Epoch:{} | Test Acc: {:.4f}\n'.format(datasite,
                                                                                                best_epoch,
                                                                                                test_acc))
            # on unseen datasets
            for client_idx, datasite in enumerate(datasets):
                _, test_acc = test(global_model, unseen_test_loader, loss_fun, device)
                print(
                    'Unseen site-{:<10s} |global_model|Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}'.format(
                        unseen_dataset,
                        datasite,
                        best_epoch,
                        test_acc))
                if args.log:
                    logfile.write(
                        'Unseen site-{:<10s} |global_model|Test site-{:<10s}| Epoch:{} | Test Acc: {:.4f}\n'.format(
                            unseen_dataset,
                            datasite,
                            best_epoch,
                            test_acc))
        if log:
            logfile.flush()
    if log:
        logfile.flush()
        logfile.close()

#  python dapfl_domainnet.py --log --mode dapfl --unseen_client 5 --gpu 4
