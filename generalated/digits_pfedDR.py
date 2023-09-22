"""
federated learning with different aggregation strategy on benchmark exp.
"""
import sys, os

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)

import torch
from torch import nn, optim
import time
import copy
# from nets.models_fuse_multitask import DigitModel
from nets.models_multitask_dbnfc import DigitModel
import argparse
import numpy as np
import torchvision.transforms as transforms
from utils import data_utils


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


# for pfeddr
def train_multitask(model, train_loader, optimizer, loss_fun, client_num, device, alpha=0.5, beta=2, mode="global"):
    def multitaskLossFn(output_local, output_global, y, alpha):
        """
        Loss function for student network: Loss = alpha * (distillation loss with soft-target) + (1 - alpha) * (cross-entropy loss with true label)
        Return: loss
        """
        if (alpha > 0):
            loss = loss_fun(output_global, y) * (1 - alpha) + loss_fun(output_local, y) * alpha
        else:
            loss = loss_fun(output_global, y)
        return loss

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
        output_global, att = model(x, mode='global')
        output_local, att = model(x, mode='local')

        loss = multitaskLossFn(output_local, output_global, y, alpha=alpha)

        KL_loss_g = torch.nn.functional.kl_div(torch.softmax(output_global, dim=1), torch.softmax(output_local, dim=1),reduction="batchmean")
        # KL_loss_l = torch.nn.functional.kl_div(torch.softmax(output_local, dim=1), torch.softmax(output_global, dim=1))
        loss -= beta * KL_loss_g
        # loss+= KL_loss_l

        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output_global.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()

    return loss_all / len(train_iter), correct / num_data


def train(model, train_loader, optimizer, loss_fun, client_num, device, mode="global"):
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
        output, att = model(x, mode=mode)

        loss = loss_fun(output, y)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()

        pred = output.data.max(1)[1]
        correct += pred.eq(y.view(-1)).sum().item()
    return loss_all / len(train_iter), correct / num_data


def train_fedprox(args, model, train_loader, optimizer, loss_fun, client_num, device, mode="global"):
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
        output, _ = model(x, mode=mode)

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


def test(model, test_loader, loss_fun, device, mode='global'):
    model.eval()
    test_loss = 0
    correct = 0
    targets = []

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()
        targets.append(target.detach().cpu().numpy())

        output, x = model(data, mode=mode)

        test_loss += loss_fun(output, target).item()
        pred = output.data.max(1)[1]

        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


# for pfeddr test
def test_ensemble(model, test_loader, loss_fun, device):
    model.eval()
    test_loss = 0
    correct = 0

    for data, target in test_loader:
        data = data.to(device).float()
        target = target.to(device).long()

        output, _ = model(data, mode='local')
        output2, __ = model(data, mode='global')
        output_bagging=(output+output2)/2

        test_loss += loss_fun(output_bagging, target).item()
        # pred2 = output_bagging.data.max(1)[1]
        pred=output_bagging.argmax(dim=1)
        correct += pred.eq(target.view(-1)).sum().item()

    return test_loss / len(test_loader), correct / len(test_loader.dataset)


def aggregation(args, server_model, models, client_weights):
    with torch.no_grad():
        # # 全局合并所有的key
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

        return server_model


def dispense(args, server_model, models, client_weights):
    with torch.no_grad():
        if args.mode.lower() == 'fedavg':
            for client_idx in range(len(models)):
                models[client_idx].load_state_dict(server_model.state_dict())

        elif args.mode.lower() == 'pfeddr':
            for client_idx in range(len(models)):
                for key in server_model.state_dict().keys():
                    if 'local' not in key:
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        elif args.mode.lower() == 'fedbn':
            for client_idx in range(len(models)):
                for key in server_model.state_dict().keys():
                    if 'bn' not in key:
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for client_idx in range(len(models)):
                models[client_idx].load_state_dict(server_model.state_dict())
    return models


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


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--log', action='store_true', help='whether to make a log')
    parser.add_argument('--test', action='store_true', help='test the pretrained model')
    parser.add_argument('--percent', type=float, default=0.1, help='percentage of dataset to train')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--iters', type=int, default=120, help='iterations for communication')
    parser.add_argument('--wk_iters', type=int, default=1,
                        help='optimization iters in local worker between communication')
    parser.add_argument('--mu', type=float, default=1e-2, help='The hyper parameter for fedprox')

    parser.add_argument('--mtl_alpha', type=float, default=0.5, help='The hyper parameter for multiloss')
    parser.add_argument('--beta', type=float, default=0, help='The hyper parameter for multiloss')

    parser.add_argument('--save_path', type=str, default='../checkpoint/digits_pfeddr',
                        help='path to save the checkpoint')
    parser.add_argument('--resume', action='store_true', help='resume training from the save path checkpoint')

    parser.add_argument('--mode', type=str, default='pfeddr', help='fedavg | fedprox | fedbn | pfeddr')
    parser.add_argument('--unseen_client', type=int, default=4, help='unseen_client')
    parser.add_argument('--gpu', type=str, default="6", help='gpu')

    args = parser.parse_args()
    print(args)

    gpu = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    device = torch.device("cuda:0")
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print('Device:', gpu)
    print('Seed:', seed)

    exp_folder = args.mode
    if args.mode == 'pfeddr':
        # exp_folder += '_m_alpha' + str(args.mtl_alpha)
        exp_folder += '_beta'+ str(args.beta)

    args.save_path = os.path.join(args.save_path, exp_folder)

    log = args.log
    if log:
        log_path = os.path.join('../logs/digits_pfeddr/', exp_folder)
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
    server_model = DigitModel().to(device)
    # getModelSize(server_model)
    print("server_model.state_dict().keys()", server_model.state_dict().keys())
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
        # SAVE_PATH = os.path.join(args.save_path, '{}_seed{}_unseen{}'.format(args.mode, seed, args.unseen_client))
        # checkpoint = torch.load(f'{args.save_path}/{args.mode.lower()}')
        #/home/liuyuan/liuyuan_home/project/FedBN/checkpoint/digits_pfeddr/pfeddr_+ordered_momentum0.5/pfeddr_seed0_unseen0
        checkpoint = torch.load(f'../checkpoint/digits_pfeddr/pfeddr_+ordered_momentum0.5/pfeddr_seed0_unseen{args.unseen_client}')
        server_model.load_state_dict(checkpoint['server_model'])
        if args.mode.lower() == 'pfeddr':
            for client_idx in range(client_num):
                pmodels[client_idx].load_state_dict(checkpoint['model_{}'.format(client_idx)])
            for test_idx, test_loader in enumerate(test_loaders):
                _, test_acc = test(pmodels[test_idx], test_loader, loss_fun, device,mode="local")
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
                if args.log:
                    logfile.write('Seen| {:<11s}| Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_acc))

                _, test_acc = test(pmodels[test_idx], unseen_test_loader, loss_fun, device, mode="global")
                print(' {:<11s}| Test  Acc: {:.4f}'.format(datasets[test_idx], test_acc))
                if args.log:
                    logfile.write('Unseen| {:<11s}| Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_acc))


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
        optimizers = [optim.SGD(params=pmodels[idx].parameters(), lr=args.lr,
                                momentum=0.5) for idx in range(client_num)]

        for wi in range(args.wk_iters):

            print("============ Train epoch {} ============".format(wi + a_iter * args.wk_iters))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(wi + a_iter * args.wk_iters))

            for client_idx in range(client_num):
                pmodel, train_loader, optimizer = pmodels[client_idx], train_loaders[client_idx], optimizers[client_idx]
                if args.mode.lower() == 'fedprox':
                    if a_iter > 0:
                        train_fedprox(args, pmodel, train_loader, optimizer, loss_fun, client_num, device)
                    else:
                        train(pmodel, train_loader, optimizer, loss_fun, client_num, device)

                elif args.mode.lower() == 'pfeddr':
                    train_multitask(pmodel, train_loader, optimizer, loss_fun, client_num, device, alpha=args.mtl_alpha, beta=args.beta)

                else:
                    train(pmodel, train_loader, optimizer, loss_fun, client_num, device)

        # aggregation
        server_model = aggregation(args, server_model, pmodels, client_weights)

        # dispense by different method
        pmodels = dispense(args, server_model, pmodels, client_weights)

        # report after aggregation
        for client_idx in range(client_num):
            pmodel, train_loader, optimizer = pmodels[client_idx], train_loaders[client_idx], optimizers[client_idx]

            if args.mode.lower() == 'pfeddr':
                train_loss, train_acc = test_ensemble(pmodel, train_loader, loss_fun, device)
            else:
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
        if (a_iter + 1) % 10 == 0 or a_iter > 115:
        # if (a_iter + 1) % 10 == 0 or a_iter > 100:
        # if a_iter > -1:

            # test on seen datasets
            for test_idx, test_loader in enumerate(test_loaders):
                if args.mode.lower() == 'pfeddr':
                    test_loss, test_acc = test_ensemble(pmodels[test_idx], test_loader, loss_fun, device)
                    # test_loss, test_acc = test(pmodels[test_idx], test_loader, loss_fun, device, mode='local')
                else:
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
                if args.mode.lower() == 'pfeddr':
                    test_loss, test_acc = test_ensemble(pmodels[test_idx], unseen_test_loader, loss_fun, device)
                else:
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

    # # Save checkpoint
    # print(' Saving checkpoints to {}...'.format(SAVE_PATH))
    # torch.save({
    #     'model_0': pmodels[0].state_dict(),
    #     'model_1': pmodels[1].state_dict(),
    #     'model_2': pmodels[2].state_dict(),
    #     'model_3': pmodels[3].state_dict(),
    #     # 'model_4': Pmodels[4].state_dict(),
    #     'server_model': server_model.state_dict(),
    # }, SAVE_PATH)

    if log:
        logfile.flush()
        logfile.close()
