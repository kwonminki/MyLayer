'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

import os
import argparse

from mkdir import mkdirlist
from addlayer import AddLayerNet, normalization_th
from utils import progress_bar, UnNormalize
import wandb

from efficientnet_pytorch import EfficientNet

best_acc = 0.0
def run(args):
    project_dir = args.project_dir/args.network/str(args.th)
    result_dir = project_dir/"png"
    mkdirlist([project_dir,result_dir])

    global best_acc # best test accuracy
    netacc = {'efficientnet-b0' : 74.078, 'efficientnet-b3':80.762, 'efficientnet-b6':84.02, 'resnet50':76.132, 'vgg19':71.59, 'resnet34':73.314, 'resnet18':69.760}
    best_acc = netacc[args.network] if args.network in netacc.keys() else 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    unnorm = UnNormalize(mean, std)
    # image_size = EfficientNet.get_image_size(args.network)
    transform_train = transforms.Compose(
        [transforms.Resize(256), #image_size
        transforms.CenterCrop(224), #224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    transform_test = transforms.Compose(
        [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    # th = normalization_th(args.th, mean, std)

    traindir = os.path.join(args.data_dir, 'train')
    valdir = os.path.join(args.data_dir, 'val')
    # testdir = os.path.join(args.data_dir, 'test')

    train_dataset = datasets.ImageFolder(traindir, transform_train)
    validation_dataset = datasets.ImageFolder(valdir, transform_train)
    test_dataset = datasets.ImageFolder(valdir, transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    # #필요해서 만듬
    num_data_train = len(train_dataset)

    # #batch 사이즈 
    num_batch_train = np.ceil(num_data_train/args.batch_size)

    # Model
    print('==> Building model..')

    net = AddLayerNet(args.network, args.th, args.pretrain).to(args.device)
    print(f'net : {args.network}, th : {args.th}, pretrain : {args.pretrain}')

    if args.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    if args.mode != 'test':
        optimizer = optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    criterion = nn.CrossEntropyLoss()

    Usewandb = False
    sudoTest = False
            
    if sudoTest:
        mynet = net.state_dict()
        for name in mynet.keys():
            if 'running' in name or 'num_batches_tracked'in name:
                mynet[name].mul_(0)

        
        exit()

    if args.train_continue:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.project_dir/args.network/str(args.th)/"checkpoint"), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(args.project_dir/args.network/str(args.th)/"checkpoint"/'ckpt.pth')
        if args.th == 0:
            print("th is 0")
            dellist = []
            for i in checkpoint['net'].keys():
                if 'mine' in i:
                    dellist.append(i)
            for i in dellist:
                del checkpoint['net'][i]
        # mynet = net.state_dict()
        # for name, param in checkpoint['net'].items():
        #     mynet['module.'+name].copy_(param.data)
        net.load_state_dict(checkpoint['net'])
        if not args.train_continue:
            optimizer.load_state_dict(checkpoint['optim'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    if args.load_pretrain and args.th != 0:
        assert os.path.isdir(args.project_dir/args.network/"0"), 'Error: no pretrained directory found!'
        net_backbone = torch.load(args.project_dir/args.network/"0"/"checkpoint"/'ckpt.pth')
        mynet = net.state_dict()
        for name, param in net_backbone['net'].items():
            if "backbone" in name:
                mynet[name].copy_(param.data)
        net.load_state_dict(mynet)
        for param in net.backbone.parameters():
            param.requires_grad = False
        for param in net.mine.parameters():
            param.requires_grad = True
            nn.init.normal_(param, 0, 0.5)

    if args.transfer_mylayer:
        assert os.path.isdir(args.project_dir/args.transfer_mylayer/args.transfer_th), 'Error: no pretrained directory found! {}'.format(args.project_dir/args.transfer_mylayer/args.transfer_th)
        net_transfer = torch.load(args.project_dir/args.transfer_mylayer/args.transfer_th/"checkpoint"/'ckpt.pth')
        mynet = net.state_dict()
        for name, param in net_transfer['net'].items():
            if "mine" in name:
                mynet[name].copy_(param.data) 

        net.load_state_dict(mynet)
        for param in net.parameters():
            if len(param) == 0:
                continue
            param.requires_grad = False
        
        args.sample_every = 1000
        best_acc = 101


    if args.pretrain != True:
        for param in net.module.backbone.parameters():
            param.requires_grad = False
        for param in net.module.mine.parameters():
            param.requires_grad = True
            if args.mode == 'train':
                nn.init.normal_(param, 0, 0.1)
        
        # mynet = net.state_dict()
        # for name in mynet.keys():
        #     if 'running' in name or 'num_batches_tracked'in name:
        #         mynet[name].mul_(0)

    if os.path.isdir(project_dir/"checkpoint"):
        tmp_check = torch.load(project_dir/"checkpoint"/'ckpt.pth')
        best_acc = tmp_check['acc']

    if Usewandb:
        if args.transfer_mylayer:
            wandb.init(project='CIFAR10_{}_transfer_2'.format(args.network), config=args, name="{}_th{}".format(args.transfer_mylayer, args.th))        
        elif args.load_pretrain :
            # wandb.init(project='MyLayerTest_CIFAR10_{}_alpha'.format(args.network), config=args, name="{}_th{}".format(args.network, args.th))
            wandb.init(project='CIFAR10_{}_transfer_2'.format(args.network), config=args, name="{}_th{}".format(args.network, args.th))
        else:
            # wandb.init(project='MyLayerTest_CIFAR10_{}'.format(args.network), config=args, name="{}_th{}".format(args.network, args.th))
            wandb.init(project='CIFAR10_{}_transfer_2'.format(args.network), config=args, name="{}_th{}".format(args.network, args.th))


    # Training
    def train_step(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        batch_num = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            outputs, middle = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | BestAcc: %.3f'
                        % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, best_acc))
            if (epoch+1) % args.sample_every == 0:
                if (batch_idx+1) % args.eval_every == 0:
                    print()
                    if test_step(epoch, validation_loader, mode="val", stopflag=True):
                        print(f'epoch is {epoch}, batch amount is {batch_idx}')
                    print()
                if (batch_idx+1) % args.print_every == 0:
                    
                    fn_tonumpy = lambda x: x.detach().cpu().numpy().transpose(0,2,3,1)
                    # for i in range(3):
                    #     inputs[:][i][:][:] = (inputs[:][i][:][:]*std[i] + mean[i])
                    #     middle[:][i][:][:] = (middle[:][i][:][:]*std[i] + mean[i])
                    inputs, middle = unnorm(inputs, middle)
                    inputs = fn_tonumpy((inputs)).squeeze()
                    middle = fn_tonumpy((middle)).squeeze()
                    middle = np.clip(middle, a_min=0, a_max=1)
                    inputs = np.clip(inputs, a_min=0, a_max=1)

                    for i in range(args.save_image_num):
                        fig = plt.figure()
                        plt.axis("off")
                        tmp = fig.add_subplot(1, 3, 1)
                        tmp.imshow(inputs[i])
                        tmp.set_xticks([]) ,tmp.set_yticks([])
                        tmp.set_xlabel("input")
                        tmp = fig.add_subplot(1, 3, 2)
                        tmp.imshow(middle[i])
                        tmp.set_xticks([]) ,tmp.set_yticks([])
                        tmp.set_xlabel("middle")

                        subimage = middle[i] - inputs[i]
                        # subimage = subimage * 10
                        subimage[subimage == 0] = 225
                        # subimage[subimage != 0] = 0

                        # subimage = [225 if x==0 else 0 for x in subimage]

                        tmp = fig.add_subplot(1, 3, 3)
                        tmp.imshow(subimage.astype(np.uint8))
                        tmp.set_xticks([]) ,tmp.set_yticks([])
                        tmp.set_xlabel("subimage")

                        id = int(num_batch_train * (epoch) + batch_idx)
                        
                        plt.savefig(os.path.join(result_dir, '{}_{}{}_result.png'.format(id, args.loop, i)))
                        plt.close()
        
        if Usewandb:            
            wandb.log({"loss_train" : train_loss/args.batch_size, "acc_train" : 100.*correct/total, "epoch" : epoch, "th" : args.th})
        # print(torch.tanh(net.mine.conditionR.alpha))
        # print(torch.tanh(net.mine.conditionG.alpha))
        # print(torch.tanh(net.mine.conditionB.alpha))



    def test_step(epoch, loader, mode, stopflag = False):
        if args.transfer_mylayer and not os.path.isdir(result_dir/"test"):
            os.mkdir(result_dir/"test")
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)
                outputs, middle = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

                if args.transfer_mylayer:
                    fn_tonumpy = lambda x: x.detach().cpu().numpy().transpose(0,2,3,1)
                    # for i in range(3):
                    #     inputs[:][i][:][:] = (inputs[:][i][:][:]*std[i] + mean[i])
                    #     middle[:][i][:][:] = (middle[:][i][:][:]*std[i] + mean[i])
                    inputs, middle = unnorm(inputs, middle)
                    inputs = fn_tonumpy((inputs)).squeeze()
                    middle = fn_tonumpy((middle)).squeeze()
                    middle = np.clip(middle, a_min=0, a_max=1)
                    inputs = np.clip(inputs, a_min=0, a_max=1)

                    for i in range(args.save_image_num):
                        fig = plt.figure()
                        plt.axis("off")
                        tmp = fig.add_subplot(1, 3, 1)
                        tmp.imshow(inputs[i])
                        tmp.set_xticks([]) ,tmp.set_yticks([])
                        tmp.set_xlabel("input")
                        tmp = fig.add_subplot(1, 3, 2)
                        tmp.imshow(middle[i])
                        tmp.set_xticks([]) ,tmp.set_yticks([])
                        tmp.set_xlabel("middle")

                        subimage = middle[i] - inputs[i]
                        # subimage = subimage * 10
                        subimage[subimage == 0] = 225
                        # subimage[subimage != 0] = 0

                        # subimage = [225 if x==0 else 0 for x in subimage]

                        tmp = fig.add_subplot(1, 3, 3)
                        tmp.imshow(subimage.astype(np.uint8))
                        tmp.set_xticks([]) ,tmp.set_yticks([])
                        tmp.set_xlabel("subimage")

                        id = int(num_batch_train * (epoch) + batch_idx)
                        
                        plt.savefig(os.path.join(result_dir, 'test/{}_{}{}_result.png'.format(id, args.loop, i)))
                        plt.close()
                # if stopflag:
                #     if batch_idx == 200:
                #         break

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc and mode == "val":
            print(f'Saving..to..{project_dir/"checkpoint"/"ckpt.pth"}')
            state = {
                'net': net.state_dict(),
                'optim' : optimizer.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir(project_dir/"checkpoint"):
                os.mkdir(project_dir/"checkpoint")
            torch.save(state, project_dir/"checkpoint"/"ckpt.pth")
            best_acc = acc
            return True
        
        if Usewandb:
            wandb.log({"loss_{}".format(mode) : test_loss/args.batch_size,  "acc_{}".format(mode) : 100.*correct/total, "epoch" : epoch, "th" : args.th, "bestacc" : best_acc})

        return False

    def batch_step(batch_num=0, amount=100):
        net.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader, start=batch_num):
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            _, _ = net(inputs)

            progress_bar(batch_idx-batch_num, amount, 'Batch_update...')
            if batch_idx == amount+batch_num:
                return amount+batch_num
                


    if args.mode == "train":
        if args.train_continue:
            scheduler.step()
        for epoch in range(start_epoch, start_epoch+args.epochs):
            train_step(epoch)
            test_step(epoch, validation_loader, mode="val")
            scheduler.step()
    if args.transfer_mylayer:
        batch_step(amount=args.batch_amount) #중요
    test_step(start_epoch, validation_loader, mode="test")

    if Usewandb:
        wandb.finish()
