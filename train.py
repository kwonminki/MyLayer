#%%
import argparse
import wandb

import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from torchvision import transforms, datasets
import torchvision.models as models
from efficientnet_pytorch import EfficientNet

from ResNet import ResNet
from model import EfficientNetB0, AddLayerNet
from dataset import Dataset
from util import save, load, normalization_th
from mkdir import mkdirlist
from data_loader import get_train_valid_loader, get_test_loader
from LR_Scheduler import LR_Scheduler


from tqdm import tqdm

def train(args):
    #하이퍼파라미터
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    norm = args.norm

    #디렉토리
    data_dir = args.data_dir
    project_dir = args.project_dir/"CIFAR10"/args.network/args.foldername/str(args.th)
    #모델 및 네트워크에 따라 저장을 다르게 하기 위해 디렉토리를 설정.
    ckpt_dir = project_dir/"ckpts"
    log_dir = project_dir/"logs"
    result_dir = project_dir/"train"

    #결과 저장용 디렉토리 생성
    mkdirlist([ckpt_dir,log_dir,result_dir])

    #args
    task = args.task
    opts = args.opts

    ny = args.ny
    nx = args.nx
    nch = args.nch
    nker = args.nker
    foldername = args.foldername

    train_continue = args.train_continue
    network = args.network

    device = args.device

    PRINT_LOSS_NUM = args.print_every # 한 epoch에 print 몇번 할지
    SAVE_SAMPLE_EVERY = args.sample_every # epoch 몇번마다 image save 할지 sample_every
    SAVE_MODEL_EVERY = args.chpt_every # epoch 몇번마다 model save 할지 chpt_every
    SAVE_IMAGE_NUM = args.save_image_num #image 몇개씩 저장할지


    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    th = normalization_th(args.th, mean, std)

    #학습데이터
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    train_loader, validation_loader, num_batch_train = get_train_valid_loader(data_dir=data_dir,batch_size=batch_size, train_transform=transform, valid_transform=transform)
    
    net = AddLayerNet(network, th, args.pretrain).to(device)

#    for param in net.backbone.parameters(): param.requires_grad = False

    #loss
    fn_loss = nn.CrossEntropyLoss().to(device)
    #fn_loss = nn.BCEWithLogitsLoss().to(device)

    parameters = [{
        'name': 'base',
        'params': net.parameters(),
        'lr': lr
    }]

    #optimizer
    optim = torch.optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=0.0005)

    lr_scheduler = LR_Scheduler(
        optim,
        warmup_epochs = 10, warmup_lr=0, 
        num_epochs=epochs, base_lr=0.03, final_lr=0, 
        iter_per_epoch=len(train_loader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_pred = lambda output: torch.softmax(output, dim=1)
    fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean

    # wandb.init(project='Efficient_{}'.format(args.foldername), config=args)
    wandb.init(project='MyLayerTest_CIFAR10_{}'.format(network), config=args, name="{}_{}_th{}".format(args.foldername, args.network, args.th))

    
    ## 네트워크 학습시키기
    st_epoch = 0
    #저장해둔 네트워크 불러오기.
    if train_continue == 'on':
        net, optim, st_epoch = load(ckpt_dir, net, optim)
    if args.load_pretrain :
        net_backbone, _, _ = load(args.project_dir/"CIFAR10"/args.network/args.foldername/"0", net, optim)
        net.backbone.weight.data = net_backbone.weight.data
        for param in net.backbone.parameters(): param.requires_grad = False


    for epoch in range(st_epoch+1, epochs+1):
        print('EPOCH %04d/%04d ' % (epoch, epochs))
        net.train()

        loss_arr = []
        acc_arr = []
 
        for batch, data in enumerate(tqdm(train_loader), 1):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output, middle = net(inputs)

            pred = fn_pred(output)

            optim.zero_grad()

            loss = fn_loss(output, labels)
            acc = fn_acc(pred, labels)

            loss.backward()
            optim.step()
            lr_scheduler.step()

            loss_arr += [loss.item()]
            acc_arr += [acc.item()]

            # wandb.log({"loss_train_step" : loss.item(), "acc_train_step" : acc.item()})


            #이미지 저장. SAVE_SAMPLE_EVERY에 맞게 띄엄띄엄 저장.
            if epoch % SAVE_SAMPLE_EVERY == 0:
                if batch%(num_batch_train//SAVE_IMAGE_NUM) == 0:
                    inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5)).squeeze()
                    middle = fn_tonumpy(fn_denorm(middle, mean=0.5, std=0.5)).squeeze()
                    middle = np.clip(middle, a_min=0, a_max=1)
                    inputs = np.clip(inputs, a_min=0, a_max=1)

                    i = int(batch//(num_batch_train//SAVE_IMAGE_NUM))

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
                    subimage[subimage == 0] = 225

                    # subimage = [225 if x==0 else 0 for x in subimage]

                    tmp = fig.add_subplot(1, 3, 3)
                    tmp.imshow(subimage.astype(np.uint8))
                    tmp.set_xticks([]) ,tmp.set_yticks([])
                    tmp.set_xlabel("subimage")

                    id = num_batch_train * (epoch - 1) + batch
                    
                    # plt.imsave(os.path.join(result_dir, '{}_{}_inputimage.png'.format(id, i)), inputs[i].squeeze())
                    # plt.imsave(os.path.join(result_dir, '{}_{}_middleimage.png'.format(id, i)), middle[i].squeeze())
                    plt.savefig(os.path.join(result_dir, '{}_{}_result.png'.format(id, i)))


        loss_arr_val = []
        acc_arr_val = []
        
        net.eval()
        with torch.no_grad():
            for batch, data in enumerate(tqdm(validation_loader), 1):
                
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                output, _ = net(inputs)
                pred = fn_pred(output)

                loss_val = fn_loss(output, labels)
                acc_val = fn_acc(pred, labels)

                loss_arr_val += [loss_val.item()]
                acc_arr_val += [acc_val.item()]
        
        wandb.log({"loss_train_{}".format(foldername) : np.mean(loss_arr), "acc_train_{}".format(foldername) : np.mean(acc_arr), "loss_val_{}".format(foldername) : np.mean(loss_arr_val), "acc_val_{}".format(foldername) : np.mean(acc_arr_val), "epoch" : epoch})

        print("")
        print('TRAIN: LOSS: %.4f | ACC %.4f' %
                (np.mean(loss_arr), np.mean(acc_arr)))
        print('VAL: LOSS: %.4f | ACC %.4f' % (np.mean(loss_arr_val), np.mean(acc_arr_val)))


        if epoch % SAVE_MODEL_EVERY == 0:
            save(ckpt_dir, net, optim, epoch)


    test_loader = get_test_loader(data_dir=data_dir,batch_size=batch_size, transform=transform)

    loss_arr_test = []
    acc_arr_test = []
    
    net.eval()
    with torch.no_grad():
        for batch, data in enumerate(tqdm(test_loader), 1):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            output, _ = net(inputs)
            pred = fn_pred(output)

            loss_val = fn_loss(output, labels)
            acc_val = fn_acc(pred, labels)

            loss_arr_test += [loss_val.item()]
            acc_arr_test += [acc_val.item()]
    wandb.log({"loss_test_{}".format(foldername) : np.mean(loss_arr_test), "acc_test_{}".format(foldername) : np.mean(acc_arr_test)})



    wandb.finish()

