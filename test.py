#%%
import argparse
import wandb

import os
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from torchvision import transforms, datasets

from model import *
from dataset import * #모두다
from util import * #save, load
from mkdir import mkdirlist

from tqdm import tqdm

def test(args):
    #하이퍼파라미터
    lr = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    norm = args.norm

    #디렉토리
    data_dir = args.data_dir
    project_dir = args.project_dir

    project_dir = project_dir/args.network/args.advloss/args.foldername
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

    th = args.th
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

    #학습데이터
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    validation_dataset = datasets.ImageFolder(root="./test", transform=transform)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size,
                                            shuffle=False)


    classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # #필요해서 만듬
    num_data_train = len(validation_dataset)
    print(num_data_train)


    # #batch 사이즈 
    num_batch_train = np.ceil(num_data_train/batch_size)

    ##
    #2
    ##네트워크 생성 -> GAN 용으로 바꿈. 나중엔 generator, discriminator도 여러개가 될 수 있으므로 그에 맞게 만듬.
    if network == "efficient":
        net = EfficientNetB0(out_sz=10, th=th).to(device)
        # init_weights(netG, init_type='normal', init_gain=0.02)


    #loss
    fn_loss = nn.CrossEntropyLoss().to(device)
    #fn_loss = nn.BCEWithLogitsLoss().to(device)

    #Adam사용
    optim = torch.optim.Adam(net.parameters(), lr=lr)

    #다시 넘파이로, 노말라이즈 돌리기, 0.5보다 크면 1리턴 함수
    fn_pred = lambda output: torch.softmax(output, dim=1)
    fn_acc = lambda pred, label: ((pred.max(dim=1)[1] == label).type(torch.float)).mean()

    fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0,2,3,1)
    fn_denorm = lambda x, mean, std: (x*std) + mean

    # wandb.init(project='Efficient_{}'.format(args.foldername), config=args)
    wandb.init(project='Efficient_Color', config=args)

    
    ## 네트워크 학습시키기
    st_epoch = 0
    #저장해둔 네트워크 불러오기.
    
    net, optim, st_epoch = load(ckpt_dir, net, optim)

    for epoch in range(st_epoch+1, st_epoch+2):
        print('EPOCH %04d/%04d ' % (epoch, epochs))
        net.eval()


        loss_arr_val = []
        acc_arr_val = []

        for batch, data in enumerate(tqdm(validation_loader), 1):
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            print(inputs.shape)
            print(labels)

            output, middle = net(inputs)
            pred = fn_pred(output)

            loss_val = fn_loss(output, labels)
            acc_val = fn_acc(pred, labels)

            loss_arr_val += [loss_val.item()]
            acc_arr_val += [acc_val.item()]
            
            inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5)).squeeze()
            middle = fn_tonumpy(fn_denorm(middle, mean=0.5, std=0.5)).squeeze()
            print(inputs.shape)
            print(middle.shape)
        
            middle = np.clip(middle, a_min=0, a_max=1)
            inputs = np.clip(inputs, a_min=0, a_max=1)
            # id = num_batch_train * (epoch - 1) + batch
            # i = int(batch//(num_batch_train//SAVE_IMAGE_NUM))
            plt.imsave(os.path.join("./test", 'inputimage{}.png'.format(batch)), inputs)
            plt.imsave(os.path.join("./test", 'middleimage{}.png'.format(batch)), middle)

        
        # wandb.log({"loss_val_{}".format(foldername) : np.mean(loss_arr_val), "acc_val_{}".format(foldername) : np.mean(acc_arr_val), "epoch" : epoch})

        print("")
        # print('TRAIN: LOSS: %.4f | ACC %.4f' %
        #         (np.mean(loss_arr), np.mean(acc_arr)))
        print('VAL: LOSS: %.4f | ACC %.4f' % (np.mean(loss_arr_val), np.mean(acc_arr_val)))


        if epoch % SAVE_MODEL_EVERY == 0:
            save(ckpt_dir, net, optim, epoch)


    wandb.finish()

