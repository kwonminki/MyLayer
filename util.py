#전역에 사용되는 것들
import os
import numpy as np

import torch
import torch.nn as nn

from scipy.stats import poisson
from skimage.transform import rescale, resize

from mkdir import mkdirlist


def normalization_th(th, mean, std):

    if isinstance(mean, tuple):
        mean = list(mean)
    if isinstance(std, tuple):
        std = list(std)
    if not isinstance(mean, list):
        mean = [mean]
    if not isinstance(std, list):
        std = [std]
    if len(mean) == 1:
        mean = [mean[0], mean[0], mean[0]]
    if len(std) == 1:
        std = [std[0], std[0], std[0]]

    th = th/255.0
    th = [th, th, th]

    th = np.array(th, np.float)
    mean = np.array(mean, np.float)
    std = np.array(std, np.float)

    th = (th - mean) / std
    th = th.sum()/3.0

    return th


## 네트워크 grad 설정하기
def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad



def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>



def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    save_dic = {}
    
    save_dic.update({'net' : net.state_dict(), 'optim' : optim.state_dict()})

    torch.save(save_dic, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))



## 네트워크 불러오기
def load(ckpt_dir, net, optim, index = -1):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[index]))
    
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])

    epoch = int(ckpt_lst[index].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch


def create_stage(n_in, n_out, num_layers, layer_type, 
                 kernel_size=3, stride=1, r=24, p=0):
    """Creates a Sequential consisting of [num_layers] layer_type"""
    layers = [layer_type(n_in, n_out, kernel_size=kernel_size,
                        stride=stride, r=r, p=p)]
    layers += [layer_type(n_out, n_out, kernel_size=kernel_size,
                            r=r, p=p) for _ in range(num_layers-1)]
    layers = nn.Sequential(*layers)
    return layers
  
  
def scale_width(w, w_factor):
    """Scales width given a scale factor"""
    w *= w_factor
    new_w = (int(w+4) // 8) * 8 #8로 나눌수 있는 숫자로 반올림함. -> 재밌는 테크닉인듯 함. 
    new_w = max(8, new_w) # 8보다 작으면 8로 바꿔줌.
    if new_w < 0.9*w: # 위 과정을 거쳤는데 10퍼센트보다 더 낮게 작아졌으면 그냥 8 더 올려줌.
        new_w += 8
    return int(new_w)