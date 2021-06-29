import argparse
import torch
import os
from pathlib import Path

from traitlets.traitlets import default

from train import run

#PATH
HOME_PATH = Path("/home/mingi/mingi")
DATA_PATH = Path("/datasets")
PATH = HOME_PATH/"MyLayerTest/CIFAR10/pytorch-cifar"

#Parser
parser = argparse.ArgumentParser(description="I want it to be success",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=0.1, type=float, dest="lr")
parser.add_argument("--batch_size", default=128, type=int, dest="batch_size")
parser.add_argument("--epochs", default=100, type=int, dest="epochs")

parser.add_argument("--data_dir", default=DATA_PATH/"cifar10", type=str, dest="data_dir")
parser.add_argument("--project_dir", default=PATH, type=str, dest="project_dir")
parser.add_argument("--checkpoint_dir", default=PATH/"checkpoint", type=str, dest="project_dir")

parser.add_argument("--device", default='cuda', type=str, dest="device")

parser.add_argument("--mode", default="train", choices=["train", "test", "val"], type=str, dest="mode")
parser.add_argument("--train_continue", action='store_true', dest="train_continue")

parser.add_argument("--task", default=None, type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=[], dest='opts') #입력으로 리스트 받기

parser.add_argument("--ny", default=64, type=int, dest="ny")
parser.add_argument("--nx", default=64, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--th", default=30, type=int, dest="th")
parser.add_argument('--thlist', help='start, end, step', type=str)
parser.add_argument("--thlist_flag",default=False, action='store_true')

parser.add_argument("--norm", default="bnorm", choices=["bnorm", "inorm"], type=str, dest="norm")
#1
parser.add_argument("--network", default="efficient", type=str, dest="network")
parser.add_argument("--advloss", default="vanilla", choices=["vanilla", "wasserstein", "ls", "hinge"], type=str, dest="advloss")
# parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")
parser.add_argument("--foldername", default='mingi', type=str, dest="foldername")

parser.add_argument("--sample_every", default=8, type=int, dest="sample_every")
parser.add_argument("--chpt_every", default=30, type=int, dest="chpt_every")
parser.add_argument("--eval_every", default=10, type=int, dest="eval_every")
parser.add_argument("--save_image_num", default=1, type=int, dest="save_image_num")
parser.add_argument("--print_every", default=2, type=int, dest="print_every")

parser.add_argument("--loop", default=1, type=int, dest="loop")

parser.add_argument("--dname_session", default="", type=str, dest="dname_session")

parser.add_argument("--load_pretrain", action='store_true')
parser.add_argument("--pretrain", default=False, action='store_true')
parser.add_argument("--transfer_mylayer", default=None, type=str)


args = parser.parse_args()

if __name__ == "__main__":
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.thlist_flag:
        thlist = [1,-1] # 0-pretrain 1-alpha -1:all
        netlist = ["vgg19","resnet18","mobilenet_v2", "efficientnetb0", "resnet34", "googlenet"]

        for k in netlist:
            if k == "vgg19":
                continue
            args.transfer_mylayer = None
            args.network = k
            args.mode = "test"
            args.th = 0
            args.train_continue = True
            args.pretrain = True
            run(args)

            args.train_continue = False
            args.pretrain = False
            for j in netlist:
                args.transfer_mylayer = j
                for i in thlist:
                    args.th = i
                    run(args)
    else:
        run(args)
    


    # n = args.loop
    # for k in netlist:
    #     args.network = k
    #     for j in thlist:
    #         args.th = j
    #         if j == 0:
    #             args.pretrain = True
    #         else :
    #             args.pretrain = False
    #         for i in range(n):
    #             args.loop = i
    #             run(args)
