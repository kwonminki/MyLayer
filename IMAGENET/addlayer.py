from torch import nn
import math
import torchvision.models as models
import numpy as np
from efficientnet_pytorch import EfficientNet
import torch

class AddLayerNet(nn.Module):
    def __init__(self, network, th, pretrain=False):
        super().__init__()
        self.th = th
        self.pretrain = pretrain
        if self.pretrain :
            print("pure model")
        else:
            self.mine = ColorConstancySolved(self.th)

        if network == "resnet18":
            self.backbone = ResNet18(pretrian=True)
        elif network == "alexnet":
            self.backbone = AlexNet(pretrian=True)
        elif network == "googlenet":
            self.backbone = GoogleNet(pretrian=True)
        elif network == "vgg16":
            self.backbone = Vgg16(pretrian=True)
        elif network == "vgg19":
            self.backbone = Vgg16(pretrian=True)
        elif network == "resnet34":
            self.backbone = ResNet34(pretrian=True)
        elif network == "resnet50":
            self.backbone = ResNet50(pretrian=True)
        elif network == "inception_v3":
            self.backbone = Inception_v3(pretrian=True)
        elif network == "mobilenet_v2":
            self.backbone = MobileNet_v2(pretrian=True)
        elif network == "efficientnet-b0":
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif network == "efficientnet-b1":
            self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
        elif network == "efficientnet-b2":
            self.backbone = EfficientNet.from_pretrained('efficientnet-b2')
        elif network == "efficientnet-b3":
            self.backbone = EfficientNet.from_pretrained('efficientnet-b3')
        elif network == "efficientnet-b6":
            self.backbone = EfficientNet.from_pretrained('efficientnet-b6')
        # net = VGG('VGG19')
        # net = ResNet18()
        # net = PreActResNet18()
        # net = GoogLeNet()
        # net = DenseNet121()
        # net = ResNeXt29_2x64d()
        # net = MobileNet()
        # net = MobileNetV2()
        # net = DPN92()
        # net = ShuffleNetG2()
        # net = SENet18()
        # net = ShuffleNetV2(1)
        # net = EfficientNetB0()
        # net = RegNetX_200MF()

    def forward(self, x):
        if self.pretrain:
            middle = x #pure model
        else :
            middle = self.mine(x)

        x = self.backbone(middle)

        return x, middle


class ColorConstancySolved(nn.Module):
    def __init__(self, th):
        super().__init__()

        self.th = th

        self.conv1_R = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2, padding_mode="reflect", bias=True)
        self.conv2_R = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2, padding_mode="reflect", bias=True)
        self.relu1_R = nn.ReLU()#nn.LeakyReLU(0.01)
        # self.conv3_R = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)#
        # self.conv4_R = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        # self.relu2_R = nn.LeakyReLU(0.3)
        self.conv5_R = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu3_R = nn.ReLU()#nn.LeakyReLU(0.01) #0.05 0.5퍼 상승 #0.1도 0.5퍼 근데 조금 더 불안정

        self.conv1_G = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2, padding_mode="reflect", bias=True)
        self.conv2_G = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2, padding_mode="reflect", bias=True)
        self.relu1_G = nn.ReLU()#nn.LeakyReLU(0.01)
        # self.conv3_G = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)#
        # self.conv4_G = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        # self.relu2_G = nn.LeakyReLU(0.3)
        self.conv5_G = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu3_G = nn.ReLU()#nn.LeakyReLU(0.01)

        self.conv1_B = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1, padding=2, padding_mode="reflect", bias=True)
        self.conv2_B = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=2, padding_mode="reflect", bias=True)
        self.relu1_B = nn.ReLU()#nn.LeakyReLU(0.01)
        # self.conv3_B = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)#
        # self.conv4_B = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        # self.relu2_B = nn.LeakyReLU(0.3)
        self.conv5_B = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu3_B = nn.ReLU()#nn.LeakyReLU(0.01)

        self.conditionR = condition()
        self.conditionG = condition()
        self.conditionB = condition()

        # self.condition_all = condition()

    def forward(self, x):

        R, G, B = torch.chunk(x, 3, dim=1)

        gray = (R+G+B)/3

        out_R = self.conv1_R(gray)
        out_R = self.conv2_R(out_R)
        out_R = self.relu1_R(out_R)
        # out_R = self.conv3_R(out_R)
        # out_R = self.conv4_R(out_R)
        # out_R = self.relu2_R(out_R)
        out_R = self.conv5_R(out_R)
        # out_R = self.relu3_R(out_R)


        out_G = self.conv1_G(gray)
        out_G = self.conv2_G(out_G)
        out_G = self.relu1_G(out_G)
        # out_G = self.conv3_G(out_G)
        # out_G = self.conv4_G(out_G)
        # out_G = self.relu2_G(out_G)
        out_G = self.conv5_G(out_G)
        # out_G = self.relu3_G(out_G)


        out_B = self.conv1_B(gray)
        out_B = self.conv2_B(out_B)
        out_B = self.relu1_B(out_B)
        # out_B = self.conv3_R(out_B)
        # out_B = self.conv4_B(out_B)
        # out_B = self.relu2_B(out_B)
        out_B = self.conv5_B(out_B)
        # out_B = self.relu3_B(out_B)


        if self.th != -1:
            condition_R = self.conditionR(R)
            condition_G = self.conditionG(G)
            condition_B = self.conditionB(B)

            # condition_R = self.condition_all(R)
            # condition_G = self.condition_all(G)
            # condition_B = self.condition_all(B)

            condition = condition_R.mul(condition_G).mul(condition_B)

            out_R = out_R.mul(condition)
            out_B = out_B.mul(condition)
            out_G = out_G.mul(condition)

        x = x + torch.cat([out_R,out_G,out_B], dim=1)

        return x


class condition(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_features = in_features

        # initialize alpha
        self.alpha = nn.Parameter(torch.FloatTensor([10.0])) # create a tensor out of alpha            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        # y = -1 * torch.tanh(x-torch.sigmoid(self.alpha))
        # return torch.min(x, y)
        return torch.sigmoid((torch.tanh(self.alpha*0.1)-x)*10)

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
    print(th)
    return th



class AlexNet(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.alexnet(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class GoogleNet(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.googlenet(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class Vgg16(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.vgg16(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class Vgg19(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.vgg19(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class ResNet18(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.resnet34(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.resnet50(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x

class Inception_v3(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.inception_v3(pretrained=pretrian, aux_logits=False)

    def forward(self, x):
        x = self.model(x)

        return x

class MobileNet_v2(nn.Module):
    def __init__(self, pretrian=True):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=pretrian)

    def forward(self, x):
        x = self.model(x)

        return x
