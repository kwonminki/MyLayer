from torch import nn
from models import *
import numpy as np

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
            self.backbone = ResNet18()
        # elif network == "alexnet":
        #     self.backbone = AlexNet(pretrian=pretrian)
        elif network == "googlenet":
            self.backbone = GoogLeNet()
        elif network == "vgg19":
            self.backbone = VGG('VGG19')
        elif network == "vgg16":
            self.backbone = VGG('VGG16')
        elif network == "resnet34":
            self.backbone = ResNet34()
        elif network == "resnet50":
            self.backbone = ResNet50()
        # elif network == "inception_v3":
        #     self.backbone = Inception_v3(pretrian=pretrian)
        elif network == "mobilenet_v2":
            self.backbone = MobileNetV2()
        elif network == "efficientnetb0":
            self.backbone = EfficientNetB0()

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
        self.conv1_R = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        # self.bn1 = nn.BatchNorm2d(10)
        # self.relu1 = nn.ReLU()
        # for image net
        # self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.conv3 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect")
        # self.conv4 = nn.Conv2d(in_channels=20, out_channels=3, kernel_size=1, stride=1)
        self.conv2_R = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.relu_R2 = nn.LeakyReLU(0.2)
#        self.conv3_R = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        # self.bn2 = nn.BatchNorm2d(20)
        # self.relu2 = nn.ReLU()
        self.conv4_R = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu_R = nn.LeakyReLU(0.05)
        
        # self.bn3 = nn.BatchNorm2d(3)
        # self.tanh = nn.Tanh()

        self.conv1_G = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.conv2_G = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.relu_G2 = nn.LeakyReLU(0.2)
#        self.conv3_G = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.conv4_G = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu_G = nn.LeakyReLU(0.05)

        self.conv1_B = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.conv2_B = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.relu_B2 = nn.LeakyReLU(0.2)
#        self.conv3_B = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.conv4_B = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu_B = nn.LeakyReLU(0.05)

        self.conditionR = condition()
        self.conditionG = condition()
        self.conditionB = condition()


    #     self.add = 

    # def AddGray(image, addsome):


    def forward(self, x):

        R, G, B = torch.chunk(x, 3, dim=1)
        #Red * 0.299 + Green * 0.587  + Blue * 0.114
        gray = (R+G+B)/3
        # gray = R* 0.299 + G * 0.587+ B * 0.114
        out_R = self.conv1_R(gray)
        # out = self.bn1(out)
        # out = self.relu1(out)
        out_R = self.conv2_R(out_R)
        out_R = self.relu_R2(out_R)

        # out = self.bn2(out)
        # out = self.relu2(out)
        # for image net
        # out = self.conv3(out)
#        out_R = self.conv3_R(out_R)
        out_R = self.conv4_R(out_R)
        # out = self.bn3(out)
        # out = self.tanh(out)
        out_R = self.relu_R(out_R)

        out_G = self.conv1_G(gray)
        out_G = self.conv2_G(out_G)
        out_G = self.relu_G2(out_G)

#        out_G = self.conv3_R(out_G)
        out_G = self.conv4_G(out_G)
        out_G = self.relu_G(out_G)

        out_B = self.conv1_B(gray)
        out_B = self.conv2_B(out_B)
        out_B = self.relu_B2(out_B)

#        out_B = self.conv3_R(out_B)
        out_B = self.conv4_B(out_B)
        out_B = self.relu_B(out_B)

        if self.th != -1:
            condition_R = self.conditionR(R)
            condition_G = self.conditionG(G)
            condition_B = self.conditionB(B)


            # condition_R = torch.as_tensor((R - self.th[0]) < 0, dtype=torch.int32)
            # condition_G = torch.as_tensor((G - self.th[1]) < 0, dtype=torch.int32)
            # condition_B = torch.as_tensor((B - self.th[2]) < 0, dtype=torch.int32)

            condition = condition_R.mul(condition_G).mul(condition_B)

            out_R = out_R.mul(condition)
            out_B = out_B.mul(condition)
            out_G = out_G.mul(condition)

        # x = x + torch.abs(out.mul(condition_R).mul(condition_G).mul(condition_B))
        # x = x + out.mul(condition_R).mul(condition_G).mul(condition_B)

        x = x + torch.cat([out_R,out_G,out_B], dim=1)

        return x


class condition(nn.Module):
    def __init__(self):
        super().__init__()
        # self.in_features = in_features

        # initialize alpha
        self.alpha = nn.Parameter(torch.FloatTensor([0.0])) # create a tensor out of alpha            
        self.alpha.requiresGrad = True # set requiresGrad to true!

    def forward(self, x):
        
        # y = -1 * torch.tanh(x-torch.sigmoid(self.alpha))
        # return torch.min(x, y)
        return torch.sigmoid((torch.tanh(self.alpha)-x)*10)

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