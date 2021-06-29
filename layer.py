import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from util import init_weights


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
        self.conv2_R = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.relu_R2 = nn.LeakyReLU(0.2)
#        self.conv3_R = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        # self.bn2 = nn.BatchNorm2d(20)
        # self.relu2 = nn.ReLU()
        self.conv4_R = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu_R = nn.LeakyReLU(0.05)
        
        # self.bn3 = nn.BatchNorm2d(3)
        # self.tanh = nn.Tanh()

        self.conv1_G = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.conv2_G = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.relu_G2 = nn.LeakyReLU(0.2)
#        self.conv3_G = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.conv4_G = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu_G = nn.LeakyReLU(0.05)

        self.conv1_B = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=True)
        self.conv2_B = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.relu_B2 = nn.LeakyReLU(0.2)
#        self.conv3_B = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False)
        self.conv4_B = nn.Conv2d(in_channels=20, out_channels=1, kernel_size=1, stride=1, bias=False)
        self.relu_B = nn.LeakyReLU(0.05)

        init_weights(self, init_type='normal', init_gain=0.01)

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

        condition_R = torch.as_tensor((R - self.th) < 0, dtype=torch.int32)
        condition_G = torch.as_tensor((G - self.th) < 0, dtype=torch.int32)
        condition_B = torch.as_tensor((B - self.th) < 0, dtype=torch.int32)

        condition = condition_R.mul(condition_G).mul(condition_B)

        out_R = out_R.mul(condition)
        out_B = out_B.mul(condition)
        out_G = out_G.mul(condition)

        # x = x + torch.abs(out.mul(condition_R).mul(condition_G).mul(condition_B))
        # x = x + out.mul(condition_R).mul(condition_G).mul(condition_B)

        x = x + torch.cat([out_R,out_G,out_B], dim=1)

        return x


class ConvBnAct(nn.Module):
    """Layer grouping a convolution, batchnorm, and activation function"""
    def __init__(self, n_in, n_out, kernel_size=3, 
                stride=1, padding=0, groups=1, bias=False,
                bn=True, act=True):
        super().__init__()
        
        self.conv = nn.Conv2d(n_in, n_out, kernel_size=kernel_size,
                            stride=stride, padding=padding,
                            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(n_out) if bn else nn.Identity()
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
        
    
class SEBlock(nn.Module):
    """Squeeze-and-excitation block"""
    def __init__(self, n_in, r=24):
        super().__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(nn.Conv2d(n_in, n_in//r, kernel_size=1),
                                        nn.SiLU(),
                                        nn.Conv2d(n_in//r, n_in, kernel_size=1),
                                        nn.Sigmoid())
    
    def forward(self, x):
        y = self.squeeze(x)
        y = self.excitation(y)
        return x * y
        
        
class DropSample(nn.Module):
    """Drops each sample in x with probability p during training"""
    def __init__(self, p=0):
        super().__init__()

        self.p = p
    
    def forward(self, x):
        if (not self.p) or (not self.training):
            return x
        
        batch_size = len(x)
        random_tensor = torch.cuda.FloatTensor(batch_size, 1, 1, 1).uniform_()
        bit_mask = self.p<random_tensor

        x = x.div(1-self.p)
        x = x * bit_mask
        return x
    
    
class MBConvN(nn.Module):
    """MBConv with an expansion factor of N, plus squeeze-and-excitation"""
    def __init__(self, n_in, n_out, expansion_factor,
                kernel_size=3, stride=1, r=24, p=0):
        super().__init__()

        padding = (kernel_size-1)//2
        expanded = expansion_factor*n_in
        self.skip_connection = (n_in == n_out) and (stride == 1)

        self.expand_pw = nn.Identity() if (expansion_factor == 1) else ConvBnAct(n_in, expanded, kernel_size=1)
        self.depthwise = ConvBnAct(expanded, expanded, kernel_size=kernel_size, 
                                stride=stride, padding=padding, groups=expanded)
        self.se = SEBlock(expanded, r=r)
        self.reduce_pw = ConvBnAct(expanded, n_out, kernel_size=1,
                                act=False)
        self.dropsample = DropSample(p)
  
    def forward(self, x):
        residual = x

        x = self.expand_pw(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.reduce_pw(x)

        if self.skip_connection:
            x = self.dropsample(x)
            x = x + residual

        return x
            
    
class MBConv1(MBConvN):
    def __init__(self, n_in, n_out, kernel_size=3,
                stride=1, r=24, p=0):
        super().__init__(n_in, n_out, expansion_factor=1,
                        kernel_size=kernel_size, stride=stride,
                        r=r, p=p)
                     
                     
class MBConv6(MBConvN):
    def __init__(self, n_in, n_out, kernel_size=3,
                stride=1, r=24, p=0):
        super().__init__(n_in, n_out, expansion_factor=6,
                        kernel_size=kernel_size, stride=stride,
                        r=r, p=p)

