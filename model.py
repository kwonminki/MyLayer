import math
from torch import nn
import torchvision.models as models

from layer import ConvBnAct, MBConv1, MBConv6, ColorConstancySolved
from util import create_stage, scale_width


class AddLayerNet(nn.Module):
    def __init__(self, network, th, pretrian):
        super().__init__()
        self.th = th
        if th == -1:
            print("pure model")
        self.mine = ColorConstancySolved(self.th)

        if network == "resnet18":
            self.backbone = ResNet18(pretrian=pretrian)
        elif network == "alexnet":
            self.backbone = AlexNet(pretrian=pretrian)
        elif network == "googlenet":
            self.backbone = GoogleNet(pretrian=pretrian)
        elif network == "vgg16":
            self.backbone = Vgg16(pretrian=pretrian)
        elif network == "resnet34":
            self.backbone = ResNet34(pretrian=pretrian)
        elif network == "resnet50":
            self.backbone = ResNet50(pretrian=pretrian)
        elif network == "inception_v3":
            self.backbone = Inception_v3(pretrian=pretrian)
        elif network == "mobilenet_v2":
            self.backbone = MobileNet_v2(pretrian=pretrian)

    def forward(self, x):
        if self.th != -1:
            middle = self.mine(x)
        else :
            middle = x #pure model
        x = self.backbone(middle)

        return x, middle

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
        self.model = models.inception_v3(pretrained=pretrian)

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





class EfficientNet(nn.Module):
    """Generic EfficientNet that takes in the width and depth scale factors and scales accordingly"""
    def __init__(self, w_factor=1, d_factor=1,
                out_sz=1000, th=51):
        super().__init__()

        base_widths = [(32, 16), (16, 24), (24, 40),
                    (40, 80), (80, 112), (112, 192),
                    (192, 320), (320, 1280)]
        base_depths = [1, 2, 2, 3, 3, 4, 1]

        scaled_widths = [(scale_width(w[0], w_factor), scale_width(w[1], w_factor)) 
                        for w in base_widths]
        scaled_depths = [math.ceil(d_factor*d) for d in base_depths]
        
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        strides = [1, 2, 2, 2, 1, 2, 1]
        ps = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]

        self.stem = ConvBnAct(3, scaled_widths[0][0], stride=2, padding=1)
        
        stages = []
        for i in range(7):
            layer_type = MBConv1 if (i == 0) else MBConv6
            r = 4 if (i == 0) else 24
            stage = create_stage(*scaled_widths[i], scaled_depths[i],
                                layer_type, kernel_size=kernel_sizes[i], 
                                stride=strides[i], r=r, p=ps[i])
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        self.pre_head = ConvBnAct(*scaled_widths[-1], kernel_size=1)

        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                nn.Flatten(),
                                nn.Linear(scaled_widths[-1][1], out_sz))

    def feature_extractor(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.pre_head(x)
        return x

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.head(x)
        return x
    
    
class EfficientNetB0(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1
        d_factor = 1
        super().__init__(w_factor, d_factor, out_sz)
    
  
class EfficientNetB1(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1
        d_factor = 1.1
        super().__init__(w_factor, d_factor, out_sz)
      
    
class EfficientNetB2(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1.1
        d_factor = 1.2
        super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB3(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1.2
        d_factor = 1.4
        super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB4(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1.4
        d_factor = 1.8
        super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB5(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1.6
        d_factor = 2.2
        super().__init__(w_factor, d_factor, out_sz)
    
    
class EfficientNetB6(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 1.8
        d_factor = 2.6
        super().__init__(w_factor, d_factor, out_sz)
        
    
class EfficientNetB7(EfficientNet):
    def __init__(self, out_sz=1000):
        w_factor = 2
        d_factor = 3.1
        super().__init__(w_factor, d_factor, out_sz)