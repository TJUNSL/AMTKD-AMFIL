# https://raw.githubusercontent.com/miraclewkf/ResNeXt-PyTorch/master/resnext.py
'''
New for ResNeXt:
1. Wider bottleneck
2. Add group for conv2
'''

import torch.nn as nn
import math

__all__ = ['ResNeXt', 'resnext18', 'resnext34', 'resnext50', 'resnext101',
           'resnext152']

def conv3x3(in_planes, out_planes, stride=1,groups=num_group):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False,groups=num_group)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes*2, stride)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes*2, planes*2, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, num_group=32):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes*2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes*2)
        self.conv2 = nn.Conv2d(planes*2, planes*2, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(planes*2)
        self.conv3 = nn.Conv2d(planes*2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):

    def __init__(self, block, layers, num_classes=1000, num_group=32,args=None):
        self.inplanes = 64
        self.num_group = num_group
        super(ResNeXt, self).__init__()
    
        if 'ifamilymalware' in args["dataset"]:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )

        if 'idrebin' in args["dataset"]:
            self.conv1 = nn.Sequential(
                nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],num_group = self.num_group)
        self.layer2 = self._make_layer(block, 128, layers[1], num_group = self.num_group, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], num_group = self.num_group, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], num_group = self.num_group, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 512 * block.expansion
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, num_group, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.num_group))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, num_group=self.num_group))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        pooled = self.avgpool(x_4)
        features = torch.flatten(pooled, 1)
        # x = self.fc(x)

        # return x
        return {
            'fmaps': [x_1, x_2, x_3, x_4],
            'features': features,
            'IB_features': torch.sum(torch.abs(features), 1).reshape(-1, 1)
        }
    
    def forward(self, x):
        return self._forward_impl(x)

def _resnext(block,layers,**kwargs):
    model = ResNeXt(block, layers, **kwargs)
    return model


def resnext18(**kwargs):
    """Constructs a ResNeXt-18 model.
    """
    # model = ResNeXt(BasicBlock, [2, 2, 2, 2], **kwargs)
    return _resnext(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnext34(**kwargs):
    """Constructs a ResNeXt-34 model.
    """
    # model = ResNeXt(BasicBlock, [3, 4, 6, 3], **kwargs)
    return _resnext(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnext50(**kwargs):
    """Constructs a ResNeXt-50 model.
    """
    # model = ResNeXt(Bottleneck, [3, 4, 6, 3], **kwargs)
    return _resnext(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnext101(**kwargs):
    """Constructs a ResNeXt-101 model.
    """
    # model = ResNeXt(Bottleneck, [3, 4, 23, 3], **kwargs)
    return _resnext(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext152(**kwargs):
    """Constructs a ResNeXt-152 model.
    """
    # model = ResNeXt(Bottleneck, [3, 8, 36, 3], **kwargs)
    return _resnext(Bottleneck, [3, 8, 36, 3], **kwargs)