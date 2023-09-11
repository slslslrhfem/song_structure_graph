"""
Code from https://gaussian37.github.io/dl-concept-dilated_residual_network/
"""
import math
import torch.nn as nn

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'BasicBlock', 'Bottleneck']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    '''
    - 입력 : 입력 채녈 수, 출력 채널 수, stride
    - 출력 : convolution 필터를 적용한 feature
    - 3x3 필터를 사용하는 기본적인 convolution 필터 함수
    - #filter = 3x3, #padding = 1로 고정
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    '''
    ResNet BasicBlock
    - 입력 : in_planes(입력 채널 수), out_planes(출력 채널 수), stride, dilation, downsample, previous_dilation
    - 출력 : BasicBlock 객체 
    - Convolution - BatchNorm - ReLU 2번을 하면서 skip connection을 만든다. 필요 시 downsample도 수행함
    '''
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, dilation=1, downsample=None, previous_dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=previous_dilation, dilation=previous_dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
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
    '''
    ResNet Bottleneck
    - 입력 : in_planes(입력 채널 수), out_planes(출력 채널 수), stride, dilation, downsample, previous_dilation
    - 출력 : Bottleneck 객체 
    - Convolution - BatchNorm - ReLU 3번을 하면서 Bottleneck 구조와 skip connection을 만든다. 필요 시 downsample도 수행함
    '''
    expansion = 4
    def __init__(self, in_planes, out_planes, stride=1, dilation=1,
                 downsample=None, previous_dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=stride,
            padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(
            out_planes, out_planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def _sum_each(self, x, y):
        assert(len(x) == len(y))
        z = []
        for i in range(len(x)):
            z.append(x[i]+y[i])
        return z

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


class ResNet(nn.Module):
    """Dilated Pre-trained ResNet Model, which reduces the stride of 8 featuremaps at conv5.
    Parameters
    ----------
    - block : BasicBlock 또는 Bottleneck
        - Residual Block의 타입을 정합니다. 
        - 타입은 BasicBlock 또는 Bottleneck으로 위에 선언한 클래스에 해당합니다.
    layers : int 값을 가지는 list
        - 각각의 block에서의 layer 수를 입력
    classes : int, 기본값은 1000
        - classification 할 class의 갯수
    dilated : bool, 기본값은 True
        - Dilated Convolution을 적용 할 지에 대한 Bool 값
    """

    def __init__(self, block, layers, num_classes=128, dilated=True):
        self.in_planes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, dilation=12,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        if dilated:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(2560 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        if dilation == 1 or dilation == 2:
            layers.append(block(self.in_planes, planes, stride, dilation=1,
                                downsample=downsample, previous_dilation=dilation))
        elif dilation == 4:
            layers.append(block(self.in_planes, planes, stride, dilation=2,
                                downsample=downsample, previous_dilation=dilation))
        else:
            raise RuntimeError("=> unknown dilation size: {}".format(dilation))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation, previous_dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


