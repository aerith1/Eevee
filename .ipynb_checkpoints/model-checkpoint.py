import torch
from torch import nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch, block_ch, stride=1, downsample = None):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_ch, block_ch, kernel_size=3, stride = stride,  padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(block_ch, block_ch, kernel_size=3, stride = 1, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(block_ch)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu2(out)
        
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, block_ch, stride=1, downsample = None):
        super().__init__()
        self.downsample = downsample

        self.conv1 = nn.Conv2d(in_ch, block_ch, kernel_size=1, stride = stride, bias=False)
        self.bn1 = nn.BatchNorm2d(block_ch)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(block_ch, block_ch, kernel_size=3, stride = 1, padding=1, bias = False)
        self.bn2 = nn.BatchNorm2d(block_ch)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(block_ch, block_ch * self.expansion, kernel_size=1, stride=1, bias = False)
        self.bn3 = nn.BatchNorm2d(block_ch * self.expansion)
        self.relu3 = nn.ReLU()


    def forward(self, x):
        identity = x
        # print('identity shape ', identity.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
            # print('identity shape ', identity.shape)
        out = self.relu1(self.bn1(self.conv1(x)))
        # print('out shape ', out.shape)

        out = self.relu2(self.bn2(self.conv2(out)))
        # print('out shape ', out.shape)

        out = self.bn3(self.conv3(out))    
        # print('out shape ', out.shape)

        out += identity
        out = self.relu3(out)
        return self.relu3(out)

class ResNet(nn.Module):
    def __init__(self, in_ch = 3, include_top = True, num_class = 9, block = BasicBlock, block_num = [2, 2, 2, 2]):
        super().__init__()
        self.include_top = include_top

        self.in_ch = in_ch
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.in_ch = 64
        
        self.layer1 = self._make_layer(block, 64, block_num[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, block_num[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, block_num[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, block_num[3], stride = 2)
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(512 * block.expansion * 7 * 7, num_class),
        #     nn.Softmax(dim = -1)
        # )
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, block_ch, block_num, stride = 1):
        layers = []
        downsample = nn.Conv2d(self.in_ch, block_ch * block.expansion, kernel_size=1, stride=stride)
        layers += [block(self.in_ch, block_ch, stride=stride, downsample = downsample)]
        self.in_ch = block_ch * block.expansion

        for _ in range(1, block_num):
            layers += [block(self.in_ch, block_ch)]
        return nn.Sequential(*layers)
        

    def forward(self, x):
        # print('conv1 shape', self.conv1(x).shape)
        # print('---------')
        # print('pool1 shape', self.maxpool1(self.bn1(self.conv1(x))).shape)

        out = self.maxpool1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # print(out.shape)
        if self.include_top:
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
        
        return out
