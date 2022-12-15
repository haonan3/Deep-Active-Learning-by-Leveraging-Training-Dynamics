'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

import torch
import torch.nn.functional as F


import torch.nn as nn

from src.utils import LogitLoss


class CNNAvgPool(nn.Module):
    def __init__(self):
        super(CNNAvgPool, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(128, 10)
        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def count_parameters_each_layer(self):
        return [p.numel() for p in self.parameters() if p.requires_grad]


    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x


    def get_embedding(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).reshape((x.shape[0], -1))
        return x


    def predict(self, x):
        # x = F.relu(self.fc1(x))
        x = self.fc1(x)
        return x


    def collect_grad(self):
        return torch.cat([p.grad.detach().view(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)

    def get_embedding_dim(self):
        return 128



class CNNnet(nn.Module):
    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def count_parameters_each_layer(self):
        return [p.numel() for p in self.parameters() if p.requires_grad]


    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x


    def get_embedding(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


    def predict(self, x):
        x = self.fc3(x)
        return x


    def collect_grad(self):
        return torch.cat([p.grad.detach().view(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)

    def get_embedding_dim(self):
        return 84


class MLP(nn.Module):

    def __init__(self, args, input_dim, output_dim):
        super().__init__()
        self.args = args
        self.input_fc = nn.Linear(input_dim, args.hidden)
        self.output_fc = nn.Linear(args.hidden, output_dim)
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()

    def forward(self, x):
        h = self.get_embedding(x)
        y_pred = self.output_fc(h)
        return y_pred


    def get_embedding(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.normalize(x, p=2)
        x = F.relu(self.input_fc(x))
        return x


    def NTK_predict(self, x):
        x = self.output_fc(x)
        return x


    def collect_grad(self):
        return torch.cat([p.grad.detach().reshape(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def get_embedding_dim(self):
        return self.args.hidden


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.have_shortcut = False
        # self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.have_shortcut = True
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_planes, self.expansion*planes,
            #               kernel_size=1, stride=stride, bias=False),
            #     nn.BatchNorm2d(self.expansion*planes)
            # )
            self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.have_shortcut:
            temp_ = self.shortcut_conv(x)
            temp_ = self.shortcut_bn(temp_)  # self.shortcut(x)
            out = out + temp_
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module): # (BasicBlock, [2, 2, 2, 2])
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()

        self.embDim = 512*block.expansion

    def get_embedding_dim(self):
        return self.embDim


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x


    def get_embedding(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


    def count_parameters_each_layer(self):
        return [p.numel() for p in self.parameters() if p.requires_grad]


    def collect_grad_each_layer(self, layer_id):
        # return self.parameters()[layer_id].grad.detach().reshape(-1,)
        counter = 0
        for p in self.parameters():
            if p.requires_grad:
                if counter == layer_id:
                    return p.grad.detach().reshape(-1,)
                else:
                    counter += 1


    def collect_grad(self):
        return torch.cat([p.grad.detach().reshape(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)


    def predict(self, out):
        out = self.linear(out)
        return out


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}



class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()
        # layer_config = cfg[vgg_name]
        # self.features = self._make_layers(cfg[vgg_name])
        in_channels = 3
        x=64
        self.cn1 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(x)
        self.relu1 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        x=128
        self.cn2 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(x)
        self.relu2 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        x=256
        self.cn3 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(x)
        self.relu3 = nn.ReLU(inplace=True)
        in_channels = x
        x=256
        self.cn4 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(x)
        self.relu4 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        x=512
        self.cn5 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(x)
        self.relu5 = nn.ReLU(inplace=True)
        in_channels = x
        x=512
        self.cn6 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(x)
        self.relu6 = nn.ReLU(inplace=True)
        in_channels = x
        self.mpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        x=512
        self.cn7 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(x)
        self.relu7 = nn.ReLU(inplace=True)
        in_channels = x
        x=512
        self.cn8 = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(x)
        self.relu8 = nn.ReLU(inplace=True)
        self.mpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.feat_extractor = nn.ModuleList([self.cn1, self.bn1, self.relu1, self.mpool1,
                                             self.cn2, self.bn2, self.relu2, self.mpool2,
                                             self.cn3, self.bn3, self.relu3, self.cn4, self.bn4, self.relu4, self.mpool3,
                                             self.cn5, self.bn5, self.relu5, self.cn6, self.bn6, self.relu6, self.mpool4,
                                             self.cn7, self.bn7, self.relu7, self.cn8, self.bn8, self.relu8, self.mpool4,])

        self.linear = nn.Linear(512, 10)
        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x


    def predict(self, out):
        out = self.linear(out)
        return out


    def get_embedding(self, x):
        out = x
        for layer in self.feat_extractor:
            out = layer(out)
        # out = self.features(x)
        emb = out.view(out.size(0), -1)
        return emb



    def _make_layers(self, cfg):
        layers = []

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def collect_grad(self):
        return torch.cat([p.grad.detach().reshape(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)



class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        layer_config = cfg[vgg_name]
        self.features = self._make_layers(cfg[vgg_name])
        self.linear = nn.Linear(512, 10)
        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x


    def predict(self, out):
        out = self.linear(out)
        return out


    def get_embedding(self, x):
        out = self.features(x)
        emb = out.view(out.size(0), -1)
        return emb



    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def get_embedding_dim(self):
        return 512


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def collect_grad(self):
        return torch.cat([p.grad.detach().reshape(-1,) for _, p in self.named_parameters() if p.requires_grad], dim=0)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def resnet_test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())
    print(net.num_params)

def vgg_test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    out, emb = net(x)
    print(out.shape)
    print(emb.shape)




if __name__ == '__main__':
    vgg_test()