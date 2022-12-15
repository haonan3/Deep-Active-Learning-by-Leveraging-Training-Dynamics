import torch
import torch.nn.functional as F
import torch.nn as nn
import math

from src.utils import LogitLoss

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3,
        stride=stride, padding=1, bias=False
    )


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, have_downsample=0):
        super(BasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = have_downsample
        if have_downsample:
            self.downsample_conv = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
            self.downsample_bn = nn.BatchNorm2d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample_conv(x)
            residual = self.downsample_bn(residual)

        out = out+residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_blocks, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, num_blocks[0])
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(256, num_blocks[2], stride=2)
        self.inside_dropout = nn.Dropout(p=0.5)
        self.layer4 = self._make_layer(512, num_blocks[3], stride=2)
        # I changed 7 -> 10 because 224 -> 299
        self.avgpool = nn.AvgPool2d(10)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

        self.num_params = self.count_parameters()
        self.compute_CELoss = nn.CrossEntropyLoss()
        self.compute_MSELoss = nn.MSELoss()
        self.compute_LogitLoss = LogitLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _make_layer(self, planes, blocks, stride=1):
        have_downsample = 0
        if stride != 1:
            have_downsample = 1
        layers = [BasicBlock(self.inplanes, planes, stride, have_downsample)]
        self.inplanes = planes

        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.predict(x)
        return x

    def get_embedding(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.inside_dropout(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        return out

    def get_embedding_dim(self):
        return 512

    def count_parameters_each_layer(self):
        return [p.numel() for p in self.parameters() if p.requires_grad]

    def collect_grad_each_layer(self, layer_id):
        # return self.parameters()[layer_id].grad.detach().reshape(-1,)
        counter = 0
        for p in self.parameters():
            if p.requires_grad:
                if counter == layer_id:
                    return p.grad.detach().reshape(-1, )
                else:
                    counter += 1

    def collect_grad(self):
        return torch.cat([p.grad.detach().reshape(-1, ) for _, p in self.named_parameters() if p.requires_grad], dim=0)

    def predict(self, out):
        out = self.fc(out)
        return out


def caltech_resnet18(out_dim):
    model = ResNet([2, 2, 2, 2], num_classes=out_dim)
    return model


############################################


def get_model(args):
    model = caltech_resnet18(args.class_num)

    # # make all params untrainable
    # for p in model.parameters():
    #     p.requires_grad = False

    # reset the last fc layer
    # normal(model.fc.weight, 0.0, 0.01)
    # constant(model.fc.bias, 0.0)

    # make some other params trainable
    # trainable_params = []
    # trainable_params += [n for n, p in model.named_parameters() if 'layer4' in n or 'layer3' in n]
    # for n, p in model.named_parameters():
    #     if n in trainable_params:
    #         p.requires_grad = True
    #
    # for m in model.layer4.modules():
    #     if isinstance(m, nn.ReLU):
    #         m.inplace = False
    #
    # for m in model.layer3.modules():
    #     if isinstance(m, nn.ReLU):
    #         m.inplace = False

    # # create different parameter groups
    # classifier_weights = [model.fc.weight]
    # classifier_biases = [model.fc.bias]
    # features_weights = [
    #     p for n, p in model.named_parameters()
    #     if n in trainable_params and 'conv' in n
    # ]
    # features_weights += [
    #     p for n, p in model.named_parameters()
    #     if n in trainable_params and 'downsample.0' in n and 'weight' in n
    # ]
    # features_bn_weights = [
    #     p for n, p in model.named_parameters()
    #     if n in trainable_params and 'weight' in n and ('bn' in n or 'downsample.1' in n)
    # ]
    # features_bn_biases = [
    #     p for n, p in model.named_parameters()
    #     if n in trainable_params and 'bias' in n
    # ]

    # # you can set different learning rates
    # classifier_lr = 1e-2
    # features_lr = 1e-2
    # # but they are not actually used (because lr_scheduler is used)
    #
    # params = [
    #     {'params': classifier_weights, 'lr': classifier_lr, 'weight_decay': 1e-3},
    #     {'params': classifier_biases, 'lr': classifier_lr},
    #     {'params': features_weights, 'lr': features_lr, 'weight_decay': 1e-3},
    #     {'params': features_bn_weights, 'lr': features_lr},
    #     {'params': features_bn_biases, 'lr': features_lr}
    # ]
    # optimizer = optim.SGD(params, momentum=0.9, nesterov=True)
    #
    # # loss function
    # criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()
    # # move the model to gpu
    # model = model.cuda()
    return model  # , criterion, optimizer
