from json import load
import os
import argparse
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F 
import sys
import torch
import numpy as np

# sys.path.append("../../")
torch.manual_seed(0)

from fedlab.core.client.scale.trainer import SubsetSerialTrainer, SubsetSerialTrainerWithDifferentialUpdate
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu, load_dict
import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
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


class ResNet(nn.Module):
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

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    # return ResNet(BasicBlock, [2, 2, 2, 2])
    resnet18 = torchvision.models.resnet18()
    resnet18.fc = nn.Linear(512, 10)

    # Change BN to GN 
    resnet18.bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)

    resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
    resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)
    resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 64)
    resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 64)

    resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 128)
    resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 128)

    resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 256)
    resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 256)

    resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups = 2, num_channels = 512)
    resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups = 2, num_channels = 512)

    assert len(dict(resnet18.named_parameters()).keys()) == len(resnet18.state_dict().keys()), 'More BN layers are there...'
    return resnet18

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])

# configuration
parser = argparse.ArgumentParser(description="Standalone training example")
parser.add_argument("--total_client", type=int, default=100)
parser.add_argument("--com_round", type=int, default=10)

parser.add_argument("--sample_ratio", type=float)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--cuda", type=bool, default=False)

args = parser.parse_args()





# get mnist dataset
root = "../dataSet/cifar10/"
trainset = torchvision.datasets.CIFAR10(root=root,
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                                transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomCrop(32, 4),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]))

testset = torchvision.datasets.CIFAR10(root=root,
                                     train=False,
                                     download=True,
                                     transform=transforms.Compose([transforms.ToTensor(),
                                                                   transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])]))

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=256,
                                          drop_last=False,
                                          shuffle=False)


# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


# model = torchvision.models.resnet18(pretrained=False)
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 10)
model = ResNet18()

if args.cuda:
    gpu = get_best_gpu()
    print("using {}".format(gpu))
    model = model.cuda(gpu)


# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate
total_client_num = args.total_client  # client总数

data_indices = CIFAR10Partitioner(trainset.targets, args.total_client, True, "iid").client_dict

# fedlab setup
local_model = deepcopy(model)

trainer = SubsetSerialTrainer(model=local_model,
                              dataset=trainset,
                              data_slices=data_indices,
                              aggregator=aggregator,
                              args={
                                  "batch_size": args.batch_size,
                                  "epochs": args.epochs,
                                  "max_norm":10,
                                  "optim":{"type":"SGD", "lr": args.lr, "weight_decay":1e-3}
                              })

# train procedure
to_select = [i for i in range(total_client_num)]
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_state_dict(model)
    selection = random.sample(to_select, num_per_round)
    aggregated_parameters = trainer.train(model_parameters=model_parameters,
                                          id_list=selection,
                                          aggregate=True)

    SerializationTool.deserialize_state_dict(model, aggregated_parameters)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("round {}, loss: {:.4f}, acc: {:.2f}".format(round + 1, loss, acc))
