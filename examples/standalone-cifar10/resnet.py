from distutils import core
from json import load
import os
import argparse
from pickletools import optimize
import random
from copy import deepcopy
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.nn import functional as F 
import sys
import torch
import numpy as np
import tqdm
# sys.path.append("../../")
torch.manual_seed(0)

from fedlab.core.client.scale.trainer import SubsetSerialTrainer, SubsetSerialTrainerWithDifferentialUpdate
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.functional import evaluate
from fedlab.utils.functional import get_best_gpu, load_dict
from r18 import ResNet18
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

train_loader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size,
                                          drop_last=False,
                                          shuffle=True)

test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=256,
                                          drop_last=False,
                                          shuffle=False)


# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)


optim = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
if args.cuda:
    gpu = get_best_gpu()
    print("using {}".format(gpu))
    model = model.cuda(gpu)
criterion = nn.CrossEntropyLoss()
for round in range(args.com_round):
    train_loss = 0
    correct = 0
    total = 0
    for x, y in tqdm.tqdm(train_loader):
        model.train()
        optim.zero_grad()
        if args.cuda:
            x = x.cuda()
            y = y.cuda()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        _, predicted = out.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
    print("[train]round {}, loss: {:.4f}, acc: {:.2f}".format(round + 1, train_loss, correct / total))
        
    loss, acc = evaluate(model, criterion, test_loader)
    print("[test]round {}, loss: {:.4f}, acc: {:.2f}".format(round + 1, loss, acc))
