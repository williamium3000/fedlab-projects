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
                                          batch_size=50,
                                          drop_last=False,
                                          shuffle=False)


# setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"


model = torchvision.models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)

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
                                  "optim":{"type":"SGD", "lr": args.lr, "weight_decay":5e-4, "momentum":0.9}
                              })

# train procedure
to_select = [i for i in range(total_client_num)]
for round in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    selection = random.sample(to_select, num_per_round)
    aggregated_parameters = trainer.train(model_parameters=model_parameters,
                                          id_list=selection,
                                          aggregate=True)

    SerializationTool.deserialize_model(model, aggregated_parameters)

    criterion = nn.CrossEntropyLoss()
    loss, acc = evaluate(model, criterion, test_loader)
    print("round {}, loss: {:.4f}, acc: {:.2f}".format(round + 1, loss, acc))
