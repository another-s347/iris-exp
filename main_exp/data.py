import os
import sys
import pdb
import torch
import numpy as np
import pickle as pkl
from PIL import Image
from random import shuffle
import copy
from torchvision import datasets, transforms
import torch.nn.functional as F

balance_config = {
    "mnist":{
        2:29000,
        3:17000,
        5: 11200,
        8: 5400,
        10: 5400
    },
    "kmnist":{
        2:None,
        3:None,
        5: None
    },
    "fmnist":{
        2:29000,
        3:17000,
        5: None
    },
    "cifar10":{
        3: None,
        5: None
    },
    "cifar100":{
        3: None,
        5: None
    }
}

def make_dataset(args, n, all, train, other=False):
    if args.dataset == "mnist":
        d = datasets.MNIST("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        targets = d.targets
    elif args.dataset == "kmnist":
        d = datasets.KMNIST("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        targets = d.targets
    elif args.dataset == "fmnist":
        d = datasets.FashionMNIST("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
        targets = d.targets
    elif args.dataset == "cifar10":
        d = datasets.CIFAR10("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
        targets = torch.tensor(d.targets)
    elif args.dataset == "cifar100":
        d = datasets.CIFAR100("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                ]))
        targets = torch.tensor(d.targets)
    else:
        raise NotImplementedError()

    if args.balance:
        try:
            tail = balance_config[args.dataset][all]
        except:
            tail = None
    else:
        tail = None
    
    step = 10 // all
    l = n * step
    h = (n+1) * step

    if other:
        indices = torch.where((~((targets >= l)*(targets < h))*(targets < (all*step))))[0]
    else:
        indices = torch.where((targets >= l)*(targets < h))[0][:tail]

    return torch.utils.data.Subset(d, indices)