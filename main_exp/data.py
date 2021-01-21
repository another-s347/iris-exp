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
        3:17000
    },
    "kmnist":{
        2:None,
        3:None
    }
}

def make_dataset(args, n, all, train, other=False):
    if args.dataset == "mnist":
        d = datasets.MNIST("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    elif args.dataset == "kmnist":
        d = datasets.KMNIST("../data", train=train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ]))
    else:
        raise NotImplementedError()

    if args.balance:
        tail = balance_config[args.dataset][all]
    else:
        tail = None
    
    step = 10 // all
    l = n * step
    h = (n+1) * step

    if n+1 == all and h != 10:
        h = 10

    if other:
        indices = torch.where(~((d.targets >= l)*(d.targets < h)))[0]
    else:
        indices = torch.where((d.targets >= l)*(d.targets < h))[0][:tail]

    return torch.utils.data.Subset(d, indices)