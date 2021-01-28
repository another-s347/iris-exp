import argparse
import itertools
from queue import Queue
import random
from typing import Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import time
from threading import Lock
import copy
from client import remote

@remote()
def apply_grad(model, grads):
    state = model.state_dict()
    for k in grads.keys():
        state[k] = state[k] + grads[k]
    model.load_state_dict(state)

@remote()
def avg_grad(*grads):
    if len(grads) == 0:
        return ()
    keys = grads[0].keys()
    for k in keys:
        grads[0][k] = sum([c[k] for c in grads])
    return grads[0]

@remote()
def compute_dc_grad(model, current_model):
    model.temp_model = copy.deepcopy(current_model)
    cur_state = current_model.state_dict() # w_{t+r}
    grad = model.diff_model() # g(w_t)
    old_state = model.base_model.state_dict() # w_t
    for k in cur_state.keys():
        cur_state[k] = grad[k]*grad[k]*(cur_state[k]-old_state[k])
    return cur_state

@remote()
def apply_temp_model(model, state):
    model.base_model.load_state_dict(model.temp_model.state_dict())
    model.model.load_state_dict(model.temp_model.state_dict())
    # model.model = model.temp_model
    del model.temp_model
    model.apply_model(state)

class ClientNode:
    def __init__(self, model:nn.Module, rank) -> None:
        super().__init__()
        # model.apply(init)
        self.base_model: nn.Module = copy.deepcopy(model)
        self.model: nn.Module = model
        self.local_version = 0
        self.global_version = 0
        self.rank = rank
        self.sync = False
        self.lock = Lock()

    def set_sync(self, s):
        old = self.sync
        self.sync = s
        return old

    def bump_local(self, a=1):
        self.local_version += a
    
    def bump_global(self, v):
        self.global_version = v
        self.local_version = 0
    
    def diff_model(self):
        base = self.base_model.state_dict()
        current = self.model.state_dict()
        for k in current.keys():
            base[k] = current[k] - base[k]
        return base

    def apply_model(self, new_state):
        state = self.base_model.state_dict()
        for k in new_state.keys():
            state[k] = state[k] + new_state[k]
        self.model.load_state_dict(state)
        self.base_model.load_state_dict(state)
    
    def acquire(self):
        self.lock.acquire()

    def release(self):
        self.lock.release()

    def load_model(self, model):
        self.base_model.load_state_dict(model.state_dict())
        self.model.load_state_dict(model.state_dict())

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
