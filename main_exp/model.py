from torchvision import models
import torch
from torch import nn
import net

def make_model(args):
    if args.dataset in ["mnist","fmnist","kmnist"]:
        return make_2_conv(args)
    else:
        return make_resnet(args)

def make_resnet(args):
    resnet = models.resnet18(pretrained=True)
    net1 = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1
    )
    net2 = nn.Sequential(
        resnet.layer2
    )
    net3 = nn.Sequential(
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool,
        nn.Flatten(),
        resnet.fc
    )
    return net1.to(args.device), net2.to(args.device), net3.to(args.device)

def make_2_conv(args):
    return net.Net1().to(args.device), net.Net3().to(args.device), net.Net4().to(args.device)