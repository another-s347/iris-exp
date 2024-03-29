from typing import List
import client
import pickle
import sys
from client.ext import IrisNode
import torch
import model as model_local
import net as net_local
from torchvision import datasets, transforms
import torch.nn.functional as F
import itertools
import datetime
import time
from client import remote, ControlContext, on, proxy
from concurrent.futures import ThreadPoolExecutor
import queue
import data
import control_node
import argparse
import export

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true", default=False, help="enable iris logging")
parser.add_argument("--color", action="store_true", default=False, help="enable iris logging color")
parser.add_argument("--dataset", default="mnist", choices=["mnist","kmnist","fmnist","cifar10","cifar100"], help="dataset")
parser.add_argument("--delay_rate", type=float, default=0., help="dalayed rate")
parser.add_argument("--delay_config", type=int, nargs="*")
parser.add_argument("--method", type=str, default="ours", choices=["ns","ours","ps"])
parser.add_argument("--run_async", action="store_true", default=False)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--balance", type=bool, default=True)
parser.add_argument("--suffix", type=str, default="default")
parser.add_argument("--export", action="store_true", default=False)
parser.add_argument("--no_edge", action="store_true", default=False)
parser.add_argument("--no_test", action="store_true", default=False)
parser.add_argument("--device", default="cpu", choices=["cuda","cpu"])
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--nodes", type=int, default=3)
parser.add_argument("--ignore_delay", action="store_true", default=False)
parser.add_argument("--local", action="store_true", default=False)
parser.add_argument("--epoch_test", type=int, default=0)
args = parser.parse_args()

print(vars(args))

config = client.IrisConfig()
# config.go_async = True
# config.go_async_sequence = True
config.go_async = args.run_async
config.go_async_sequence = True
config.debug = args.log
config.log_color = args.color

run_result = export.RunResult()
run_result.args = args
run_result.datetime = str(datetime.datetime.now())
run_result.suffix = args.suffix

c = client.IrisContext(config)

net = proxy(net_local, c)
torch = proxy(torch, c)
data = proxy(data, c)

if args.local:
    clients = [c.create_node(f"node{x}",ip=f"local",port=12345) for x in range(args.nodes)]
    for i in range(1, len(clients)):
        clients[i-1].connect(clients[i], bi=True)
    clients[0].connect(clients[-1],bi=True)
    # edge nodes
    edge_node = c.create_node("edge",ip="local",port=12345)
    for n in clients:
        n.connect(edge_node, bi=True)
    # cloud nodes
    cloud_node = c.create_node("cloud", ip="local",port=12345)
    edge_node.connect(cloud_node, bi=True)
    for n in clients:
        n.connect(cloud_node, bi=True)
    # todo: connect
else:
    # client nodes
    clients = [c.create_node(f"node{x}",ip=f"10.0.0.{x+1}",port=12345) for x in range(args.nodes)]
    for i in range(1, len(clients)):
        clients[i-1].connect(clients[i], bi=True)
    clients[0].connect(clients[-1],bi=True)
    # edge nodes
    edge_node = c.create_node("edge",ip="10.0.0.14",port=12345)
    for n in clients:
        n.connect(edge_node, bi=True)
    # cloud nodes
    cloud_node = c.create_node("cloud", ip="10.0.0.15",port=12345)
    edge_node.connect(cloud_node, bi=True)
    for n in clients:
        n.connect(cloud_node, bi=True)
    # todo: connect

@remote()
def compute_correct(result, target):
    pred = result.argmax(dim=1, keepdim=True)
    return pred.eq(target.view_as(pred)).sum().item()

client_model, edge_model, cloud_model = model_local.make_model(args)

models: List[control_node.ClientControlNode] = []
train_loaders = []
test_self_loaders = []
test_other_loaders = []

with on(cloud_node):
    model_cloud = cloud_node.send(cloud_model)
    optimizer_cloud = torch.optim.SGD(model_cloud.parameters(), lr=args.lr)
    cloud_control_node = control_node.CloudControlNode(net.ClientNode(model_cloud, 10),None, optimizer_cloud, 10, "cloud", None)

with on(edge_node):
    model_edge = edge_node.send(edge_model)
    server_model = edge_node.send(client_model)
    optimizer_edge = torch.optim.SGD(model_edge.parameters(), lr=args.lr)
    edge_control_node = control_node.CloudControlNode(net.ClientNode(model_edge, 5),net.ClientNode(server_model, 6), optimizer_edge, 5, "edge", cloud_control_node)

for i, node in enumerate(clients):
    with on(node):
        model = node.send(client_model)
        dataset = data.make_dataset(args, n=i, all=len(clients), train=True, other=False)
        train_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=64)
        test_self_dataset = data.make_dataset(args, n=i, all=len(clients), train=False, other=False)
        test_other_dataset = data.make_dataset(args, n=i, all=len(clients), train=False, other=True)
        test_self_loader = torch.utils.data.DataLoader(test_self_dataset, batch_size=64)
        test_other_loader = torch.utils.data.DataLoader(test_other_dataset, batch_size=64)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

        model = net.ClientNode(model, i)
        models.append(control_node.ClientControlNode(model, optim=optimizer, rank=i, next_node=edge_control_node))
        train_loaders.append(train_loader)
        test_self_loaders.append(test_self_loader)
        test_other_loaders.append(test_other_loader)

loss_fn = client.RemoteTensorFunction(F.cross_entropy)

# models[0].run(train_loaders[0], nll_loss)
with ThreadPoolExecutor(30) as executor:
    if (not args.no_edge) and args.nodes > 1:
        edge_future = executor.submit(edge_control_node.run, args)
        cloud_future = executor.submit(cloud_control_node.run, args)
    futures = [executor.submit(models[i].run, args, train_loaders[i], loss_fn, test_self_loaders[i], test_other_loaders[i]) for i in range(len(clients))]
    client_results = [f.result() for f in futures]
    if (not args.no_edge) and args.nodes > 1:
        run_result.edge_results = edge_future.result()
    for i, r in enumerate(client_results):
        run_result.client_results[i] = r

if not args.no_test:
    for i in range(len(clients)):
        eval_result = export.EvalResult()
        eval_result.rank = i
        correct = 0.
        test_loss = 0.
        len_testdataset = len(test_self_loaders[i].dataset)
        if len_testdataset > 0:
            for batch_idx, data in enumerate(test_self_loaders[i]):
                data, target = data[0].to(args.device), data[1].to(args.device)
                result = models[i].forward(data)
                correct += compute_correct.on(result.node)(result, target).get()
                test_loss /= len_testdataset

            print('\n[{}] Self test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    i,
                    test_loss, correct, len_testdataset,
                    100. * correct / len_testdataset))

            eval_result.self_acc = 100. * correct / len_testdataset
            eval_result.self_correct = correct

        correct = 0.
        test_loss = 0.
        len_testdataset = len(test_other_loaders[i].dataset)
        if len_testdataset > 0:
            for batch_idx, data in enumerate(test_other_loaders[i]):
                data, target = data[0].to(args.device), data[1].to(args.device)
                result = models[i].forward(data)
                correct += compute_correct.on(result.node)(result, target).get()
                test_loss /= len_testdataset

            print('\n[{}] Other test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                    i,
                    test_loss, correct, len_testdataset,
                    100. * correct / len_testdataset))

            eval_result.other_acc = 100. * correct / len_testdataset
            eval_result.other_correct = correct
            run_result.eval_results[i] = eval_result

    correct = 0.
    test_loss = 0.
    len_testdataset = len(test_self_loaders[0].dataset) + len(test_other_loaders[0].dataset)
    for batch_idx, data in enumerate(itertools.chain(test_self_loaders[0],test_other_loaders[0])):
        data, target = data[0].to(args.device), data[1].to(args.device)
        result = models[0].iris_node.base_model(data)
        result = edge_control_node.iris_node.model(result)
        result = cloud_control_node.iris_node.model(result)
        correct += compute_correct.on(result.node)(result, target).get()
        test_loss /= len_testdataset

    print('\nGlobal test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len_testdataset,
            100. * correct / len_testdataset))

    run_result.global_accs = 100. * correct / len_testdataset

c.close()
if args.export:
    run_result.write()