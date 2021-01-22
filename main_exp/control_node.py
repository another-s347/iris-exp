from export import ClientResult, EdgeResult
from typing import List, Any, Optional

from client.ext import IrisObject
from net import *
from queue import Queue
import random
import copy
import client
import method

class ControlNode:
    def __init__(self, iris_node: 'IrisObject', optim: Any, rank: int, name: str, next_node: Optional['ControlNode'] = None) -> None:
        super().__init__()
        self.name = name
        self.rank = rank
        self.next_node = next_node
        self.model = client.IrisModel(iris_node.model)
        self.iris_node = iris_node
        self.optim = optim
        self.sub_nodes:List['IrisObject'] = []
        if self.next_node is not None:
            self.next_node.reg_sub_node(self)

    def zero_grad(self):
        self.optim.zero_grad()
        if self.next_node is not None:
            self.next_node.zero_grad()
    
    def step(self):
        if self.next_node is not None:
            self.next_node.step()
        # print(f"step {self.name}")
        self.optim.step()

    def forward(self, *args: Any, **kwds: Any) -> Any:
        result = self.model(*args, **kwds)
        if self.next_node is not None:
            return self.next_node.forward(result, **kwds)
        else:
            return result

    def notify(self, *args, **kwds):
        pass

    def notify_finish(self, *args, **kwds):
        pass

    def acquire(self):
        self.iris_node.acquire()
        if self.next_node != None:
            self.next_node.acquire()
    
    def release(self):
        self.iris_node.release()
        if self.next_node != None:
            self.next_node.release()

    def reg_sub_node(self, sub_node: 'ControlNode'):
        self.sub_nodes.append(sub_node.iris_node)

class CloudControlNode(ControlNode):
    def __init__(self, iris_node: IrisObject, optim:Any, rank: int, name: str, next_node: Optional['ControlNode']) -> None:
        super().__init__(iris_node, optim=optim, rank=rank, name=name, next_node=next_node)
        self.notify_queue = Queue()

    def notify(self, *args, **kwds):
        rank = args[0]
        self.notify_queue.put(rank)

    def notify_finish(self, *args, **kwds):
        self.notify_queue.put(-1)

    def run(self, args):
        self.iris_node.ctx.control_context.set(client.ControlContext(cid=self.rank))
        counts = [0 for i in range(len(self.sub_nodes))]
        ignored_counts = [0 for i in range(len(self.sub_nodes))]
        ignore_padding = int(len(self.sub_nodes) / args.delay_rate)-len(self.sub_nodes) if args.delay_rate != 0. else 0 
        n_sync = [5 for i in range(len(self.sub_nodes))] if len(args.delay_config) == 0 else args.delay_config
        finish_flag = 0
        global_version = 0
        ignored_count = 0
        print(f"run {self.name}")
        run_result = EdgeResult()
        server_model = self.sub_nodes[0].model.to_node(self.iris_node.node)
        while True:
            r = self.notify_queue.get()
            if r == -1:
                finish_flag += 1
                if finish_flag == len(self.sub_nodes):
                    break
            else:
                counts[r] += 1
            if sum([counts[i]-n_sync[i] for i in range(2)]) > 0:
                delayed_node = []
                not_delayed_node = []
                for i, m in enumerate(self.sub_nodes):
                    if m.global_version.get() != global_version:
                        print(f"[{self.name}] delayed node {m.rank.get()}")
                        delayed_node.append(m)
                    else:
                        not_delayed_node.append(m)

                if (args.delay_rate != 0.) and (len(not_delayed_node) > 0):
                    # ignore one client node randomly to simulate delayed nodes, or select additional one for not having delayed node 
                    ignored_node_index = random.randint(0, len(self.sub_nodes)+ignore_padding)
                    if ignored_node_index >= len(self.sub_nodes):
                        print(f"do not ignore nodes")
                        current_models = self.sub_nodes
                    else:
                        ignored_count += 1
                        ignored_counts[ignored_node_index] += 1
                        print(f"ignore node {ignored_node_index}")
                        current_models = copy.copy(self.sub_nodes)
                        del current_models[ignored_node_index]
                else:
                    current_models = self.sub_nodes
                # do sync
                # clean counts
                for i in range(2):
                    counts[i] = 0
                print(f"[{self.name}] sync: {global_version}")
                
                for m in self.sub_nodes:
                    m.acquire()
                
                method.run_method(args, self.iris_node.node, delayed_node, not_delayed_node, current_models, global_version, server_model)

                for m in self.sub_nodes:
                    m.release()

                # accumulate global version
                global_version += 1
        
        for m in self.sub_nodes:
            m.acquire()
        
        method.run_method(args, self.iris_node.node, [], self.sub_nodes, self.sub_nodes, global_version, server_model)

        for m in self.sub_nodes:
            m.release()

        if self.next_node is not None:
            self.next_node.notify_finish(-1)
        print(f"{self.name} done, ignored rate: {ignored_count / global_version}")
        run_result.delayed_count_total = ignored_count
        run_result.delayed_counts = ignored_counts
        run_result.delayed_rate = ignored_count / global_version
        run_result.sync_count = global_version
        run_result.sync_config = n_sync
        run_result.delayed_padding = ignore_padding
        return run_result

class EdgeControlNode(CloudControlNode):
    def __init__(self, iris_node: 'IrisObject', optim: Any, rank: int, name: str, next_node: Optional['ControlNode']) -> None:
        super().__init__(iris_node, optim, rank, name, next_node=next_node)

    def step(self):
        super().step()    
        if self.next_node is not None:
            self.next_node.notify(self.rank)
    
class ClientControlNode(ControlNode):
    def __init__(self, iris_node: 'IrisObject', optim:Any, rank: int, next_node: Optional['ControlNode']) -> None:
        super().__init__(iris_node, optim, rank, name=f"client:{rank}", next_node=next_node)
    
    def run(self, loader, loss_fn) -> 'ClientResult':
        self.iris_node.ctx.control_context.set(client.ControlContext(cid=self.rank))
        run_result: ClientResult = ClientResult()
        run_result.len_dataset = len(loader.dataset)
        run_result.rank = self.rank
        for epoch in range(1):
            for batch_idx, data in enumerate(loader,1):
                data, target = data[0], data[1]
                self.acquire()
                self.zero_grad()

                result = self.forward(data)
                loss = loss_fn.on(result.node)(result, target)
                loss.backward()

                self.step()
                if self.next_node is not None:
                    self.next_node.notify(self.rank)

                self.release()

                loss_value = loss.get().item()
                if self.iris_node.set_sync(False).get():
                    run_result.sync_flags.append(len(run_result.losses))
                run_result.losses.append(loss_value)
                
                if batch_idx % 5 == 0:
                    print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                        self.rank,
                        epoch, batch_idx * len(data), len(loader.dataset),
                        100. * batch_idx / len(loader),loss_value))

        print(f"node {self.rank} done")
        if self.next_node is not None:
            self.next_node.notify_finish()
        
        return run_result