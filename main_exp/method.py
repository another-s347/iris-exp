from copy import Error
from net import *

def run_method(args, node, delayed_node, not_delayed_node, current_models, global_version, server_model = None):
    if args.method == "ours":
        ours(args, node, delayed_node, not_delayed_node, current_models, global_version)
    elif args.method == "ps":
        parameter_server(args, node, delayed_node, not_delayed_node, current_models, global_version, server_model)
    elif args.method == "ns":
        for c in current_models:
            c.bump_global(global_version+1)
        # simply do nothing
        pass

def ours(args, node, delayed_node, not_delayed_node, current_models, global_version):
    if len(delayed_node) == 0:
        grads = []
        for m in current_models:
            grads.append(m.diff_model())
        avged_grad = avg_grad.on(node)(*grads)
        for c in current_models:
            c.apply_model(avged_grad)
            c.set_sync(True)
            c.bump_global(global_version+1)
    else:
        if (not args.ignore_delay):
            if len(not_delayed_node) == 0:
                not_delayed_node
            # compute w(t+r)
            grads = []
            for m in not_delayed_node:
                grads.append(m.diff_model())
            avged_grad = avg_grad.on(node)(*grads)
            for c in not_delayed_node:
                c.set_sync(True)
                c.apply_model(avged_grad)

            cur = not_delayed_node[0].model
            # cur = not_delayed_node[0].model.to_node(edge_model.node)

            delayed_grads = []
            for m in delayed_node:
                delayed_grads.append(compute_dc_grad.on(m.node)(m, cur))
            avged_delayed_grads = avg_grad.on(node)(*delayed_grads)
            for c in not_delayed_node:
                c.apply_model(avged_delayed_grads)
                c.set_sync(True)
            for c in delayed_node:
                apply_temp_model.on(c.node)(c, avged_delayed_grads)
                c.set_sync(True)

        for c in delayed_node:
            c.bump_global(global_version+1)
        for c in not_delayed_node:
            c.bump_global(global_version+1)

def parameter_server(args, node, delayed_node, not_delayed_node, current_models, global_version, server_model):
    if len(delayed_node) == 0:
        grads = []
        for m in current_models:
            grads.append(m.diff_model())
        avged_grad = avg_grad.on(node)(*grads)
        
        # apply to server model
        server_model.apply_model(avged_grad)
        for c in current_models:
            c.load_model(server_model.model)
            c.bump_global(global_version+1)
    else:
        if (not args.ignore_delay):
            # compute w(t+r)
            grads = []
            for m in not_delayed_node:
                grads.append(m.diff_model())
            avged_grad = avg_grad.on(node)(*grads)
            server_model.apply_model(avged_grad)

            cur = not_delayed_node[0].model.to_node(node)

            delayed_grads = []
            for m in delayed_node:
                delayed_grads.append(compute_dc_grad.on(m.node)(m, cur))
            avged_delayed_grads = avg_grad.on(node)(*delayed_grads)
            server_model.apply_model(avged_delayed_grads)

            for c in not_delayed_node:
                c.load_model(server_model.model)
            for c in delayed_node:
                c.load_model(server_model.model)

        for c in delayed_node:
            c.bump_global(global_version+1)
        for c in not_delayed_node:
            c.bump_global(global_version+1)