import json
import matplotlib.pyplot as plt
from functools import partial

def plot_loss(suffix):
    f = open(f"./results/{suffix}/loss.json","r")
    d = json.load(f)

    for result in d['results']:
        y = d['results'][result]
        x = [i for i in range(len(y))]
        markers_on = d['sync_flags'][result]
        plt.plot(x, y, label=f"node-{result}", markevery = markers_on)
    plt.legend()
    plt.savefig("loss.pdf")

table1_template = r"""
\begin{{table}}[]
    \begin{{tabular}}{{|c|c|c|c|c|c|c|c|}}
    \hline
    \multicolumn{{2}}{{|c|}}{{\multirow{{2}}{{*}}{{}}}} & \multicolumn{{3}}{{c|}}{{MNIST}} & \multicolumn{{3}}{{c|}}{{KMNIST}} \\ \cline{{3-8}} 
    \multicolumn{{2}}{{|c|}}{{}}                        & 不同步     & OURS     & PS    & 不同步     & OURS     & PS     \\ \hline
    n0                        & Self                    & {n0_mnist_self_ns:.2f} & {n0_mnist_self_ours:.2f} & {n0_mnist_self_ps:.2f} & {n0_kmnist_self_ns:.2f} & {n0_kmnist_self_ours:.2f} & {n0_kmnist_self_ps:.2f} \\ \hline
                              & Other                   & {n0_mnist_othe_ns:.2f} & {n0_mnist_othe_ours:.2f} & {n0_mnist_othe_ps:.2f} & {n0_kmnist_othe_ns:.2f} & {n0_kmnist_othe_ours:.2f} & {n0_kmnist_othe_ps:.2f} \\ \hline
    \multirow{{2}}{{*}}{{n1}}       & Self              & {n1_mnist_self_ns:.2f} & {n1_mnist_self_ours:.2f} & {n1_mnist_self_ps:.2f} & {n1_kmnist_self_ns:.2f} & {n1_kmnist_self_ours:.2f} & {n1_kmnist_self_ps:.2f} \\ \cline{{2-8}} 
                              & Other                   & {n1_mnist_othe_ns:.2f} & {n1_mnist_othe_ours:.2f} & {n1_mnist_othe_ps:.2f} & {n1_kmnist_othe_ns:.2f} & {n1_kmnist_othe_ours:.2f} & {n1_kmnist_othe_ps:.2f} \\ \hline
    \multirow{{2}}{{*}}{{n2}}       & Self              & {n2_mnist_self_ns:.2f} & {n2_mnist_self_ours:.2f} & {n2_mnist_self_ps:.2f} & {n2_kmnist_self_ns:.2f} & {n2_kmnist_self_ours:.2f} & {n2_kmnist_self_ps:.2f} \\ \cline{{2-8}} 
                              & Other                   & {n2_mnist_othe_ns:.2f} & {n2_mnist_othe_ours:.2f} & {n2_mnist_othe_ps:.2f} & {n2_kmnist_othe_ns:.2f} & {n2_kmnist_othe_ours:.2f} & {n2_kmnist_othe_ps:.2f} \\ \hline
    \multicolumn{{2}}{{|c|}}{{平均}}                    & {avg_mnist_ns:.2f}     & {avg_mnist_ours:.2f}     & {avg_mnist_ps:.2f}     & {avg_kmnist_ns:.2f}     & {avg_kmnist_ours:.2f}     & {n0_kmnist_ps:.2f}      \\ \hline
    \end{{tabular}}
\end{{table}}
"""

def format_table(dataset, method, result, format_func):
    assert method in ["ns","ours","ps"]
    format_dict = {}
    for node in result["eval_results"]:
        format_dict[f"n{node}_{dataset}_self_{method}"] = result["eval_results"][node]["self_acc"]
        format_dict[f"n{node}_{dataset}_othe_{method}"] = result["eval_results"][node]["other_acc"]
    format_dict[f"avg_{dataset}_{method}"] = sum(format_dict.values()) / 6.
    return partial(format_func, **format_dict)

# 
# s = "{foo} {bar} {zee}".format 
# s_foo = partial(s, **{
#     "foo":"FOO"
# }) 
# s_bar = partial(s_foo, **{
#     "bar":"BAR"
# })
# s_bar()
# print(s_bar(**{
#     "zee":"ZEE"
# })) # FOO BAR print(s(foo="FOO", bar="BAR")) # FOO BAR
plot_loss("default")