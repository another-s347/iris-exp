import json
import matplotlib.pyplot as plt

def plot_loss(name, intf):
    f = open(f"./network/{name}.json","r")
    d = json.load(f)

    for interface in d:
        if interface != intf:
            continue
        y = d[interface]['receive_speed']
        y = list(map(lambda i: i/1024./1024., y))
        x = [i for i in range(len(y))]
        plt.plot(x, y, label=f"receive")
        y = d[interface]['send_speed']
        y = list(map(lambda i: i/1024./1024., y))
        x = [i for i in range(len(y))]
        plt.plot(x, y, label=f"send")
    plt.legend()
    plt.savefig(f"{name}-{intf}.pdf")

plot_loss("10.0.0.4","ed1-eth0")