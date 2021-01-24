import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline

def plot_loss(name, intf):
    f = open(f"./network/{name}.json","r")
    d = json.load(f)

    for interface in d:
        if interface != intf:
            continue
        y = d[interface]['avg_recv_speed']
        y = list(map(lambda i: i/1024./1024., y))
        x = [i for i in range(len(y))]
        spl = make_interp_spline(x, y)
        xs = np.linspace(0, x[-1], 300)
        power_smooth = spl(xs)
        # plt.plot(x, y)
        plt.plot(xs, power_smooth, label=f"receive")
        y = d[interface]['avg_send_speed']
        y = list(map(lambda i: i/1024./1024., y))
        x = [i for i in range(len(y))]
        spl = make_interp_spline(x, y)
        xs = np.linspace(0, x[-1], 300)
        power_smooth = spl(xs)
        # plt.plot(x, y)
        plt.plot(xs, power_smooth, label=f"send")
    plt.legend()
    plt.savefig(f"{name}-{intf}.pdf")

plot_loss("10.0.0.14","ed1-eth0")