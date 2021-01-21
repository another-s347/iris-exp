import json
import matplotlib.pyplot as plt

def plot_loss(suffix):
    f = open(f"./results/{suffix}/loss.json","r")
    d = json.load(f)

    for result in d['results']:
        y = d['results'][result]
        x = [i for i in range(len(y))]
        plt.plot(x, y, label=f"node-{result}")
    plt.legend()
    plt.savefig("loss.pdf")

