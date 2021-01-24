from mininet.log import setLogLevel, info
# from mn_wifi.link import wmediumd, adhoc
# from mn_wifi.cli import CLI_wifi
from mininet.cli import CLI
from mininet.net import Mininet
# from mininet.link import TCLink
# from mn_wifi.topo import Topo_WiFi
# from mn_wifi.link import wmediumd, adhoc
# from mn_wifi.wmediumdConnector import interference
from mininet.node import RemoteController
from mininet.link import TCIntf, TCLink
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", action="store_true", default=False, help="enable iris logging")
parser.add_argument("--nodes", type=int, default=3)
parser.add_argument("--mode", type=str, default="release", choices=["release","debug"])
args = parser.parse_args()

def topology():
    "Create a network."
    net = Mininet()
    # ctrl = net.addController('c0', controller=RemoteController, ip="172.17.0.2", port=6653)
    bw = 100

    info("*** Creating nodes\n")
    cns = [net.addHost(f"cn{x+1}",ip=f"10.0.0.{x+1}/24") for x in range(args.nodes)]

    ed1 = net.addHost("ed1",ip="10.0.0.14/24")

    # cloud1 = net.addHost("cloud",ip="10.0.1.2/24")
    cloud1 = net.addHost("cloud",ip="10.0.0.15/24")

    nodes = [*cns, ed1, cloud1]

    info("*** Creating switch")
    fw1 = net.addSwitch("fw1",protocols=["OpenFlow14"],failMode="standalone")
    fw2 = net.addSwitch("fw2",protocols=["OpenFlow14"],failMode="standalone")
    # fw3 = net.addSwitch("fw3",protocols=["OpenFlow14"],failMode="standalone")

    info("*** Creating links\n")
    for cn in cns:
        net.addLink(fw1, cn)
    
    net.addLink(fw1, ed1)
    # net.addLink(fw2, ed1,params2={ 'ip' : '10.0.1.1/24' })

    # net.addLink(fw3, cloud1,intf=TCIntf)
    net.addLink(fw1, cloud1,intf=TCIntf)
    # net.addLink(fw1, cn1, cls=TCLink, delay='5ms',use_htb=True)
    # net.addLink(fw1, cn2, cls=TCLink, delay='5ms',use_htb=True)

    info("*** Starting network\n")
    net.build()
    fw1.start([])
    fw2.start([])
    # fw3.start([])

    info("*** Start iris server\n")
    # logs = []
    ps = []
    for n in nodes:
        log = open(f'{n.IP()}.log', 'w')
        p1 = n.popen([f"/home/skye/iris/iris-client/target/{args.mode}/server","-l",n.IP(),"-p","12345"],stdout=log,stderr=log)
        ps.append(p1)
        intfs = list(n.nameToIntf.keys())
        p2 = n.popen(["python","/home/skye/iris-exp/main_exp/track_network.py","--interface",*intfs,"--name",n.IP()])
        ps.append(p2)

    info("*** Running CLI\n")
    CLI(net)

    for p in ps:
        p.terminate()

    info("*** Stopping network\n")
    net.stop()


if __name__ == '__main__':
    if args.nodes <= 10:
        setLogLevel('debug')
        topology()
