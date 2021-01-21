import argparse
import psutil
import signal
import time
import json
from json import JSONEncoder

class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

parser = argparse.ArgumentParser()
parser.add_argument('--interface', type=str, nargs='+', required=True)
parser.add_argument('--interval', type=float, default=1.)
parser.add_argument('--name', type=str, default="default")
args = parser.parse_args()

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

class Stats:
    def __init__(self, interval) -> None:
        self.send = []
        self.send_speed = []
        self.receive = []
        self.receive_speed = []
        self.interval = interval

    def update(self, send, receive):
        if len(self.send) != 0:
            self.send_speed.append((send-self.send[-1]) / self.interval)
            self.receive_speed.append((receive-self.receive[-1]) / self.interval)
        self.send.append(send)
        self.receive.append(receive)

killer = GracefulKiller()
s = {}
for interface in args.interface:
    s[interface] = Stats(args.interval)

while not killer.kill_now:
    time.sleep(args.interval)
    stats = psutil.net_io_counters(pernic=True)
    for i in stats:
        if i in args.interface:
            s[i].update(stats[i].bytes_sent, stats[i].bytes_recv)

with open(f"network/{args.name}.json", "w") as f:
    json.dump(s, f, cls=MyEncoder)