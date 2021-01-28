from typing import Any, List
import os
import json
from json import JSONEncoder

class MyEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__  

class RunResult:
    def __init__(self) -> None:
        self.suffix: str = ''
        self.datetime: str = ''

        self.args: Any = None
        self.client_results: dict[int,'ClientResult'] = {}
        self.eval_results: dict[int,'EvalResult'] = {}
        self.edge_results: 'EdgeResult'
        self.global_accs: float = 0.

    def write(self):
        os.makedirs(f"results/{self.suffix}/", exist_ok=True)

        with open(f"results/{self.suffix}/loss.json", "w") as f:
            j = {
                "time": self.datetime,
                "results": {},
                "sync_flags": {},
                "epoch_train_accs": {},
                "epoch_test_self_accs": {},
                "epoch_test_other_accs": {}
            }
            for i in self.client_results:
                j["results"][i] = self.client_results[i].losses
                j["sync_flags"][i] = self.client_results[i].sync_flags
                j["epoch_train_accs"][i] = self.client_results[i].epoch_train_acc
                j["epoch_test_self_accs"][i] = self.client_results[i].epoch_test_self_acc
                j["epoch_test_other_accs"][i] = self.client_results[i].epoch_test_other_acc
            json.dump(j, f)
            path = os.path.realpath(f.name)

        print(f"write loss data to {path}")
        
        try:
            with open(f"results/{self.suffix}/result.json","r") as f:
                body = f.read()
                if len(body) == 0:
                    old = {'run':[]}
                else:
                    old = json.loads(body)
                old['run'].append({
                    "args": self.args,
                    "suffix": self.suffix,
                    "datetime": self.datetime,
                    "eval_results": self.eval_results,
                    "edge_results": self.edge_results
                })
        except:
            old = {'run':[]}
    
        with open(f"results/{self.suffix}/result.json","w") as f:
            json.dump(old, f, cls=MyEncoder, indent=4)

            path = os.path.realpath(f.name)

        print(f"write result data to {path}")

class ClientResult:
    def __init__(self) -> None:
        self.losses: List[float] = []
        self.epoch_train_acc: List[float] = []
        self.epoch_test_self_acc: List[float] = []
        self.epoch_test_other_acc: List[float] = []
        self.sync_flags: List[int] = []
        self.rank: int = -1
        self.len_dataset: int = -1

class EvalResult:
    def __init__(self) -> None:
        self.self_acc: float = 0.
        self.self_correct: float = 0.
        self.other_acc: float = 0.
        self.other_correct: float = 0.
        self.rank: int = 0

class EdgeResult:
    def __init__(self) -> None:
        self.sync_count: int = 0
        self.sync_config: List[int] = []
        self.delayed_rate: float = 0.
        self.delayed_counts: List[int] = []
        self.delayed_count_total: int = 0
        self.delayed_padding:int = 0
