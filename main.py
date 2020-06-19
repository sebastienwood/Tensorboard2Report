import os
import io
import yaml
import warnings
import argparse
import numpy as np
import pandas as pd

from enum import Enum
from typing import Union, List

# Global utils
def argmax(listT:List[Union[int, float]]):
    return listT.index(max(listT))

def arglast(listT:List[Union[int, float]]):
    return len(listT) - 1

class Reduction(Enum):
    MAX = argmax
    LAST = arglast

# Tensorboard utils
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def get_scalars(scalars: List[str], exp_path:str):
    """ Adapted from https://github.com/Spenhouet/tensorboard-aggregator/blob/master/aggregator.py """
    scalar_accumulators = [EventAccumulator(exp_path).Reload().scalars for dname in os.listdir(exp_path)]
    scalar_accumulators = [scalar_accumulator for scalar_accumulator in scalar_accumulators if scalar_accumulator.Keys()]

     # Get and validate all scalar keys
    all_keys = [tuple(scalar_accumulator.Keys()) for scalar_accumulator in scalar_accumulators]
    if len(all_keys) == 0: return None
    assert len(set(all_keys)) == 1, "All runs need to have the same scalar keys. There are mismatches in {}".format(all_keys)
    keys = all_keys[0]

    all_scalar_events_per_key = [[scalar_accumulator.Items(key) for scalar_accumulator in scalar_accumulators] for key in scalars]

    values_per_key = [[[scalar_event.value for scalar_event in scalar_events] for scalar_events in all_scalar_events]
                      for all_scalar_events in all_scalar_events_per_key]

    all_per_key = dict(zip(scalars, values_per_key))
    return all_per_key

# Filters classes
from abc import ABC, abstractmethod

class Filter(ABC):
    def __init__(self, key, value):
        self.key = key
        self.value = value
        super().__init__()
    
    @abstractmethod
    def check(self):
        pass

    def __repr__(self):
        return f'{type(self)} with key {self.key} and value {self.value}'

    def explain(self):
        print(f"--> The key {self.key} value is not {self.value} in this experiment.")

class AtomicFilter(Filter):
    def check(self, other:Union[str, int, float]):
        if not self.value == other:
            self.explain()
            return False
        return True

class RangeFilter(Filter):
    def check(self, other:Union[int, float]):
        raise NotImplementedError("The RangeFilter has not yet been implemented.")

class ORFilterBank(Filter):
    def check(self, other:Union[int, float, str]):
        raise NotImplementedError("The ORFilter has not yet been implemented.")

class ANDFilterBank(Filter):
    def check(self, other:Union[int, float, str]):
        raise NotImplementedError("The ANDFilter has not yet been implemented.")

def report(vars_of_interest: List[str],
            experiment_key_metric: str,
            groupby: str,
            experiment_filters: Union[List[Filter], Filter],
            log_dir:str='./',
            reduction_strategy: Reduction = Reduction.LAST):
    r"""
    Produce a .csv report with all the `vars_of_interest` grouped by `groupby`

    Args:
        vars_of_interest: list of strings with the name of variables as they are logged in Tensorboard

        experiment_key_metric: a string with the name of variable which the report is about (e.g. top1 accuracy). Is used to reduce the data.

        groupby: by which hyperparameter are the experiments grouped

        experiment_filters: list or single instance of Filter. Experiments whose hyperparameters do not comply with these filters won't be kept in the report.

        reduction_strategy: how to reduce the data to a single datapoint. Member of Reduction enum.

        log_dir: where are the tensorboard logs stored

    """
    # Ensure the experiment key metric is not in vars of interest 
    if experiment_key_metric in vars_of_interest: vars_of_interest.remove(experiment_key_metric)

    # Get a list of all experiments, prepare list of results
    experiments = [f.path for f in os.scandir(log_dir) if f.is_dir()]
    results = []

    for exp in experiments:
        print(f"-> Processing experiment {exp}")
        # Does the experiment folder adhere to lightning convention ?
        hparams_path = f'{exp}/hparams.yaml'
        if not os.path.isfile(hparams_path): 
            warnings.warn(f"The experiment {exp} does not have an hparams.yaml file. Skipping it.")
            continue

        # Parse hparams.yaml, check if `experiment_filter` are verified
        hparams = yaml.load(io.open(hparams_path, 'r'), Loader=yaml.FullLoader)
        assert groupby in hparams, f"{groupby} is not an existing hparam"
        if is_experiment_filtered(hparams, experiment_filters):
            print(f"---> The experiment {exp} is thus skipped.")
            continue
        
        # Parse tensorboard events and get the `vars_of_interest`+`hparams` into results as a dict
        res = get_scalars(vars_of_interest+[experiment_key_metric], exp)
        if res is not None:
            # Now reduce to a value per experiment
            idx_tar = apply_reduction_strategy(reduction_strategy, res[experiment_key_metric][0]) #TODO getting the 0th idx is not clean
            res_dict = {
                f'{var}_{groupby}{hparams[groupby]}': [res[var][0][idx_tar]] for var in vars_of_interest+[experiment_key_metric]
            }
            print(res_dict)
            results.append(pd.DataFrame(res_dict))
        else: 
            print(f"---> The experiment was empty :c")
    
    # Produce .csv
    #TODO are the NaN from concat an issue ?
    pd.concat(results, ignore_index=True).to_csv(f'./report_per_{groupby}.csv')

def is_experiment_filtered(hparams:dict,
                            experiment_filters:List[Filter]) -> bool:
    for filt in experiment_filters:
        # Does the experiment has the filter key ?
        if not filt.key in hparams: 
            warnings.warn(f"--> The hparams.yaml file does not have this hyperparameter {filt.key}.")
            return True
        if not filt.check(hparams[filt.key]):
            return True
    return False

def apply_reduction_strategy(red:Reduction, key_metric:list):
    return red(key_metric)

def parse_filter(path:str) -> List[Filter]:
    r"""
        Parse a .yaml file comprising dict for each filter with key/value entries
    """
    filts = []
    filts_raw = yaml.load(io.open(path, 'r'), Loader=yaml.FullLoader)
    for filterDict in filts_raw:
        nf = [globals()[filterDict](k, v) for k, v in filts_raw[filterDict].items()]
        filts.extend(nf)
    return filts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filters', type=str, help='Path to your yaml.config file', default='example_filters.yaml')
    parser.add_argument('--logdir', type=str, help='Path to your tensorboard log dir', default='/export/tmp/henwood/lightning_logs')
    parser.add_argument('--groupby', type=str, help='Hparams to group experiments', default='alpha')
    parser.add_argument('--target', type=str, help='Experiment key metric, will be used to reduce the exp. to a single value', default="Acc/val_acc1")
    parser.add_argument('--reduction', type=Reduction, choices=list(Reduction), help='How is reduced the experiment to a single value', default=Reduction.LAST)
    parser.add_argument('--metrics', nargs='+', help='All the variables to include in the report', default=['Cons/act_cons'])
    args = parser.parse_args()

    filts = parse_filter(args.filters)
    report(vars_of_interest=args.metrics, 
            experiment_key_metric=args.target, 
            groupby=args.groupby, 
            experiment_filters=filts, 
            reduction_strategy=args.reduction,
            log_dir=args.logdir)