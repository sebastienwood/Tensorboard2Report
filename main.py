import os
import io
import yaml
import shutil
import tarfile
import warnings
import argparse
import numpy as np
import pandas as pd

from enum import Enum
from pathlib import Path
from functools import partial
from typing import Union, List, Callable

# CONSTANTS
ARCHIVES_FMT = ['.zip', '.tar', '.gz']

# Global utils
def argmax(listT:List[Union[int, float]]) -> int:
    return listT.index(max(listT))

def arglast(listT:List[Union[int, float]]) -> int:
    return lastk(listT, 1)

def topk(listT:List[Union[int, float]], k:int=5) -> List[int]:
    return sorted(range(len(listT)), key=lambda i: listT[i])[-k:]
    
def lastk(listT:List[Union[int, float]], k:int=5) -> List[int]:
    return len(listT) - k

def bestoflastk(listT:List[Union[int, float]], k:int=25) -> int:
    listT = listT[-k:]
    return listT.index(max(listT))

def kbestoflastl(listT:List[Union[int, float]]) -> List[int]:
    pass #TODO generalized version of everything

reductions = ['argmax', 'arglast', 'topk', 'lastk']

def fast_scandir(dirname):
    subfolders= [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(fast_scandir(dirname))
    return subfolders

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
    r""" Assert values are in a range (low <= value <= high)
    Args:
        value: a list of size 2 with index 0 the lower bound and 1 the higher bound of the range
    """
    def __init__(self, key, value):
        assert isinstance(value, list) and len(value) == 2 and value[1] > value[0]
        super().__init__(key, value)

    def check(self, other:Union[int, float]):
        if not self.value[1] >= other or not self.value[0] <= other:
            self.explain()
            return False
        return True

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
            reduction_strategy: Callable[[List[Union[int, float]], int], Union[int, List[int]]],
            log_dir:str='./',
            reduction_kwargs=None):
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
    experiments = fast_scandir(log_dir)#[f.path for f in os.scandir(log_dir) if f.is_dir()]
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
            idx_tar = apply_reduction_strategy(reduction_strategy, res[experiment_key_metric][0], reduction_kwargs) #TODO getting the 0th idx is not clean
            res_dict = {
                f'{var}_{groupby}{hparams[groupby]}': [res[var][0][idx_tar]] for var in vars_of_interest+[experiment_key_metric]
            }
            results.append(pd.DataFrame(res_dict))
        else: 
            print(f"---> The experiment was empty :c")
    
    # Produce .csv
    #TODO are the NaN from concat an issue ?
    with open(f'./report_per_{groupby}.csv', 'w') as f:
        for filt in experiment_filters: f.write(f'# {filt}\n')
    pd.concat(results, ignore_index=True).to_csv(f'./report_per_{groupby}.csv', mode='a', index=False)

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

def apply_reduction_strategy(red:str, key_metric:list, kwargs=None):
    return globals()[red](key_metric) if kwargs is None else globals()[red](key_metric, kwargs)

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

def main(args):
    r"""
        Main process
    """
    F_archive = False

    p = Path(args.logdir)
    if p.suffix in ARCHIVES_FMT:
        shutil.unpack_archive(p, p.parent/ 'temp')
        F_archive = True
        args.logdir = p.parent / 'temp'
        print(f'Successfully unpacked the archive at {args.logdir}')

    filts = parse_filter(args.filters)
    report(vars_of_interest=args.metrics, 
            experiment_key_metric=args.target, 
            groupby=args.groupby, 
            experiment_filters=filts, 
            reduction_strategy=args.reduction,
            reduction_kwargs=args.reduction_kwargs,
            log_dir=args.logdir)

    if F_archive:
        shutil.rmtree(args.logdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filters', type=str, help='Path to your yaml.config file', default='example_filters.yaml')
    parser.add_argument('--logdir', type=str, help='Path to your tensorboard log dir (can be an archive file)', default='/export/tmp/henwood/lightning_logs')
    parser.add_argument('--groupby', type=str, help='Hparams to group experiments', default='alpha')
    parser.add_argument('--target', type=str, help='Experiment key metric, will be used to reduce the exp. to a single value', default="Acc/val_acc1")
    parser.add_argument('--reduction', choices=reductions, type=str, help='How is reduced the experiment to a single value', default='arglast')
    parser.add_argument('--reduction-kwargs', help='Extra parameter passed to Reduction')
    parser.add_argument('--metrics', nargs='+', help='All the variables to include in the report', default=['Cons/act_cons', 'Cons/cls_weight_cons'])
    args = parser.parse_args()

    main(args)
    # filts = parse_filter(args.filters)
    # report(vars_of_interest=args.metrics, 
    #         experiment_key_metric=args.target, 
    #         groupby=args.groupby, 
    #         experiment_filters=filts, 
    #         reduction_strategy=args.reduction,
    #         reduction_kwargs=args.reduction_kwargs,
    #         log_dir=args.logdir)

    # /export/tmp/henwood/archive_logs/faulty_weights_exp.tar.gz