import torch
import torch.nn as nn
import numpy as np

from typing import List


def get_important_regions(expl: np.ndarray, set_to_one: bool = True):
    m = expl.mean()
    temp_idx = expl < m
    expl[temp_idx] = 0
    if set_to_one:
        expl[~temp_idx] = 1
    return expl


def compare_explanations_recall(expl: np.ndarray, target: np.ndarray):
    print("expl", expl)
    TP = expl * target
    max = target.sum()
    print("TP", TP)
    recall = TP.sum()/max
    print("recall", recall)
    return recall


def compare_explanations_precision(expl: np.ndarray, target: np.ndarray):
    TP = expl * target
    max = expl.sum()
    precision = TP.sum()/max
    print("precision:", precision)
    return precision


def get_consensus(targets: List[np.ndarray],
                  boolean_choice: bool = True):
    consensus = sum(targets) / len(targets)
    if boolean_choice:
        temp_idx = consensus >= 0.5
        consensus[temp_idx] = 1
        consensus[~temp_idx] = 0
    return np.asarray(consensus)