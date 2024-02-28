import torch

def accuracy(tp, tn, all):
    acc = ((tp + tn) / all)
    return torch.round(acc * 100, decimals=2)

def precision(tp, fp):
    pre = (tp / (tp + fp))
    return torch.round(pre * 100, decimals=2)

def recall(tp, fn):
    sn = (tp / (tp + fn))
    return torch.round(sn * 100, decimals=2)

def specificity(tn, fp):
    sn = (tn / (tn + fp))
    return torch.round(sn * 100, decimals=2)

def f1_score(precision, recall):
    f1 = (2 *((precision * recall) / (precision + recall)))
    return torch.round(f1, decimals=2)
