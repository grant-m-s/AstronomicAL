
import numpy as np
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
# from torch import nn
# import json

def adjust_learning_rate(optimizer, epoch, gammas, schedule, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    "Add by YU"
    lr = args.lr
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu


def initialise_metrics():
    metrics = {
        "Accuracy": [],
        "F1": [],
        "Loss": [],
        "Global Informativeness": [],
        "Local Informativeness": [],
        "Samples Per Tess": [],
        "Sample Entropy": [],
        "Sample Hist": [],
        "Regret - True": [],
        "Regret - Above": [],
        "Regret - Place": [],
    }

    # self.metrics["Local Informativeness"].append([])
    # for reg_idx, region in enumerate(self.regions):
    #     self.metrics["Local Informativeness"].append([])
    #     self.metrics["Samples Per Tess"].append([])

    # assert len(self.metrics["Local Informativeness"]) == len(self.regions)
    return metrics
