"""
    File containing the project's evaluation and visualization functions
"""

from formatacao import EpidemicDataset
from models import GCN

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auc_statistics(
        data: EpidemicDataset,
        model: GCN,
):
    # TODO: Documentation
    n_instances = len(data)
    aucs = np.ndarray(n_instances)

    statistics = {}

    for idx in range(n_instances):
        ins = data[idx]
        truth = ins.y.detach().numpy()
        # print(f"Evaluation instance {idx}: {ins}")

        prediction = model(ins).detach().numpy()     # TODO: Confirmar isto...

        auc = roc_auc_score(truth, prediction)   # prediction must be a list with the corresponding probability of being infected for each node
        aucs[idx] = auc

        # TODO: get AUC from output and ground truth and add to the aucs array
        # statistics[idx] = auc

    # TODO: compute AUC statistics and return as a dictionary
    #   Quero por o número de amostras como info do dicionário?
    statistics['number of instances'] = float(n_instances)
    statistics['mean'] = float(np.mean(aucs))
    statistics['std'] = float(np.std(aucs))
    statistics['median'] = float(np.percentile(aucs, 50))
    statistics['1st quartile'] = float(np.percentile(aucs, 25))
    statistics['3rd quartile'] = float(np.percentile(aucs, 75))
    statistics['max'] = float(np.nanmax(aucs))
    statistics['min'] = float(np.nanmin(aucs))

    return statistics

# TODO: ROC values / plot ROC
#   Maybe our own ROC curves, so that we can compute the average ROC???

