"""
    File containing the project's evaluation and visualization functions
"""

from formatacao import EpidemicDataset
from models import GCN

import numpy as np

def auc_statistics(
        data: EpidemicDataset,
        model: GCN,
):
    # TODO: Documentation
    n_instances = len(data)
    aucs = np.ndarray(n_instances)

    for idx in range(n_instances):
        ins = data[idx]
        truth = ins.y
        print(f"Evaluation instance {idx}: {ins}")

        prediction = model(ins)     # TODO: Confirmar isto...

        # TODO: get AUC from output and ground truth and add to the aucs array

    # TODO: compute AUC statistics and return as a dictionary
    #   Quero por o número de amostras como info do dicionário?


# TODO: ROC values / plot ROC
#   Maybe our own ROC curves, so that we can compute the average ROC???

