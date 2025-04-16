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
        device: 'cpu',
        inputs: list,
):
    # TODO: Documentation

    model.eval()
    baseline_metrics = ["OBS_B", "CONT", "CONT_k2"]
    n_instances = len(data)
    n_metrics = len(baseline_metrics)

    aucs = np.ndarray(n_instances)
    base_aucs = np.ndarray((n_instances, n_metrics))

    for idx in range(n_instances):
        ins = data[idx].to(device)
        # print(f"Evaluation instance {idx}: {ins}")

        x_tensor = ins.x.cpu().detach().numpy()
        observed_nodes = x_tensor.T[0]
        truth = ins.y.cpu().detach().numpy()
        evaluation_truth = truth[observed_nodes == 0]

        # Evaluation of the GNN model
        prediction = model(ins).cpu().detach().numpy()  # list with the probabilities of nodes being infected
        evaluation_prediction = prediction[observed_nodes == 0]
        auc = roc_auc_score(evaluation_truth, evaluation_prediction)
        aucs[idx] = auc

        # Evalutation of the betweenness metric
        for mIdx, m in enumerate(baseline_metrics):
            metric = get_input_value(ins, m, inputs)
            evaluation_metric = metric[observed_nodes == 0]
            base_aucs[idx, mIdx] = roc_auc_score(evaluation_truth, evaluation_metric)

    # print(base_aucs)

    gnn_stats = {
        'number of instances': int(n_instances),
        'mean': float(np.mean(aucs)),
        'std': float(np.std(aucs)),
        'median': float(np.percentile(aucs, 50)),
        '1st quartile': float(np.percentile(aucs, 25)),
        '3rd quartile': float(np.percentile(aucs, 75)),
        'max': float(np.nanmax(aucs)),
        'min': float(np.nanmin(aucs))
    }

    statistics = {
        "GNN": gnn_stats,
    }

    for mIdx in range(n_metrics):
        base_stats = {
            'number of instances': int(n_instances),
            'mean': float(np.mean(base_aucs[:, mIdx])),
            'std': float(np.std(base_aucs[:, mIdx])),
            'median': float(np.percentile(base_aucs[:, mIdx], 50)),
            '1st quartile': float(np.percentile(base_aucs[:, mIdx], 25)),
            '3rd quartile': float(np.percentile(base_aucs[:, mIdx], 75)),
            'max': float(np.nanmax(base_aucs[:, mIdx])),
            'min': float(np.nanmin(base_aucs[:, mIdx]))
        }
        statistics[baseline_metrics[mIdx]] = base_stats

    return statistics


def get_input_value(datapoint, input_name, inputs_list):
    """
        Function to extract a metric value from the data.

    :param datapoint: Data instance
    :param input_name: Name of the input to be read.
    :param inputs_list: List of data set main metrics
    :return: Tensor of the metric's values for all nodes in the network of the data instance.
    """

    obs_b_pos = None if not (input_name in inputs_list) else inputs_list.index(input_name)

    if obs_b_pos:
        return datapoint.x[:, obs_b_pos]
    else:
        if input_name == "OBS_B":
            return datapoint.obs_b
        elif input_name == "CONT":
            return datapoint.cont
        elif input_name == "CONT_k2":
            return datapoint.cont_k2
        else:
            raise KeyError(f"Value {input_name} not a metric available on the data set. "
                           f"Please choose a valid input.")


# TODO: ROC values / plot ROC
#   Maybe our own ROC curves, so that we can compute the average ROC???
