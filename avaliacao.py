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
    n_instances = len(data)

    aucs = np.ndarray(n_instances)
    bet_aucs = np.ndarray(n_instances)

    gnn_stats = {}
    betweeness_stats = {}

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
        metric = get_observed_betweenness(ins, inputs) 
        evaluation_metric = metric[observed_nodes == 0].cpu().detach().numpy()
        bet_aucs[idx] = roc_auc_score(evaluation_truth, evaluation_metric)

    gnn_stats['number of instances'] = float(n_instances)
    gnn_stats['mean'] = float(np.mean(aucs))
    gnn_stats['std'] = float(np.std(aucs))
    gnn_stats['median'] = float(np.percentile(aucs, 50))
    gnn_stats['1st quartile'] = float(np.percentile(aucs, 25))
    gnn_stats['3rd quartile'] = float(np.percentile(aucs, 75))
    gnn_stats['max'] = float(np.nanmax(aucs))
    gnn_stats['min'] = float(np.nanmin(aucs))

    betweeness_stats['number of instances'] = float(n_instances)
    betweeness_stats['mean'] = float(np.mean(bet_aucs))
    betweeness_stats['std'] = float(np.std(bet_aucs))
    betweeness_stats['median'] = float(np.percentile(bet_aucs, 50))
    betweeness_stats['1st quartile'] = float(np.percentile(bet_aucs, 25))
    betweeness_stats['3rd quartile'] = float(np.percentile(bet_aucs, 75))
    betweeness_stats['max'] = float(np.nanmax(bet_aucs))
    betweeness_stats['min'] = float(np.nanmin(bet_aucs))

    statistics = {
        "GNN": gnn_stats,
        "OBS_B": betweeness_stats
    }

    return statistics


def get_observed_betweenness(datapoint, inputs):
    # TODO: Documentation
    # Função para extrair o observed betweenness dos dados

    obs_b_pos = None if not ("OBS_B" in inputs) else inputs.index("OBS_B")

    if obs_b_pos:
        return datapoint.x[:, obs_b_pos]
    else:
        return datapoint.obs_b


# TODO: ROC values / plot ROC
#   Maybe our own ROC curves, so that we can compute the average ROC???
