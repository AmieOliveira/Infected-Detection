"""
    File containing the project's evaluation and visualization functions
"""

from formatacao import EpidemicDataset
from models import GCN

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score


def auc_statistics(
        data: EpidemicDataset,
        model: GCN,
        inputs: list,
        device='cpu',
):
    # TODO: Documentation

    model.eval()
    n_instances = len(data)

    aucs = np.ndarray(n_instances)
    bet_aucs = np.ndarray(n_instances)

    gnn_stats = {}
    betweeness_stats = {}

    runtimeerrors = 0
    successes = 0

    for idx in range(n_instances):
        try:
            ins = data[idx].to(device)
        except RuntimeError as e:
            print("Caught runtime erorr while trying to get instance -- skipping it")
            print(f"\t{e}\n")
            runtimeerrors += 1
            continue
        # print(f"Evaluation instance {idx}: {ins}")

        x_tensor = ins.x.cpu().detach().numpy()
        observed_nodes = x_tensor.T[0]
        truth = ins.y.cpu().detach().numpy()
        evaluation_truth = truth[observed_nodes == 0]

        # Evaluation of the GNN model
        try:
            prediction = model(ins)
        except RuntimeError as e:
            print("Caught exception trying to infer data -- will be skipping it!")
            print(f"\t{e}\n")
            runtimeerrors += 1
            continue
        prediction = prediction.cpu().detach().numpy()  # list with the probabilities of nodes being infected
        evaluation_prediction = prediction[observed_nodes == 0]
        auc = roc_auc_score(evaluation_truth, evaluation_prediction)
        aucs[idx] = auc

        # Evalutation of the betweenness metric
        metric = get_observed_betweenness(ins, inputs)  #.cpu().detach().numpy()
        evaluation_metric = metric[observed_nodes == 0].cpu().detach().numpy()
        bet_aucs[idx] = roc_auc_score(evaluation_truth, evaluation_metric)

        successes += 1

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

    print(
        f"Finished evaluation. "
        f"RuntimeErrors caught: {runtimeerrors}. "
        f"Evaluated {successes} samples overall."
    )

    return statistics


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


def topk_statistics(
        data: EpidemicDataset,
        model: GCN,
        inputs: list,
        device='cpu',
        k_vals=(0.01, 0.05),
):
    # TODO: Documentation

    model.eval()
    n_instances = len(data)
    n_nodes = data[0].num_nodes
    n_ks = len(k_vals)

    topk = np.ndarray((n_instances, n_ks))
    bet_topk = np.ndarray((n_instances, n_ks))

    gnn_stats = {}
    betweeness_stats = {}
    for k in k_vals:
        gnn_stats[f"top-{k*100}%"] = {}
        betweeness_stats[f"top-{k*100}%"] = {}

    runtimeerrors = 0
    successes = 0

    for idx in range(n_instances):
        try:
            ins = data[idx].to(device)
        except RuntimeError as e:
            print("Caught runtime erorr while trying to get instance -- skipping it")
            print(f"\t{e}\n")
            runtimeerrors += 1
            continue
        # print(f"Evaluation instance {idx}: {ins}")

        x_tensor = ins.x.cpu().detach().numpy()
        observed_nodes = x_tensor.T[0]
        truth = ins.y.cpu().detach().numpy()
        evaluation_truth = truth[observed_nodes == 0]

        # Evaluation of the GNN model
        try:
            prediction = model(ins)
        except RuntimeError as e:
            print("Caught exception trying to infer data -- will be skipping it!")
            print(f"\t{e}\n")
            runtimeerrors += 1
            continue
        prediction = prediction.cpu().detach().numpy()  # list with the probabilities of nodes being infected
        evaluation_prediction = prediction[observed_nodes == 0]
        
        # TODO: Finish implementation


def top_k_score(evaluation_truth, evaluation_prediction, top_k):
    top_indices = np.argsort(evaluation_prediction)[-top_k:]
    y_true_top = evaluation_truth[top_indices]
    y_pred_top = np.ones_like(y_true_top)

    precision = precision_score(y_true_top, y_pred_top)
    recall = recall_score(evaluation_truth, np.isin(np.arange(len(evaluation_truth)), top_indices).astype(int))
    return precision, recall

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
