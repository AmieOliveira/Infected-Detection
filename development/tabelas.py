import pandas as pd
import numpy as np


# Getting data files
mean_file = "../../Dados/performance/summary_stats_latex.csv"
df_auc = pd.read_csv(mean_file, index_col=0)

mean_file = "../../Dados/performance/summary_stats_top1_latex.csv"
df_top1 = pd.read_csv(mean_file, index_col=0)

std_file = "../../Dados/performance/summary_stats_top5_latex.csv"
df_top5 = pd.read_csv(std_file, index_col=0)


# Refactor column and line names
base_model_name = "GNN {}, $\\theta = {}$"
base_data_name = "{} {}k, $\\theta = {}$"
obs_bet_name = "Obs. Betweenness"

col_map = {}
line_map = {}
for o in [10, 25, 50, 75, 90]:
    for n in [1, 3, 6, 12]:
        col_map[f"BA_teste_{n}knos_pObs{o}"] = base_data_name.format("BA", n, o/100)
        col_map[f"WS_teste_n{n}k_obs{o}"] = base_data_name.format("WS", n, o/100)

    for net in ["BA", "WS"]:
        line_map[f"{net}_obs_{o}"] = base_model_name.format(net, o/100)
line_map["OBS_B"] = obs_bet_name
df_auc = df_auc.rename(index=line_map, columns=col_map)
df_top1 = df_top1.rename(index=line_map, columns=col_map)
df_top5 = df_top5.rename(index=line_map, columns=col_map)


# Fazer tabelas
nets = ["WS"]  # "BA"
obs_train_values = [0.9]
obs_values = [0.1, 0.5, 0.9]
n_values = [1, 3, 6, 12]

lines = [base_model_name.format(net, o) for o in obs_train_values for net in nets] + [obs_bet_name]
columns = [base_data_name.format(net, n, o) for o in obs_values for n in n_values for net in nets]

print("AUC results")
print(df_auc.loc[lines, columns].T.to_latex())

print("Top-1% results")
print(df_top1.loc[lines, columns].T.to_latex())

print("Top-5% results")
print(df_top5.loc[lines, columns].T.to_latex())
