import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


title = False
plot_std = False

# Getting data files
mean_file = "../../Dados/performance/summary_stats_mean.csv"
df_mean = pd.read_csv(mean_file, index_col=0)

std_file = "../../Dados/performance/summary_stats_std.csv"
df_std = pd.read_csv(std_file, index_col=0)

# Refactor column and line names
base_model_name = "GNN {}, $\\theta = {}$"
base_data_name = "{} {}k, $\\theta = {}$"
obs_bet_name = "Obs. Betweenness"
y_axis_label = "AUC"    # "Average AUC"
std_str = "_std" if plot_std else ""


col_map = {}
line_map = {}
for o in [10, 25, 50, 75, 90]:
    for n in [1, 3, 6, 12]:
        col_map[f"BA_teste_{n}knos_pObs{o}"] = base_data_name.format("BA", n, o/100)
        col_map[f"WS_teste_n{n}k_obs{o}"] = base_data_name.format("WS", n, o/100)

    for net in ["BA", "WS"]:
        line_map[f"{net}_obs_{o}"] = base_model_name.format(net, o/100)
line_map["OBS_B"] = obs_bet_name
df_mean = df_mean.rename(columns=col_map)
df_mean = df_mean.rename(index=line_map)
df_std = df_std.rename(index=line_map, columns=col_map)
# print(df_mean)

nets = ["BA", "WS"]


# 1. Varying obs plots
fig_size = (5, 4)       # (6, 4) # usei pra figura 12k... (onde pus só WS)
transparency = 0.1

obs_values = [0.1, 0.25, 0.50, 0.75, 0.90]
n_vals = [1, 3, 6, 12]

for n in n_vals:
    for net in nets:
        fig = plt.figure(figsize=fig_size)

        columns = [base_data_name.format(net, n, o) for o in obs_values]
        # lines = [base_model_name.format(net, o) for o in obs_values] + [obs_bet_name]
        # df_mean.loc[lines, columns].transpose().to_latex(index=True)

        for o in obs_values:
            row = base_model_name.format(net, o)
            means = df_mean.loc[row, columns]
            stds = df_std.loc[row, columns]
            plt.plot(obs_values, means, label=row, marker="^")  # , markeredgecolor="gray"
            if std_str:
                plt.fill_between(obs_values, means + stds, means - stds, alpha=transparency)
        means = df_mean.loc[obs_bet_name, columns]
        sdts = df_std.loc[obs_bet_name, columns]
        plt.plot(obs_values, means, "--", label=obs_bet_name, marker="v")  # , markeredgecolor="gray"

        if plot_std:
            plt.fill_between(obs_values, means + stds, means - stds, alpha=transparency)

        plt.legend()
        plt.xticks(obs_values)
        plt.xlabel("Observation probability in the evaluation set ($\\theta_{\\text{eval}}$)")
        plt.ylabel(y_axis_label)
        plt.grid(alpha=0.5, linestyle="--")
        if title:
            plt.title(f"{net} {n}k nodes")
        plt.tight_layout()

        plt.savefig(f"../../Dados/performance/saidas_validacao_{net}_n{n}k_variacao_observacao{std_str}.pdf",
                    transparent=True)

plt.close("all")
print("Finished first set of images")
# --------------------------


# 2. Varying the network size
fig_size = (5, 4)  # (7, 3.5)
f_size = 5  # None
# transparency = 0.1

n_values = [1, 3, 6, 12]
n_labels = [f"${n}\\, \\text{{k}}$" for n in n_values]
obs_values = [0.1, 0.25, 0.50, 0.75, 0.90]
# colors = plt.cm.get_cmap("tab20")(np.linspace(0, 1, 20))
colors_models = plt.get_cmap("YlOrRd")(np.linspace(0.3, 0.95, 5))
colors_obsbet = plt.get_cmap("GnBu")(np.linspace(0.3, 0.95, 5))

# "GNN {}, $\\theta = {}$"
label_model_name = "GNN, $\\theta = {}$"

for net in nets:
    fig = plt.figure(figsize=fig_size)

    for cIdx, o in enumerate(obs_values):
        columns = [base_data_name.format(net, n, o) for n in n_values]

        r_model = base_model_name.format(net, o)
        m_model = df_mean.loc[r_model, columns]
        s_model = df_std.loc[r_model, columns]
        label_name = label_model_name.format(o)  # r_model
        plt.plot(n_values, m_model, label=label_name, color=colors_models[cIdx], marker="^")
        if plot_std:
            plt.fill_between(n_values, m_model + s_model, m_model - s_model, alpha=transparency,
                             color=colors_models[cIdx], linewidth=0)

    for cIdx, o in enumerate(obs_values):
        columns = [base_data_name.format(net, n, o) for n in n_values]

        m_o_bet = df_mean.loc[obs_bet_name, columns]
        s_o_bet = df_std.loc[obs_bet_name, columns]
        plt.plot(n_values, m_o_bet, "--", label=f"Obs. Bet. $\\theta={o}$", color=colors_obsbet[cIdx], marker="v")  # , markeredgecolor="gray"
        # f"Obs. Bet. $\\theta_\\text{{eval}}={o}$"
        if plot_std:
            plt.fill_between(n_values, m_o_bet + s_o_bet, m_o_bet - s_o_bet,
                             alpha=transparency, color=colors_obsbet[cIdx], linewidth=0)

    # # plt.legend(ncols=1, bbox_to_anchor=(1.3, 0.5), loc='center')  # Com fig_size = (7, 3.5)
    # plt.legend(ncols=2, bbox_to_anchor=(0.5, 1.1), loc='lower center', fontsize=f_size)
    plt.xticks(n_values, labels=n_labels)
    plt.xlabel("Number of nodes in the evaluation set ($n_\\text{eval}$)")
    plt.ylabel(y_axis_label)
    plt.grid(alpha=0.5, linestyle="--")
    if title:
        plt.title(f"{net} data")
    plt.tight_layout()

    plt.savefig(f"../../Dados/performance/saidas_validacao_{net}_variacao_n_multiplos_obs{std_str}.pdf",
                transparent=True)

plt.close("all")
print("Finished second set of images")
# ----------------------------------


# 3. Modelo treinado em uma rede, e testado das duas redes (no mesmo gráfico)
fig_size = (5, 4)  # (8, 6)
f_size = 5  # None
# transparency = 0.1
n_values = [1, 3, 6, 12]
n_labels = [f"${n}\\, \\text{{k}}$" for n in n_values]
obs_values = [0.1, 0.25, 0.50, 0.75, 0.90]

marcadores = {"BA": "o", "WS": "x", obs_bet_name: "s"}
l_style = {"BA": "--", "WS": "-"}

# 3.1 Eixo X é o número de nós
for o in obs_values:
    plt.figure(figsize=fig_size)

    for net in nets:
        r_model = base_model_name.format(net, o)
        for net_option in nets:
            columns = [base_data_name.format(net_option, n, o) for n in n_values]

            m_model = df_mean.loc[r_model, columns]
            s_model = df_std.loc[r_model, columns]
            plt.plot(n_values, m_model, l_style[net_option], label=f"GNN trained in {net}, tested in {net_option}",
                     marker=marcadores[net])
            if plot_std:
                plt.fill_between(n_values, m_model + s_model, m_model - s_model, alpha=transparency)

    for net in nets:
        columns = [base_data_name.format(net, n, o) for n in n_values]
        m_o_bet = df_mean.loc[obs_bet_name, columns]
        s_o_bet = df_std.loc[obs_bet_name, columns]
        plt.plot(n_values, m_o_bet, l_style[net], label=f"Obs. Bet. in {net} data",
                 marker=marcadores[obs_bet_name])  # , markeredgecolor="gray"
        if plot_std:
            plt.fill_between(n_values, m_o_bet + s_o_bet, m_o_bet - s_o_bet,
                             alpha=transparency, linewidth=0)

        # TODO: obs. betweenness??

    plt.legend(ncols=3, bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize=f_size)
    plt.xticks(n_values, labels=n_labels)
    plt.xlabel("Number of nodes in the evaluation set ($n_\\text{eval}$)")
    plt.ylabel(y_axis_label)
    plt.grid(alpha=0.5, linestyle="--")
    if title:
        plt.title(f"$\\theta_\\text{{eval}} = {o}$")
    plt.tight_layout()

    plt.savefig(f"../../Dados/performance/saidas_validacao_obs_{int(o*100)}_variacao_n_e_rede{std_str}.pdf",
                transparent=True)


print("Finished third set of images")


# 3.1.1 Eixo X é o número de nós -- só rede obs 90
obs = 0.9
for o in obs_values:
    if o == obs:
        continue

    plt.figure(figsize=fig_size)

    for net in nets:
        r_model = base_model_name.format(net, obs)

        for net_option in nets:
            columns = [base_data_name.format(net_option, n, o) for n in n_values]

            m_model = df_mean.loc[r_model, columns]
            s_model = df_std.loc[r_model, columns]
            plt.plot(n_values, m_model, l_style[net_option], label=f"GNN trained in {net}, tested in {net_option}",
                     marker=marcadores[net])
            if plot_std:
                plt.fill_between(n_values, m_model + s_model, m_model - s_model, alpha=transparency)

    for net in nets:
        columns = [base_data_name.format(net, n, o) for n in n_values]
        m_o_bet = df_mean.loc[obs_bet_name, columns]
        s_o_bet = df_std.loc[obs_bet_name, columns]
        plt.plot(n_values, m_o_bet, l_style[net], label=f"Obs. Bet. in {net} data",
                 marker=marcadores[obs_bet_name])  # , markeredgecolor="gray"
        if plot_std:
            plt.fill_between(n_values, m_o_bet + s_o_bet, m_o_bet - s_o_bet,
                             alpha=transparency, linewidth=0)

        # TODO: obs. betweenness??

    plt.legend(ncols=3, bbox_to_anchor=(0.5, 1.15), loc='upper center', fontsize=f_size)
    plt.xticks(n_values, labels=n_labels)
    plt.xlabel("Number of nodes in the evaluation set ($n_\\text{eval}$)")
    plt.ylabel(y_axis_label)
    plt.grid(alpha=0.5, linestyle="--")
    if title:
        plt.title(f"$\\theta_\\text{{eval}} = {o}, \\theta = {obs}$")
    plt.tight_layout()

    plt.savefig(f"../../Dados/performance/saidas_validacao_obs_{int(o*100)}_treinoObs_{obs}_variacao_n_e_rede{std_str}.pdf",
                transparent=True)

print("Finished fourth set of images")


# # 3.2 Eixo X é a taxa de observação
# for n in n_values:
#     plt.figure(figsize=fig_size)
#         for o in obs_values:
#             r_model = base_model_name.format(net, o)

plt.close("all")

# plt.show()
