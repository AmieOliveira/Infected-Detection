"""
    Script to generate submission files and their attached configuration files
    for cluster submission of instance generation jobs
"""

import os
import yaml
import random as rd

# --------------------------------
# Parameters to be given as inputs
# --------------------------------
base_folder = "results"

# Random seed
base_seed = 463
change_seed = False

experiment_name = "test"

# Network parameters
model = "WS"  # "WS", "ER", "adjacency"
n_nodes = 20
ba_m = 2
ws_k = 4
ws_p = 0.3
er_p = 0.3
adjacency_path = None
adjacency_name = None

# Epidec parameters
beta = [0.1, 0.2]
i_infected = 1
tot_time = 0
stop_fraction = [0.2, 0.3]
observation_prob = [0.5, 0.7]
# --------------------------------

# -------------
# Naming scheme
# -------------
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
folder = f"results/{model}"
if not os.path.exists(folder):
    os.makedirs(folder)

rede = adjacency_name if model == adjacency_name else model
base_name = f"instancias_rede-{rede}_{experiment_name}_seed{base_seed}"

rd.seed(base_seed)
# -------------

# -------------------------
# Create configuration file
# -------------------------
c_name = f"{base_name}.yaml"
c_filename = f"{folder}/{c_name}"

configs = {
    "rd_seed": base_seed,
    "network": {
        "model": model,
        "n_nodes": n_nodes,
        "adjacency": {"path": adjacency_path},
        "BA": {"m": ba_m},
        "WS": {"k": ws_k, "p": ws_p},
        "ER": {"p": er_p},
    },
    "epidemic": {
        "beta": beta if type(beta) != list else beta[0],
        "init_infec": i_infected,
        "total_time": tot_time,
        "stop_frac": stop_fraction if type(stop_fraction) != list else stop_fraction[0],
        "observ_prob": observation_prob if type(observation_prob) != list else observation_prob[0],
    },
}

with open(c_filename, "w") as yamlfile:
    data = yaml.dump(configs, yamlfile)
    print(f"Wrote configuration file to path: {c_filename}")
# -------------------------

# ----------------------
# Create submission file
# ----------------------
s_name = f"{base_name}.sub"
s_filename = f"{folder}/{s_name}"
w_dir = os.getcwd()
execution_file = "gera_instancia.py"

with open(s_filename, "w") as subfile:
    subfile.write(f"Executable\t\t\t\t= {w_dir}/{execution_file}\n")
    subfile.write(f"initialdir\t\t\t\t= {w_dir}/{folder}\n")
    subfile.write(f"Universe\t\t\t\t= vanilla\n")
    subfile.write(f"should_transfer_files\t\t= YES\n")
    subfile.write(f"when_to_transfer_output\t\t= ON_EXIT\n")
    subfile.write(f"transfer_input_files\t\t= {w_dir}/epidemic.py,{w_dir}/centrality.py,{w_dir}/formatacao.py,{w_dir}/{c_filename}\n")
    subfile.write(f'transfer_output_remaps\t\t= "{w_dir}/epidemic.py = epidemic.py"\n')
    subfile.write(f'transfer_output_remaps\t\t= "{w_dir}/centrality.py = centrality.py"\n')
    subfile.write(f'transfer_output_remaps\t\t= "{w_dir}/formatacao.py = formatacao.py"\n')
    subfile.write(f'transfer_output_remaps\t\t= "{w_dir}/{c_filename} = {c_name}"\n')

    s = base_seed

    for b in beta:
        for f in stop_fraction:
            for o in observation_prob:
                if change_seed:
                    s += rd.randint(-10, 10)

                params = ""
                if model != "adjacency":
                    configs["network"][model]["n"] = n_nodes
                keys = list(configs["network"][model].keys())
                keys.sort()
                for key in keys:
                    params += f"{key}{configs['network'][model][key]}"

                epinfo = f"b{b}-ii{i_infected}-t{tot_time}-f{f}-o{o}"

                outfilename = f"instance_model{model}-{params}_epidemic-{epinfo}_s{s}.pkl"

                subfile.write(f"\n\n")
                subfile.write(f'Arguments\t\t\t\t= "{c_name} -b {b} -f {f} -o {o} -s {s} -d ."\n')
                subfile.write(f"Log\t\t\t\t\t= {w_dir}/log/{base_name}.log\n")
                subfile.write(f"Error\t\t\t\t\t= {w_dir}/error/{base_name}_b{b}_f{f}_o{o}_s{s}.err\n")
                subfile.write(f"Output\t\t\t\t\t= {w_dir}/out/{base_name}_b{b}_f{f}_o{o}_s{s}.out\n")
                subfile.write(f"transfer_output_files\t= {outfilename}\n")
                subfile.write(f"Queue 1")
    # TODO: Definir o nome do arquivo de saída para terminar o arquivo de configuração

    print(f"Wrote submission file to path: {s_filename}")
# ----------------------
