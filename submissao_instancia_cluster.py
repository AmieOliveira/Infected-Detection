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
base_seed = 0

experiment_name = "experiment_name"
n_instances = 1000

# Network parameters
model = "WS"    # "BA", "WS", "ER", "adjacency"
n_nodes = 3000
ba_m = 2
ws_k = 4
ws_p = 0.3
er_p = 0.3
adjacency_path = None
adjacency_name = None

# Epidemic parameters
beta = [0.1, 0.3, 0.5]
i_infected = 1
tot_time = 0
stop_fraction = [0.2]
observation_prob = [0.1, 0.25, 0.5, 0.75, 0.9]
# --------------------------------

# -------------
# Naming scheme
# -------------
if not os.path.exists(base_folder):
    os.makedirs(base_folder)
folder = f"{base_folder}/{model}/{experiment_name}"
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
        "beta": beta,
        "init_infec": i_infected,
        "total_time": tot_time,
        "stop_frac": stop_fraction if type(stop_fraction) != list else stop_fraction[0],
        "observ_prob": observation_prob,
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
    subfile.write('requirements = (Machine == "node04") || (Machine == "node03") || (Machine == "node01")')

    for f in stop_fraction:

        params = ""
        if model != "adjacency":
            configs["network"][model]["n"] = n_nodes
        keys = list(configs["network"][model].keys())
        keys.sort()
        for key in keys:
            params += f"{key}{configs['network'][model][key]}"

        epinfo = f"ii{i_infected}-t{tot_time}-f{f}"

        outfilename = f"instance_model{model}-{params}_epidemic-{epinfo}_run$(Step).pkl"

        subfile.write(f"\n\n")
        subfile.write(f'Arguments\t\t\t\t= "{c_name} -f {f} -s $(Step) -d ."\n')
        subfile.write(f"Log\t\t\t\t\t\t= {w_dir}/log/{base_name}.log\n")
        subfile.write(f"Error\t\t\t\t\t= {w_dir}/error/{base_name}_f{f}_run$(Step).err\n")
        subfile.write(f"Output\t\t\t\t\t= {w_dir}/out/{base_name}_f{f}_run$(Step).out\n")
        subfile.write(f"transfer_output_files\t= {outfilename}\n")
        subfile.write(f"Queue {n_instances}")

    print(f"Wrote submission file to path: {s_filename}")
# ----------------------
