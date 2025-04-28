"""
    Script to generate configuration files for model evaluation (and run them?)
"""

import os
import yaml
import subprocess

# Input elements
output_path = "../Dados/Avaliacao"
model_path = "../Dados/Resultados"
nk_nodes = [1, 3, 6]

data_paths = [f"../Dados/testeBA_n{n}k" for n in nk_nodes]
base_model_name = "{}_{}"
net_models = ["BA", "WS"]
obs_values = [0.1, 0.25, 0.5, 0.75, 0.9]
obs_strs = [f"obs_{int(100*o)}" for o in obs_values]
# --------------


# Begin extracting configurations
configs = {
    "rd_seed": 0,
    "output_path": output_path,
}

models = []
names = []
for filename in os.listdir(model_path):
    if filename[-4:] == ".gnn":
        filebase = filename[6:-4]
        # print(filebase)
        models += [filebase]
        net = None
        for m in net_models:
            if m in filebase:
                net = m
                break
        obs = None
        for o in obs_strs:
            if o in filebase:
                obs = o
                break
        names += [base_model_name.format(net, obs)]

# print(models)
# print(names)

for d_path in data_paths:
    for folder in os.listdir(d_path):
        net = None
        for m in net_models:
            if m in d_path:
                net = m
                break

        name = net + "_" + folder
        dataset = os.path.join(d_path, folder)
        # print(dataset)

        configs["dataset"] = {
            "name": name,
            "path": dataset,
        }

        for idx, mod in enumerate(models):
            configs["model"] = {
                "name": names[idx],
                "path": model_path,
                "filebase": mod,
            }

            c_filename = f"development/modelo-{names[idx]}_data-{name}.yaml"

            with open(c_filename, "w") as yamlfile:
                data = yaml.dump(configs, yamlfile)
            print(f"Wrote configuration file to path: {c_filename}")

            command = f"python avalia_modelo.py {c_filename}"
            print(f"Executing command: '{command}'\n\n")
            subprocess.run(command.split(" "))
            print("\n\nEnd of command\n\n")

