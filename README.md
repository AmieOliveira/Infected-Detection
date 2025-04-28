# Identifying Asymptomatic Nodes in Network Epidemics using Graph Neural Networks

---

This repository contains code designed to train GNNs to find assymptomatic nodes 
in a network epidemics. In this code version, an SI epidemic model is assumed, 
and the GNN makes its predictions based solely on the network's structure and 
the afforded knowledge of the currently observed infected nodes. 

Five inputs values are considered, aside from the network graph itself:
- ``OBS_I``: Observed infected nodes. That is, the infected nodes that are known to be infected
- ``DEG``: The nodes degrees.
- ``CONT``: Node's contact measure, which is the fraction of neighbors that are observed as infected.
- ``CONT_2``: Contact-2 -- fraction of observed infected nodes at distance 2 of the target node
- ``CONT_3``: Contact-3 -- fraction of observed infected nodes at distance 3 of the target node
- ``CONT_k2``: Neighborhood Contact-2 -- fraction of observed infected nodes within distance 2 of the target node
- ``BETW``: Node's betweenness metric.
- ``OBS_B``: Node's observed betweenness. In this instance, both source and target sets are observed infected nodes.

Aside from the GNN training script, a data generation script and a data 
evaluation script are made avaliable. Their usages are specified below.

This code is used to generate the results reported in the homonymous paper: 
[Link to be made available](www.google.com). 

## Data Generation
The file [gera_instancia.py](gera_instancia.py) is a script designed to crate 
and save an instance of the asymptomatic detection problem. The parameters used 
in the network and subsequent epidemics creation can be found in the 
configuration file [gera_instancia_config.yaml](gera_instancia_config.yaml), which 
contains the basic descriptions of each hyperparameter and can be changed as needed.

As it stands, epidemics can be developed in random networks originated from three 
network models:
 - `BA`: the Barabási-Albert model.
 - `WS`: the Watts-Strogatz or Small World model.
 - `ER`: the Erdös-Rényi or G(n,p) model.

One can also specify a predefined model, which should be saved in the 
[NetworkX](https://networkx.org/documentation/stable/index.html) package format.

To use the script as it is (which will generate an epidemic in a WS graph with 
30 nodes), one can simply run:
```shell
python gera_instancia.py
```

Note that the script assumes that there is a folder ``results/WS`` in your 
directory, so make sure to create one if it does not exist. 

Aside from the configuration file, one can also specify the arguments via 
command line. This will overwrite the hyperparameters of the configuration 
file. For example, if one wanted to run a network with 50 nodes, simply run:
```shell
python gera_instancia.py -n 50
```

A list of all possible command line arguments will appear with using the 
argument ``-h``.

Lastly, [submissao_instancia_cluster.py](submissao_instancia_cluster.py) 
provides an implementation used to submit instance generation jobs in a HTCondor 
cluster. This is a very system-specific script, so if your cluster is different, 
it will need to be adapted for it. 

## GNN Training 
The GNN can be trained using the file [treina_gnn.py](treina_gnn.py). As with 
the instance generation script, its parameters should be specified with a 
configuration file and/or command line arguments. An exemplifying configuration 
file is available at [treina_gnn_config.yaml](treina_gnn_config.yaml), which is 
used by default by the training script. 

Before running the script, make sure to overwrite the ``output_path`` and the 
``dataset:path`` arguments to represent the path where the GNN model and the 
training statistics should be saved and the path where the data set is stored,
respectively. 

...

## GNN Evaluation
...


