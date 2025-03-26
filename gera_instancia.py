"""
    Script to generate an instance for the infected nodes detection problem
"""

import yaml
import argparse

# TODO: Use argparse to receive alternative configuration files via command line
config_file = "gera_instancia_config.yaml"

# 1. Read from configuration file
with open(config_file, "r") as f:
    cfg = yaml.load(f, Loader=yaml.SafeLoader)

net_model = cfg["network"]["model"]
print(net_model)
print(cfg)
# TODO: Terminar de organizar o arquivo de configuração e extrair as variáveis relevantes


# TODO: Gerar rede
# TODO: Gerar epidemia
# TODO: Calcular métricas
# TODO: Salvar dados (usando a classe, e definir nome de arquivo)

