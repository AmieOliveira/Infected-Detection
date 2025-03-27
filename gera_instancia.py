"""
    Script to generate an instance for the infected nodes detection problem
"""

import yaml
import argparse


parser = argparse.ArgumentParser(
    prog="Hidden network epidemics generator",
    description="Script designed to generate an instance of a network "
                "epidemie with hidden infected nodes. Customizations "
                "should be given via a configuration file."
)
parser.add_argument(
    "configfile",
    default="gera_instancia_config.yaml", nargs="?",
    help="Name of the configuration file to be used. "
         "If none is given, then the default "
         "'gera_instancia_config.yaml' will be used."
)


# 1. Read from configuration file
args = parser.parse_args()
config_file = args.configfile
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

