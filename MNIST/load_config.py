import yaml


def load_config(config) :
    with open(config) as file :
        config = yaml.safe_load(file)

    return config

