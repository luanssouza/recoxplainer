import yaml
from box import Box


def load_config(path: str) -> Box:
    with open(path, "r") as yml_file:
        full_cfg = yaml.safe_load(yml_file)
    return Box({**full_cfg["base"]}, default_box=True, default_box_attr=None)
