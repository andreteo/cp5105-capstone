# Script to load and parse the config file

import json
from pathlib import Path


def load_config(config_filename='config/pluto-config.json'):
    config_path = Path(__file__).resolve().parent.parent / config_filename

    with open(config_path, "r") as f:
        return json.load(f)
