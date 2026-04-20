"""
utils/config_loader.py
Loads config.json and resolves {base} placeholders.
"""

import json
import os


def load_config(config_path: str = None) -> dict:
    """
    Load config.json from the given path (or from the default location
    next to this file's parent directory) and resolve all {base}
    placeholders with the actual base_path value.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.json"
        )

    with open(config_path, "r") as f:
        cfg = json.load(f)

    base = cfg["base_path"]

    def resolve(obj):
        if isinstance(obj, str):
            return obj.replace("{base}", base)
        if isinstance(obj, dict):
            return {k: resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(cfg)
