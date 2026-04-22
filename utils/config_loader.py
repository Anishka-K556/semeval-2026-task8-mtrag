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

    # Keys whose values must NOT have {base} substituted
    # (they use {context}/{query} as runtime .format() placeholders)
    _skip_keys = {"prompt_template"}

    def resolve(obj, key=None):
        if key in _skip_keys:
            return obj                          # leave template strings untouched
        if isinstance(obj, str):
            return obj.replace("{base}", base)
        if isinstance(obj, dict):
            return {k: resolve(v, key=k) for k, v in obj.items()}
        if isinstance(obj, list):
            return [resolve(v) for v in obj]
        return obj

    return resolve(cfg)


def load_task_c_config(config_path: str = None) -> dict:
    """
    Convenience wrapper: returns only the resolved `task_C` sub-dict.
    Task-C scripts should call this instead of load_config() directly.
    """
    full = load_config(config_path)
    if "task_C" not in full:
        raise KeyError("'task_C' section not found in config.json")
    return full["task_C"]
