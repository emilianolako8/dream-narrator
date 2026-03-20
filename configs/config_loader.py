"""
config_loader.py
----------------
PURPOSE: Load and provide access to all project settings.

WHY:
Every pipeline file needs access to the config.
Instead of each file loading it separately,
we have one central loader that everyone imports.

USAGE in any pipeline file:
    from configs.config_loader import config
    
    embedding_dim = config['encoder']['embedding_dim']
    model_path    = config['encoder']['model_path']
"""

import yaml
import os


def load_config(config_path=None):
    """
    Load the YAML config file.
    """
    if config_path is None:
        # Always find config.yaml relative to THIS file
        # no matter where Python is called from
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, "config.yaml")
    
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def expand_paths(cfg):
    """
    Expand ~ in paths to full home directory path.
    
    WHY:
    YAML stores paths like "~/mne_data/..."
    Python needs the full path like "C:/Users/USER/mne_data/..."
    This function converts them automatically.
    """
    cfg['data']['raw_dir'] = os.path.expanduser(
        cfg['data']['raw_dir']
    )
    return cfg


# ── Load config once when imported ──
# Any file that does "from configs.config_loader import config"
# gets this pre-loaded dictionary instantly
config = expand_paths(load_config())


if __name__ == "__main__":
    # Test the config loader
    print("Config loaded successfully!\n")
    
    print("DATA SETTINGS:")
    for key, value in config['data'].items():
        print(f"  {key}: {value}")
    
    print("\nENCODER SETTINGS:")
    for key, value in config['encoder'].items():
        print(f"  {key}: {value}")
    
    print("\nLLM SETTINGS:")
    for key, value in config['llm'].items():
        print(f"  {key}: {value}")