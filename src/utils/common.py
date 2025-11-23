import yaml
from pathlib import Path

def read_params(config_path: Path) -> dict:
    # Reads the YAML configuration file and returns its content as a dictionary.
    try:
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return {}
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return {}
