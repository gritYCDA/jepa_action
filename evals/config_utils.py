# config_utils.py

import yaml
import re
from typing import Dict, Any
import copy

def replace_env_vars(config: Dict[Any, Any], env_paths: Dict[str, str]) -> Dict[Any, Any]:
    """
    Replace environment variables in the config with actual values
    
    Args:
        config: The configuration dictionary
        env_paths: Environment-specific paths
        
    Returns:
        Config with replaced values
    """
    def replace_in_value(value: Any) -> Any:
        if isinstance(value, str) and value.startswith('${env:'):
            key = value[6:-1]  # Remove ${env: and }
            return env_paths[key]
        return value

    def process_dict(d: Dict[Any, Any]) -> Dict[Any, Any]:
        result = {}
        for key, value in d.items():
            if isinstance(value, dict):
                result[key] = process_dict(value)
            else:
                result[key] = replace_in_value(value)
        return result

    return process_dict(config)

def load_config(fname: str, env: str) -> Dict[Any, Any]:
    """
    Load and process the config file based on the specified environment.
    
    Args:
        fname: Path to the config file
        env: Environment name ('local' or 'azure')
        
    Returns:
        Processed config dictionary
    """
    with open(fname, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'env_paths' not in config:
        raise KeyError("Config file must contain 'env_paths' section")
    
    if env not in config['env_paths']:
        raise KeyError(f"Environment '{env}' not found in config file")
    
    # Get environment-specific paths
    env_paths = config['env_paths'][env]
    
    # Remove env_paths section from config
    config_without_env = copy.deepcopy(config)
    del config_without_env['env_paths']
    
    # Replace environment variables
    final_config = replace_env_vars(config_without_env, env_paths)
    
    return final_config