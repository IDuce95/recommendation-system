
import os
import re
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML is required. Install it with: pip install PyYAML")

try:
    from dotenv import load_dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

class ConfigManager:

    def __init__(self, environment: str = None, config_dir: str = None):
        if _HAS_DOTENV:
            env_file = Path('.env')
            if env_file.exists():
                load_dotenv(env_file)

        self.environment = environment or os.getenv('ENVIRONMENT', 'development')
        self.config_dir = Path(config_dir or os.path.join(os.getcwd(), 'config'))
        self.config_file = self.config_dir / 'environments.yaml'
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")

        with open(self.config_file, 'r', encoding='utf-8') as file:
            all_configs = yaml.safe_load(file)

        if self.environment not in all_configs:
            raise ValueError(f"Environment '{self.environment}' not found in config file")

        config = all_configs[self.environment]

        config = self._process_env_vars(config)

        return config

    def _process_env_vars(self, config: Union[Dict, str, Any]) -> Any:
        if isinstance(config, dict):
            return {key: self._process_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._process_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config

    def _substitute_env_vars(self, value: str) -> Union[str, int, float, bool]:
        pattern = r'\$\{([^}]+)\}'

        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                env_value = os.getenv(var_name.strip(), default_value.strip())
            else:
                env_value = os.getenv(var_expr.strip())
                if env_value is None:
                    raise ValueError(f"Environment variable '{var_expr}' is required but not set")

            return self._convert_type(env_value)

        if '${' in value:
            return re.sub(pattern, replace_var, value)
        return self._convert_type(value)

    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        if value.lower() in ('true', 'yes', 'on', '1'):
            return True
        elif value.lower() in ('false', 'no', 'off', '0'):
            return False
        elif value.lower() in ('null', 'none', ''):
            return None

        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            env_key = '_'.join(keys).upper()
            env_value = os.getenv(env_key)
            if env_value is not None:
                return self._convert_type(env_value)
            return default

    def get_database_config(self) -> Dict[str, Any]:
        return self.get('database', {})

    def get_api_config(self) -> Dict[str, Any]:
        return self.get('api', {})

    def get_streamlit_config(self) -> Dict[str, Any]:
        return self.get('streamlit', {})

    def get_model_config(self) -> Dict[str, Any]:
        return self.get('model', {})

    def get_embeddings_config(self) -> Dict[str, Any]:
        return self.get('embeddings', {})

    def get_mlflow_config(self) -> Dict[str, Any]:
        return self.get('mlflow', {})

    def get_logging_config(self) -> Dict[str, Any]:
        return self.get('logging', {})

    def setup_logging(self) -> None:
        log_config = self.get_logging_config()

        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
        log_file = log_config.get('file')

        logging_kwargs = {
            'level': level,
            'format': format_str,
            'force': True
        }

        if log_file:
            logging_kwargs['filename'] = log_file

        logging.basicConfig(**logging_kwargs)

_config_manager: Optional[ConfigManager] = None

def get_config() -> ConfigManager:
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def init_config(environment: str = None, config_dir: str = None) -> ConfigManager:
    global _config_manager
    _config_manager = ConfigManager(environment=environment, config_dir=config_dir)
    return _config_manager

def get_database_config() -> Dict[str, Any]:
    return get_config().get_database_config()

def get_api_config() -> Dict[str, Any]:
    return get_config().get_api_config()

def get_streamlit_config() -> Dict[str, Any]:
    return get_config().get_streamlit_config()

def get_model_config() -> Dict[str, Any]:
    return get_config().get_model_config()

def get_embeddings_config() -> Dict[str, Any]:
    return get_config().get_embeddings_config()

def setup_logging() -> None:
    get_config().setup_logging()
