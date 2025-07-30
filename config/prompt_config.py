import os
import toml
from typing import Dict, Any

class PromptConfig:

    def __init__(self, config_path: str = None):
        if config_path is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(project_root, "config", "prompts.toml")

        self.config_path = config_path
        self._config = None
        self._load_config()

    def _load_config(self):
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = toml.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt configuration file not found: {self.config_path}")
        except Exception as e:
            raise Exception(f"Error loading prompt configuration: {e}")

    def get_system_prompt(self) -> str:
        return self._config.get("prompts", {}).get("system_prompt", "")

    def get_recommendation_template(self) -> str:
        return self._config.get("prompts", {}).get("recommendation_template", "")

    def get_product_format_template(self) -> str:
        return self._config.get("prompts", {}).get("product_format", "")

    def get_all_prompts(self) -> Dict[str, Any]:
        return self._config.get("prompts", {})
