import os

import toml


class Configuration:
    """
        To Fill
    """

    def __init__(self, settings_file):
        with open(settings_file, "r", encoding="utf-8") as file:
            self.app_config = toml.load(file)

    @staticmethod
    def _get_current_env():
        # Possible values: unset, DEV, INTEGRATION, RELEASE, PROD, LOCAL
        return os.environ.get("DEPLOYMENT_ENV", "LOCAL")

    def get_config(self, not_env_specific=False) -> dict:
        if not_env_specific:
            return self.app_config

        default_config = self.app_config.get("DEFAULT", {})
        # Override default config with environment specific config
        updated_config = self.update_config(default_config, self._get_current_env())

        return updated_config

    def update_config(self, default_config, environment) -> dict:
        """
        Merge/Update default config with header config
        """
        environment_config = self.app_config.get(environment, {})
        default_config.update(environment_config)

        return default_config