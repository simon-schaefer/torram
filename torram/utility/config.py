import logging
import os
import yaml

from typing import Any, Dict, Optional


class RequiredEntryError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


class ExistingEntryError(RuntimeError):
    def __init__(self, arg):
        self.args = arg


class Config:

    def __init__(self, config: Dict[str, Any]):
        config_flat = self.__flatten(config, sep="/")
        self._dict = config_flat

    @classmethod
    def empty(cls):
        return cls({})

    @property
    def dict(self):
        return self._dict

    ###################################################################################################################
    # Getter/Setter Functions #########################################################################################
    ###################################################################################################################
    def get(self, key: str, default: Optional[Any] = None, required: bool = False):
        value = self._dict.get(key, None)
        logging.debug(f"Retrieving config for key {key} ... got value {value}")
        if value is not None:
            return value

        # If the key is required, raise an error and abort.
        if required:
            raise RequiredEntryError(f"Key {key} is required and missing")

        # Otherwise add the default to the internal config and return it.
        logging.debug(f"... added {key} = {default} to config")
        self._dict[key] = default
        return default

    def set(self, key: str, value: Any):
        if key in self._dict.keys():
            raise ExistingEntryError(f"Key {key} is already existing in config")
        self._dict[key] = value

    ###################################################################################################################
    # I/O Functions ###################################################################################################
    ###################################################################################################################
    @classmethod
    def from_yaml(cls, f_path: str):
        assert os.path.isfile(f_path)
        with open(f_path) as f:
            config = yaml.full_load(f)
        return cls(config)

    def save_yaml(self, f_path: str):
        os.makedirs(os.path.dirname(f_path), exist_ok=True)
        with open(f_path, 'w+') as f:
            yaml.dump(self.__un_flatten(self._dict), f)

    ###################################################################################################################
    # Utility #########################################################################################################
    ###################################################################################################################
    def __flatten(self, d, parent_key='', sep="/"):
        """from https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys."""
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.__flatten(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    @staticmethod
    def __un_flatten(dictionary, sep="/"):
        """from https://stackoverflow.com/questions/6037503/python-unflatten-dict"""
        resultDict = dict()
        for key, value in dictionary.items():
            parts = key.split(sep)
            d = resultDict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = dict()
                d = d[part]
            d[parts[-1]] = value
        return resultDict

    def __str__(self):
        space_between = max(len(key) for key in self._dict.keys()) + 10
        outputs = []
        for key, value in self._dict.items():
            text_key_value = str(key) + " " * (space_between - len(key) - 1) + str(value)
            outputs.append(text_key_value)
        return "\n".join(outputs)
