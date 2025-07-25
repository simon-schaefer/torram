import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

OmegaConfigType = DictConfig | ListConfig


def load_config_from_files_and_cli(schema: Any) -> Tuple[Any, bool]:
    """Load a configuration from argparse and merge it with a schema."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", required=True)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    args, unknown_args = parser.parse_known_args()
    config = read_config(args.config, schema, unknown_args)
    return config, args.debug


def read_config(
    config_files: Optional[List[Path]],
    schema: Any,
    unknown_args: Optional[List[str]] = None,
):
    """Read a config file and optionally overwrite values using command-line arguments.

    @param config_path (str): Path to YAML or JSON config file.
    @param unknown_args (list): List of unknown args from argparse, e.g., ["--lr", "0.01", "--batch_size", "64"].
    @return Final configuration dictionary.
    """
    if config_files is not None and len(config_files) > 0:
        config = [OmegaConf.load(config_file) for config_file in config_files]
    else:
        config = OmegaConf.create([])

    # Merge with default schema and enforce structure.
    config = OmegaConf.create(config)

    # If unknown_args is provided, parse and add them to the config.
    if unknown_args:
        unknown_args = [arg.replace("--", "") for arg in unknown_args if arg.startswith("--")]
        config.append(OmegaConf.from_dotlist(unknown_args))

    return OmegaConf.merge(OmegaConf.structured(schema), *config)
