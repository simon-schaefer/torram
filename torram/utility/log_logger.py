import logging
from typing import Any, Dict


class LogLogger:

    def __getattr__(self, item):
        def no_action(*args, **kwargs):
            logging.debug(f"Logger calling {item}")
        return no_action

    @staticmethod
    def add_scalar(tag, scalar_value, global_step: int, *args, **kwargs):
        logging.info(f"Logging @ {global_step}: {tag} = {scalar_value}")

    def add_scalar_dict(self, tag: str, scalar_dict: Dict[str, Any], global_step: int, *args, **kwargs):
        for key, value in scalar_dict.items():
            self.add_scalar(f"{tag}/{key}", value, global_step=global_step)
