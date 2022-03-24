import glob
import os
from typing import Any, Dict, List

__all__ = ['find_all_files',
           'write_dict_to_csv']


def find_all_files(directory: str, suffix: str) -> List[str]:
    """Find all files in directory and subdirectories of the directory, with given suffix.

    Args:
        directory: query directory
        suffix: file suffix, e.g. ".npy".
    """
    query_path = os.path.join(directory, "**", f"*{suffix}")
    return glob.glob(query_path, recursive=True)


def write_dict_to_csv(dictionary: Dict[str, Any], csv_file: str):
    """Write dictionary into csv file.

    Args:
        dictionary: dictionary to save.
        csv_file: file to save dictionary in.
    """
    if not csv_file.endswith(".csv"):
        raise ValueError(f"Invalid filename for CSV file {csv_file}")
    with open(csv_file, 'w+') as f:
        for key, value in dictionary.items():
            space = " " * (20 - len(key))
            line = f"{key}:{space}{value}\n"
            f.write(line)
