import warnings
from pathlib import Path
from typing import Union

import numpy as np
import toml


class Conf(dict):
    """Sub-class of dict that overrides __getitem__ to allow for keys not in
    the original dict, defaulting to None.
    """

    def __init__(self, *args, **kwargs):
        """Update dict with all keys from dict"""
        self.update(*args, **kwargs)

    def __getitem__(self, key):
        """Get key from dict. If not present, return None and raise warning
        Parameters
        ----------
        key : Hashable
            key to get from original dict
        Returns
        -------
            original value in the dict or None if not present
        """
        if key not in self:
            warnings.warn(f"Key '{key}' not in conf. Defaulting to None")
            val = None
        else:
            val = dict.__getitem__(self, key)

        # Transform list to numpy
        if type(val) is list:
            val = np.array(val)
        return val


def load_conf(path: Union[str, Path], key: str = None) -> Conf:
    """Load TOML config as dict-like

    Parameters
    ----------
    path : str
        Path to TOML config file
    key : str, optional
        Section of the conf file to load

    Returns
    -------
    Conf
        Config dictionary

    """
    config = toml.load(path)
    return Conf(config) if key is None else Conf(config[key])
