"""
This module contains all the functions used to set the seeds for the package, to make the experiments reproducible.
"""

import os
import random

import numpy as np


def set_seeds(seed=42):
    """"
    It sets the seeds for all the libraries that can eventually
    use random seeds.
    (Scikit-learn is not present because it does not have its own
    global random state but uses the numpy random state instead).

    Args:
        - seed (int): used to specify a seed for the libraries.

    Returns: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
