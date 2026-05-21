import os
import sys
from typing import Optional, Union, Tuple
import logging
from omegaconf import DictConfig
import numpy as np
from time import sleep, perf_counter

log = logging.getLogger(__name__)

class OnlineClient:
    """Class for the online training client
    """
    def __init__(self) -> None:
        self.client = None
