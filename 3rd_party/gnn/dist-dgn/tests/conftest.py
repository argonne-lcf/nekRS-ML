"""Pytest configuration: put the dist-dgn directory on sys.path so the bare
imports inside the package (``from gnn import ...``, ``from graph_transformer
import ...``) resolve when tests are invoked from anywhere.
"""

import os
import sys

_DIST_DGN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DIST_DGN_DIR not in sys.path:
    sys.path.insert(0, _DIST_DGN_DIR)
