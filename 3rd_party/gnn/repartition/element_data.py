"""Element-block data model shared by sources / partitioners / rebuild."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LocalElements:
    """Elements owned by this rank, node data element-major (blocks of Np).

    ordinals are global element ordinals: the position of the element in the
    concatenation of the source ranks' element lists (source-rank major).
    They give every element a partition-independent identity so that any
    node-level data laid out in source order can be routed consistently.
    """

    Np: int
    ordinals: np.ndarray  # (Ne,) int64
    pos: np.ndarray  # (Ne*Np, 3) float64
    gids: np.ndarray  # (Ne*Np,) int64
    fields: dict = field(default_factory=dict)  # name -> (Ne*Np, k) float64

    @property
    def n_elements(self):
        return self.ordinals.shape[0]

    @property
    def n_nodes(self):
        return self.ordinals.shape[0] * self.Np

    def centroids(self):
        return self.pos.reshape(self.n_elements, self.Np, 3).mean(axis=1)
