from typing import Any, Dict, Optional, Union

import numpy as np
from ase.cell import Cell
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from numpy.typing import NDArray

from rlmep.utils.plane_to_indices import plane_to_indices


def plot_cell(
    ax: Axes,
    cell: Union[Cell, NDArray],
    *,
    plane: str = "xy+",
    offset: Optional[NDArray] = None,
    collection_kwargs: Dict[str, Any] = None,
):
    """Plot a 2D projection of a `Cell` object into an `Axes` object.

    Parameters
    ----------
    ax : Axes
        Axes to plot the cell in.
    cell : Union[Cell, NDArray]
        Cell to plot.
    plane : str, optional
        Plane to project the cell on. See `plane_to_indices` for information on
        valid plane strings, by default 'xy+'.
    offset : Optional[NDArray], optional
        Cartesian coordinates to apply as offset to the base of the cell, by
        default None (no offset).
    collection_kwargs : Dict[str, Any], optional
        Keyword arguments additionally forwarded to the `PolyCollection`
        constructor for the cell, by default None.

    Raises
    ------
    ValueError
        Raised if `offset` or `cell` has an incorrect shape.
    """

    default_collection_kwargs = dict(edgecolors="k", facecolors="none", linestyles="dotted")

    if offset is None:
        offset = np.zeros(3)
    else:
        offset = np.asarray(offset)

    if offset.shape != (3,):
        raise ValueError(f"`offset` must have shape (3,), input has {offset.shape}")

    if isinstance(cell, Cell):
        cell = cell.complete()

    if cell.shape != (3, 3):
        raise ValueError(f"`cell` must have shape (3, 3), input has {cell.shape}")

    if collection_kwargs is None:
        collection_kwargs = {}

    collection_kwargs = {**default_collection_kwargs, **collection_kwargs}

    d0, d1, _, _ = plane_to_indices(plane)

    verts_uc = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],  # xy, z = 0
            [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],  # xy, z = 1
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],  # xz, y = 0
            [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],  # xz, y = 1
            [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],  # yz, x = 0
            [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],  # yz, x = 1
        ]
    )
    verts_3d = verts_uc @ cell + offset
    verts_2d = verts_3d[:, :, [d0, d1]]

    p = PolyCollection(verts_2d, **collection_kwargs)
    ax.add_collection(p)
    ax.autoscale_view()
