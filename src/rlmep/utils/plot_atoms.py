import itertools
from typing import Any, Dict, List, Optional, Union

import ase.data
import ase.data.colors
import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path

from rlmep.utils.colors import ColorType, Colors
from rlmep.utils.plane_to_indices import plane_to_indices

def plot_atoms(
    ax: Axes,
    atoms: Atoms,
    *,
    colors: Optional[Union[Colors, List[ColorType]]] = None,
    plane: str = "xy+",
    radius_factor: float = 1.0,
    repeat: int = 0,
    plot_constraint: bool = False,
    patch_kwargs: Optional[Dict[str, Any]] = None,
    repeat_patch_kwargs: Optional[Dict[str, Any]] = None,
) -> PatchCollection:
    """Plot a 2D projection of an `Atoms` object into an `Axes` object.

    Parameters
    ----------
    ax : Axes
        Axes to plot the atoms in.
    atoms : Atoms
        Atoms to plot.
    colors: Optional[Union[Colors, List[ColorType]]], optional
        List of colors for every atom. If None, use the jmol colors. The
        `Colors` class can be used and provided to easily generate a list of
        colors.
    plane : str, optional
        Plane to project the atoms on. See `plane_to_indices` for information
        on valid plane strings, by default 'xy+'.
    radius_factor : float, optional
        Factor to multiply the atomic covalent radii with, by default 1.0.
    repeat : int, optional
        Number of times to repeat the Atoms object in dimensions with periodic
        boundary conditions, by default 0. Repeated copies are not included in
        autoscaling. Note that if none of the two projection axes have periodic
        boundary conditions, this parameter is effectively ignored.
    plot_constraint : bool, optional
        Whether to draw crosses on top of the atoms that are fixed by a
        `FixAtoms` constraint, similar to the ASE GUI, by default False.
    patch_kwargs : Optional[Dict[str, Any]], optional
        Keyword arguments forwarded to the `Circle` constructor for each atom,
        by default None.
    repeat_patch_kwargs : Optional[Dict[str, Any]], optional
        Keyword arguments additionally forwarded to the `Circle` constructor
        for each repeated atom, by default None.

    Returns
    -------
    PatchCollection
        Collection containing patches that make up the Atoms object.
    """

    if len(atoms) == 0:
        return PatchCollection([])

    if colors is None:
        colors = Colors(atoms)

    if patch_kwargs is None:
        patch_kwargs = {}
    if repeat_patch_kwargs is None:
        repeat_patch_kwargs = {}

    d0, d1, d2, kv = plane_to_indices(plane)

    radii = radius_factor * ase.data.covalent_radii[atoms.numbers]

    fixed = np.zeros(len(atoms), dtype=bool)
    if plot_constraint:
        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                fixed[constraint.index] = True

    xy_min = np.min(atoms.positions[:, [d0, d1]] - radii[:, np.newaxis], axis=0)
    xy_max = np.max(atoms.positions[:, [d0, d1]] + radii[:, np.newaxis], axis=0)

    if repeat > 0:
        atoms_ = atoms.copy()
        pbc = atoms.get_pbc()

        repeat_range_d0 = range(-repeat, repeat + 1) if pbc[d0] else [0]
        repeat_range_d1 = range(-repeat, repeat + 1) if pbc[d1] else [0]

        for x, y in itertools.product(repeat_range_d0, repeat_range_d1):
            if x == y == 0:
                continue
            offset_uc = np.zeros(3)
            offset_uc[[d0, d1]] = [x, y]
            offset = atoms.cell.cartesian_positions(offset_uc)
            repeat_atoms = atoms.copy()
            repeat_atoms.translate(offset)
            atoms_ += repeat_atoms
    else:
        atoms_ = atoms

    n_atoms = len(atoms)
    patches: List[Circle] = []

    for i in np.argsort(kv * atoms_.positions[:, d2]):
        patch_kwargs_ = {**patch_kwargs} if i < n_atoms else {**patch_kwargs, **repeat_patch_kwargs}

        position = atoms_[i].position
        radius = radii[i % n_atoms]
        color = colors[i % n_atoms]

        circle = Circle(position[[d0, d1]], radius=radius, ec="k", fc=color, **patch_kwargs_)

        patches.append(circle)

        if fixed[i % n_atoms]:
            path_patch_kwargs = {k: v for k, v in patch_kwargs_.items() if k not in ["fc", "facecolor"]}

            delta = radius * 0.5 * np.array([np.sqrt(2), np.sqrt(2)])

            p1 = position[[d0, d1]] + delta * [-1, 1]
            p2 = position[[d0, d1]] + delta * [1, -1]
            p3 = position[[d0, d1]] + delta * [-1, -1]
            p4 = position[[d0, d1]] + delta * [1, 1]

            verts = [p1, p2, p3, p4]
            codes = [Path.MOVETO, Path.LINETO, Path.MOVETO, Path.LINETO]
            path = Path(verts, codes)
            path_patch = PathPatch(path, **path_patch_kwargs)

            patches.append(path_patch)

    p = PatchCollection(patches, match_original=True)

    ax.add_collection(p, autolim=False)
    ax.update_datalim((xy_min, xy_max))
    ax.set_aspect("equal")
    ax.tick_params(
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labeltop=False,
        labelleft=False,
        labelright=False,
    )
    ax.autoscale_view()

    return p
