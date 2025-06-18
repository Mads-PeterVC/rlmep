from typing import List, Optional, Tuple, Union

import numpy as np
from ase.atoms import Atoms
from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from matplotlib.colors import to_rgba, to_rgba_array
from numpy.typing import NDArray
from typing_extensions import Self
from rlmep.utils.plane_to_indices import plane_to_indices

# matplotlib 3.8.0 defines `maptplotlib.typing.ColorType`
ColorType = Union[
    Tuple[float, float, float],  # RGB
    Tuple[float, float, float, float],  # RGBA
    str,  # hex, grayscale, color names, CN
]

IndicesType = Optional[Union[int, List[int], NDArray]]


class Colors:
    def __init__(self, atoms: Atoms, color_palette: Optional[List[ColorType]] = None):
        """Container to store atomic colors for plotting purposes.

        The public methods of this class all return `self`, such that they can
        be chained. For example:
        `Colors(atoms).lighten(...).set_template('white')`

        Parameters
        ----------
        atoms : Union[Atoms, CandidateBaseClass]
            Atoms or Candidate object to store colors for.
        color_palette : Optional[List[ColorType]], optional
            Color palette to map elements to initial colors. If not set,
            `ase.data.colors.jmol_colors` is used.
        """

        self.atoms = atoms

        if color_palette is None:
            color_palette = jmol_colors

        self.colors: NDArray = to_rgba_array([color_palette[atom.number] for atom in self.atoms])

    def set_color(self, color: ColorType, *, indices: IndicesType = None) -> Self:
        """Set the color of one or more atoms.

        Parameters
        ----------
        color : ColorType
            Color to set. This can be any matplotlib-supported color.
        indices : IndicesType, optional
            Atomic indices to set color for, by default None (all atoms).

        Returns
        -------
        Self
        """

        indices = self._get_indices(indices)

        self.colors[indices] = to_rgba(color)

        return self

    def set_alpha(self, alpha: float, *, indices: IndicesType = None) -> Self:
        """Set the alpha value of one or more atoms.

        Parameters
        ----------
        alpha : float
            Alpha value to set.
        indices : IndicesType, optional
            Atomic indices to set color for, by default None (all atoms).

        Returns
        -------
        Self
        """

        indices = self._get_indices(indices)

        self.colors[indices, 3] = alpha

        return self

    def set_element(self, element: Union[str, int], color: ColorType) -> Self:
        """Set the color of all atoms of one element.

        Parameters
        ----------
        element : Union[str, int]
            Element to set color for. This can be a symbol or an atomic number.
        color : ColorType
            Color to set. This can be any matplotlib-supported color.

        Returns
        -------
        Self
        """

        if isinstance(element, str):
            element = atomic_numbers[element]

        self.set_color(color, indices=np.flatnonzero(self.atoms.numbers == element))

        return self

    def set_template(self, color: ColorType) -> Self:
        """Set the color of all atoms of the template. This requires that this
        object was instantiated with a Candidate object.

        Parameters
        ----------
        color : ColorType
            Color to set. This can be any matplotlib-supported color.

        Returns
        -------
        Self

        Raises
        ------
        ValueError
            Raised when this object was not instantiated with a Candidate
            object.
        """
        self.set_color(color, indices=self.atoms.template_indices)

        return self

    def darken(
        self, *, factor: float = 0.1, threshold: Optional[float] = None, plane: str = "xy+", indices: IndicesType = None
    ) -> Self:
        """Darken atom colors by a given factor.

        If `threshold` is set, atom colors are darkened based on the distance
        from `threshold` to the atomic coordinate (in the direction pointing
        out of the figure), multiplied by `factor`.
        Otherwise, atom colors are darkened by a constant factor `factor`.

        Parameters
        ----------
        factor : float, optional
            Factor to darken atom colors by, by default 0.1.
        threshold : Optional[float], optional
            Threshold to darken atom colors from, by default None (constant
            darken factor).
        plane : str, optional
            Plane to project the atoms on. See `plane_to_indices` for
            information on valud plane strings, by default 'xy+'.
        indices : IndicesType, optional
            Atomic indices to darken color for, by default None (all atoms).

        Returns
        -------
        Self
        """

        indices = self._get_indices(indices)
        _, _, d2, kv = plane_to_indices(plane)

        factor_ = (
            factor
            if threshold is None
            else ((kv * self.atoms[indices].positions[:, d2] - threshold) * factor)[:, np.newaxis]
        )
        self.colors[indices, :3] = self.colors[indices, :3] * np.minimum(1, 1 - np.minimum(1, factor_))

        return self

    def lighten(
        self, *, factor: float = 0.1, threshold: Optional[float] = None, plane: str = "xy+", indices: IndicesType = None
    ) -> Self:
        """Lighten atom colors by a given factor.

        If `threshold` is set, atom colors are lightened based on the distance
        from `threshold` to the atomic coordinate (in the direction pointing
        out of the figure), multiplied by `factor`.
        Otherwise, atom colors are lightened by a constant factor `factor`.

        Parameters
        ----------
        factor : float, optional
            Factor to lighten atom colors by, by default 0.1.
        threshold : Optional[float], optional
            Threshold to lighten atom colors from, by default None (constant
            lighten factor).
        plane : str, optional
            Plane to project the atoms on. See `plane_to_indices` for
            information on valud plane strings, by default 'xy+'.
        indices : IndicesType, optional
            Atomic indices to lighten color for, by default None (all atoms).

        Returns
        -------
        Self
        """

        indices = self._get_indices(indices)
        _, _, d2, kv = plane_to_indices(plane)

        factor_ = (
            factor
            if threshold is None
            else ((kv * self.atoms[indices].positions[:, d2] - threshold) * factor)[:, np.newaxis]
        )
        self.colors[indices, :3] = 1 - (1 - self.colors[indices, :3]) * np.minimum(1, 1 - np.minimum(1, factor_))

        return self

    def _get_indices(self, indices: IndicesType) -> NDArray:
        """Internal method to convert `indices` arguments to a numpy array of
        indices.

        Parameters
        ----------
        indices : IndicesType
            Indices to convert; either `int`, list of `int`, or numpy array.

        Returns
        -------
        NDArray
            Numpy array of indices.
        """
        if isinstance(indices, int):
            return np.array([indices], dtype=int)
        elif indices is None:
            return np.arange(len(self.atoms))
        else:
            return np.array(indices, dtype=int)

    def __getitem__(self, index) -> NDArray:
        """Return the color of an atom or atoms.

        Parameters
        ----------
        index
            Index or indices of atom or atoms.

        Returns
        -------
        NDArray
            Color of atom or atoms.
        """
        return self.colors[index]
