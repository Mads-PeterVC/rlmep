from typing import Tuple

import numpy as np


def plane_to_indices(plane: str) -> Tuple[int, int, int, int]:
    """Convert a string representation of a plane to a 4-tuple of indices
    usable to perform operations using this plane as a projection.

    Parameters
    ----------
    plane : str
        String representation of the plane, in the format /[xyz]{2}[+-]?/. The
        first two characters define the plane's Cartesian axes (and should not
        be equal), and the third optional character defines whether the
        perpendicular axis is positive (+) or negative (-); the default is
        positive.

        Examples of string representations: 'xy', 'xy+', 'xy-', 'xz', 'zx+'.

    Returns
    -------
    Tuple[int, int, int, int]
        A tuple (d0, d1, d2, kv) describing this plane:
        - d0: index of the first plane axis (0, 1, or 2);
        - d1: index of the second plane axis (0, 1, or 2);
        - d2: index of the axis perpendicular to the plane (0, 1, or 2);
        - kv: sign of the axis perpendicular to the plane (-1 or 1).

    Raises
    ------
    ValueError
        Raised if the plane specification is invalid.
    """
    if not isinstance(plane, str):
        plane = str(plane)

    if len(plane) < 2:
        raise ValueError(f"Invalid plane specification {plane}")

    I = np.eye(3)

    d0 = ord(plane[0]) - ord("x")  # 'x' -> 0, 'y' -> 1, 'z' -> 2
    d1 = ord(plane[1]) - ord("x")  # 'x' -> 0, 'y' -> 1, 'z' -> 2

    if len(plane) > 2:
        s = -(ord(plane[2]) - ord(","))  # '+' -> 1, '-' -> -1
    else:
        s = 1

    if d0 not in [0, 1, 2] or d1 not in [0, 1, 2] or d0 == d1 or s not in [-1, 1]:
        raise ValueError(f"Invalid plane specification {plane}")

    d2 = ({0, 1, 2} - {d0} - {d1}).pop()

    i, j = I[d0, :], I[d1, :]
    k = np.cross(i, j)
    kv = s * int(np.sum(k))  # -1 or 1

    return (d0, d1, d2, kv)
