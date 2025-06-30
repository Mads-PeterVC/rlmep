import matplotlib.pyplot as plt
import numpy as np


class GridSpec:
    """
    A class to define the grid specifications for the discrete MEP environment.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_spacing: float,
        corner: tuple[float, float] | None = None,
        height: float = 0.0,
    ):
        """
        Parameters
        ----------
        grid_size: tuple[int, int]
            The size of the grid.
        grid_spacing: float
            The spacing between the grid points.
        """
        self.grid_size = grid_size
        self.grid_spacing = grid_spacing
        self.corner = corner if corner is not None else (0.0, 0.0)
        self.height = height

    def get_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the grid points in the x and y directions.
        """

        p0 = np.array(self.corner)
        p1 = self.ij_to_xyz(self.grid_size[0] - 1, self.grid_size[1] - 1)

        x = np.linspace(p0[0], p1[0], self.grid_size[0])
        y = np.linspace(p0[1], p1[1], self.grid_size[1])
        return x, y

    def ij_to_xyz(self, i: int, j: int) -> tuple[float, float, float]:
        """
        Converts grid indices to Cartesian coordinates.
        """
        x = self.corner[0] + i * self.grid_spacing
        y = self.corner[1] + j * self.grid_spacing
        return x, y, self.height

    def xy_to_ij(self, x: float, y: float) -> tuple[int, int]:
        """
        Converts Cartesian coordinates to grid indices.
        """
        i = int(np.rint((x - self.corner[0]) / self.grid_spacing))
        j = int(np.rint((y - self.corner[1]) / self.grid_spacing))
        return i, j

    def visualize(self, ax):
        """
        Visualizes the grid on the given axes.
        """

        x, y = self.get_grid()

        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                rect = plt.Rectangle(
                    (x[i] - self.grid_spacing / 2, y[j] - self.grid_spacing / 2),
                    self.grid_spacing,
                    self.grid_spacing,
                    fill=False,
                    edgecolor="black",
                    linestyle="-",
                    alpha=0.5,
                    lw=0.5,
                )
                ax.add_patch(rect)
