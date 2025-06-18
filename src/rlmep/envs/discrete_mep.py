import gymnasium as gym
from ase import Atoms
import numpy as np
import matplotlib.pyplot as plt
from rlmep.utils import plot_atoms, plot_cell
from ase.calculators.calculator import Calculator

class GridSpec:
    """
    A class to define the grid specifications for the discrete MEP environment.
    """

    def __init__(
        self,
        grid_size: tuple[int, int],
        grid_spacing: float,
        corner: tuple[float, float, float] | None = None,
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
        self.corner = corner if corner is not None else (0.0, 0.0, 0.0)

    def get_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the grid points in the x and y directions.
        """

        p0 = np.array(self.corner)
        p1 = self.ijk_to_xyz(self.grid_size[0] - 1, self.grid_size[1] - 1, 0)

        x = np.linspace(p0[0], p1[0], self.grid_size[0])
        y = np.linspace(p0[1], p1[1], self.grid_size[1])
        return x, y

    def ijk_to_xyz(self, i: int, j: int, k: int) -> tuple[float, float, float]:
        """
        Converts grid indices to Cartesian coordinates.
        """
        x = self.corner[0] + i * self.grid_spacing
        y = self.corner[1] + j * self.grid_spacing
        z = self.corner[2] + k * self.grid_spacing
        return x, y, z

    def xyz_to_ijk(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        """
        Converts Cartesian coordinates to grid indices.
        """
        i = int((x - self.corner[0]) / self.grid_spacing)
        j = int((y - self.corner[1]) / self.grid_spacing)
        k = int((z - self.corner[2]) / self.grid_spacing)
        return i, j, k

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

class DiscreteMEP(gym.Env):
    """
    A simple discrete MEP environment.
    """

    def __init__(
        self,
        initial_config: Atoms,
        final_config: Atoms,
        gridspec: GridSpec,
        calculator: Calculator,
        moving_atom: int = 0,
        max_steps: int | None = None,
        barrier_max: float = 2.0,
    ):
        """
        Parameters
        ----------
        initial_config : Atoms
            The initial atomic configuration.
        final_config : Atoms
            The final atomic configuration.
        moving_atom : int, optional
            The index of the atom that is moving, by default 0
        """
        super(DiscreteMEP, self).__init__()
        self.initial_config = initial_config
        self.final_config = final_config
        self.moving_atom = moving_atom
        self.gridspec = gridspec
        self.calculator = calculator
        self.max_steps = max_steps
        self.barrier_max = barrier_max

        self.action_space = gym.spaces.Discrete(n=4)
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0]), high=np.array(gridspec.grid_size), dtype=np.int32
        )
        self.reset()
        self._terminal_state = self.gridspec.xyz_to_ijk(*self.final_config.positions[self.moving_atom, :])
        self._e_initial = self.calculator.get_potential_energy(self.initial_config)

    def step(self, action: int):

        """
        Takes a step in the environment based on the action.

        Parameters
        ----------
        action : int
            The action to take.

        Returns
        -------
        tuple
            A tuple containing the next state, reward, done flag, and info.
        """
        # Define the action space
        actions = {
            0: (1, 0),  # Move right
            1: (-1, 0),  # Move left
            2: (0, 1),  # Move up
            3: (0, -1),  # Move down
        }

        dx, dy = actions[action]
        d0 = self.get_distance_to_terminal()
        self.state[0] = min(max(self.state[0] + dx, 0), self.gridspec.grid_size[0] - 1)
        self.state[1] = min(max(self.state[1] + dy, 0), self.gridspec.grid_size[1] - 1)
        d1 = self.get_distance_to_terminal()

        # Check if the new state is the terminal state
        config = self.get_current_confg()

        info = {
            'config': config,            
        }

        if tuple(self.state) == self._terminal_state:
            terminal = True
            barrier = np.min([np.max(self.history), self.barrier_max])
            reward = 1 - barrier / self.barrier_max
            info['barrier'] = barrier
        else:
            terminal = False
            config.calc = self.calculator
            E = config.get_potential_energy() - self._e_initial
            self.history.append(E)
            reward = -0.1 * (d1 - d0)  # Reward based on the change in distance to terminal state

        if self.max_steps is not None:
            if len(self.history) >= self.max_steps and not terminal:
                truncated = True
                reward = -1
            else:
                truncated = False
        else:
            truncated = False

        return self.state, reward, terminal, truncated, info

    def get_current_state(self) -> np.ndarray:
        """
        Returns the current state of the environment.
        """
        return self.state
    
    def get_current_confg(self) -> Atoms:
        """
        Returns the current atomic configuration.
        """
        config = self.initial_config.copy()
        config.positions[self.moving_atom, :] = self.gridspec.ijk_to_xyz(*self.state)
        return config

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state.
        """
        super().reset()
        self.state = list(self.gridspec.xyz_to_ijk(
            *self.initial_config.positions[self.moving_atom, :]
        ))
        self.history = []

        return self.state.copy(), {}
    
    def get_distance_to_terminal(self) -> float:
        """
        Returns the distance to the terminal state.
        """
        d = np.sum(self._terminal_state - np.array(self.state))
        return d

    def visualize(self, ax, state_history=None):
        """
        Visualizes the current state of the environment.
        """
        config = self.initial_config.copy()
        config.positions[self.moving_atom, :] = self.gridspec.ijk_to_xyz(*self.state)

        # Plot the atoms and the cell
        plot_atoms(ax, config, repeat= True in self.initial_config.pbc)
        plot_cell(ax, self.initial_config.cell, plane="xy+")

        rect = plt.Rectangle(
            (
                config.positions[self.moving_atom, 0] - self.gridspec.grid_spacing / 2,
                config.positions[self.moving_atom, 1] - self.gridspec.grid_spacing / 2,
            ),
            self.gridspec.grid_spacing,
            self.gridspec.grid_spacing,
            fill=True,
            color="red",
            linestyle="-",
            alpha=0.9,
            lw=0.5,
        )
        ax.add_patch(rect)

        # Plot the grid:
        self.gridspec.visualize(ax)

        if state_history is None:
            return

        cmap = plt.get_cmap("Reds")
        for i, state in enumerate(state_history):
            x, y, _ = self.gridspec.ijk_to_xyz(*state)
            color = cmap(i / len(state_history))
            rect = plt.Rectangle(
                    (
                        x - self.gridspec.grid_spacing / 2,
                        y - self.gridspec.grid_spacing / 2,
                    ),
                    self.gridspec.grid_spacing,
                    self.gridspec.grid_spacing,
                    fill=True,
                    color=color,
                    linestyle="-",
                    alpha=0.8,
                    lw=0.5,
            )
            ax.add_patch(rect)

def get_diffusion_env(grid_size: tuple[int, int] = (20, 20), 
                      grid_spacing: float = 0.4, 
                      max_steps: int = 100, 
                      barrier_max: float = 2.0, 
                      adsorbate_atom: str = 'Cu') -> DiscreteMEP:
    from ase.build import fcc100
    from ase.build import add_adsorbate
    from ase.calculators.emt import EMT

    # Define states:
    slab = fcc100("Cu", size=(5, 5, 2), vacuum=10.0)
    initial_state = slab.copy()
    add_adsorbate(initial_state, adsorbate_atom, height=1.5, position="hollow", offset=(1, 2))
    final_state = slab.copy()
    add_adsorbate(final_state, adsorbate_atom, height=1.5, position="hollow", offset=(2, 2))

    # Grid 
    c = (initial_state.positions[-1] + final_state.positions[-1]) / 2
    corner = (
        c[0] - grid_size[0] * grid_spacing / 2,
        c[1] - grid_size[1] * grid_spacing / 2,
        c[2],
    )
    gridspec = GridSpec(grid_size, grid_spacing, corner=corner)

    # Calculator:
    calculator = EMT()

    env = DiscreteMEP(
        initial_state, 
        final_state, 
        gridspec, 
        moving_atom=len(initial_state) - 1,
        calculator=calculator,
        max_steps=max_steps,
        barrier_max=barrier_max,
    )

    return env