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
        gridspec : GridSpec
            The grid specifications for the environment.
        calculator : Calculator
            The calculator to use for energy calculations.
        max_steps : int, optional
            The maximum number of steps in an episode, by default None (no limit).
        barrier_max : float, optional
            The maximum barrier height for the reward calculation, by default 2.0.
        """
        super(DiscreteMEP, self).__init__()
        self.moving_atom = moving_atom
        self.gridspec = gridspec
        self.calculator = calculator
        self.max_steps = max_steps
        self.barrier_max = barrier_max

        self._set_initial_final_states(initial_config, final_config)
        self.action_space = self.setup_action_space()
        self.observation_space = self.setup_observation_space()
        self.reset()

    def _set_initial_final_states(self, initial_config: Atoms, final_config: Atoms):
        """
        Sets the initial and final states of the environment.
        """
        self.initial_config = initial_config
        self.final_config = final_config

        self.initial_config_discrete = self.initial_config.copy()
        state = list(
            self.gridspec.xy_to_ij(
                *self.initial_config.positions[self.moving_atom, 0:2]
            )
        )
        xyz = self.gridspec.ij_to_xyz(*state)
        self.initial_config_discrete.positions[self.moving_atom, :] = xyz


        self._terminal_state = self.gridspec.xy_to_ij(
            *self.final_config.positions[self.moving_atom, 0:2]
        )
        self._e_initial = self.calculator.get_potential_energy(self.initial_config_discrete)

    def setup_action_space(self):
        return gym.spaces.Discrete(n=4)
    
    def setup_observation_space(self):
        return gym.spaces.Box(
            low=np.array([0, 0]), high=np.array(self.gridspec.grid_size), dtype=np.int32
        )

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
        # Take step: 
        next_state = self.update_state(self.state, action)
        self.state = next_state.copy()

        # Check if the new state is the terminal state
        self.update_history()
        terminal = self.check_terminal()
        reward = self.get_reward()
        truncated = self.check_truncated()

        config = self.get_current_confg()
        info = {
            "config": config,
        }

        return next_state, reward, terminal, truncated, info
    
    def get_reward(self) -> float:
        """
        Calculates the reward based on the current state and history.
        """
        if self.check_terminal():
            barrier = np.min([np.max(self.history), self.barrier_max])
            reward = 10 * (1 - barrier / self.barrier_max)
        elif self.check_truncated():
            reward = 0.0
        else:
            reward = 0.0

        return reward
    
    def update_history(self):
        """
        Updates the history of the environment with the current state.
        """
        config = self.get_current_confg()
        config.calc = self.calculator
        E = config.get_potential_energy() - self._e_initial
        self.history.append(E)
    
    def update_state(self, state: list, action: int):
        """
        Modify the current state based on the action taken - in place.

        Remember to ensure that the new state is within the grid boundaries. 
        That is, the state should not go below (0, 0) or above the grid size.
        """
        actions = {
            0: (1, 0),  # Move right
            1: (-1, 0),  # Move left
            2: (0, 1),  # Move up
            3: (0, -1),  # Move down
        }

        dx, dy = actions[action]
        state[0] = min(max(self.state[0] + dx, 0), self.gridspec.grid_size[0] - 1)
        state[1] = min(max(self.state[1] + dy, 0), self.gridspec.grid_size[1] - 1)
        return state.copy()
    
    def check_terminal(self) -> bool:
        """
        Checks if the current state is the terminal state.
        """
        terminal = tuple(self.state) == self._terminal_state
        # if max(self.history) > self.barrier_max:
        #     terminal = True
        
        return terminal
    
    def check_truncated(self) -> bool:
        """
        Checks if the episode is truncated.
        """
        truncated = False

        if self.max_steps is not None:
            if len(self.history) >= self.max_steps:
                truncated = True

        return truncated

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
        config.positions[self.moving_atom, :] = self.gridspec.ij_to_xyz(*self.state)
        return config

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[list, dict]:
        """
        Resets the environment to the initial state.

        This method should initialize the state as a list of integers representing the grid indices. 
        Additionally, it should reset the history of the environment as an empty list.
        """
        super().reset()
        self.state = list(
            self.gridspec.xy_to_ij(
                *self.initial_config.positions[self.moving_atom, 0:2]
            )
        )
        self.history = [0]

        return self.state.copy(), {}

    def visualize(
        self,
        ax,
        state_history=None,
        dx: float = 0.0,
        dy: float = 0.0,
        plot_moving: bool = True,
    ):
        """
        Visualizes the current state of the environment.
        """
        config = self.initial_config.copy()
        config.positions[self.moving_atom, :] = self.gridspec.ij_to_xyz(*self.state)

        config_for_plot = config.copy()
        if not plot_moving:
            del config_for_plot[self.moving_atom]

        # Plot the atoms and the cell
        plot_atoms(
            ax,
            config_for_plot,
            repeat=True in self.initial_config.pbc,
            radius_factor=0.95,
        )
        plot_cell(ax, self.initial_config.cell, plane="xy+")

        self._plot_position(
            ax,
            config.positions[self.moving_atom, 0],
            config.positions[self.moving_atom, 1],
            c="red",
        )

        xf, yf, _ = self.gridspec.ij_to_xyz(self._terminal_state[0], self._terminal_state[1])
        self._plot_position(
            ax, 
            xf, yf, 
            c = "green"
        )

        self._plot_state_history(ax, state_history)

        # Plot the grid:
        self.gridspec.visualize(ax)
        self._set_axis_limits(ax, dx=dx, dy=dy)

    def _set_axis_limits(self, ax, dx: float = 0.0, dy: float = 0.0):
        gs = self.gridspec.grid_spacing
        x, y = self.gridspec.get_grid()
        
        x0 = min(x) - dx - 0.5 * gs
        x1 = max(x) + 0.5 * gs + dx
        y0 = min(y) - 0.5 * gs - dy
        y1 = max(y) + 0.5 * gs + dy
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)


    def _plot_position(self, ax, x, y, c=None):
        ax.plot(x, y, color=c, marker="o", markersize=5, linestyle="None")

    def _plot_state_history(self, ax, state_history):
        if state_history is None:
            pass
        else:
            cmap = plt.get_cmap("Reds")
            for i, state in enumerate(state_history):
                x, y, _ = self.gridspec.ijk_to_xyz(*state)
                color = cmap(i / len(state_history))
                self._plot_position(ax, x, y, c=color)



def get_diffusion_env(
    grid_size: tuple[int, int] = (20, 20),
    grid_spacing: float = 0.4,
    max_steps: int = 100,
    barrier_max: float = 2.0,
    adsorbate_atom: str = "Cu",
    grid_shift: tuple[float, float] | None = None,
) -> DiscreteMEP:
    from ase.build import fcc100
    from ase.build import add_adsorbate
    from ase.calculators.emt import EMT

    # Define states:
    slab = fcc100("Cu", size=(5, 5, 2), vacuum=10.0)
    initial_state = slab.copy()
    add_adsorbate(
        initial_state, adsorbate_atom, height=1.5, position="hollow", offset=(1, 2)
    )
    final_state = slab.copy()
    add_adsorbate(
        final_state, adsorbate_atom, height=1.5, position="hollow", offset=(2, 2)
    )

    if grid_shift is None:
        grid_shift = (0.0, 0.0)

    index = len(initial_state) - 1
    dx = final_state.positions[index, 0] - initial_state.positions[index, 0]
    grid_spacing = dx / (dx // grid_spacing)

    # Grid
    c = initial_state.positions[-1]
    corner = (
        c[0] + grid_shift[0] * grid_spacing,
        c[1] + grid_shift[1] * grid_spacing,
        c[2],
    )

    gridspec = GridSpec(grid_size, grid_spacing, corner=corner, height=c[2])

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
