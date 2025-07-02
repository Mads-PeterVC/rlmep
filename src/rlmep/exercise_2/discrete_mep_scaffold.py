from typing import Callable

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from rlmep.exercise_2 import GridSpec
from rlmep.utils import plot_atoms, plot_cell


class ScaffoldDiscreteMEP(gym.Env):
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
        reward_scale: float = 10.0,
        barrier_max: float = 2.0,
        distance_parameter: float = 0.1,
        functions: dict[str, Callable] = None,        
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
        super(ScaffoldDiscreteMEP, self).__init__()
        self.moving_atom = moving_atom
        self.gridspec = gridspec
        self.calculator = calculator
        self.max_steps = max_steps
        self.barrier_max = barrier_max
        self.reward_scale = reward_scale
        self.distance_parameter = distance_parameter

        self._set_methods(functions)

        self._set_initial_final_states(initial_config, final_config)
        self.action_space = self.setup_action_space()
        self.observation_space = self.setup_observation_space()
        self.reset()

    def _set_methods(self, functions: dict[str, Callable]):
        assert isinstance(functions, dict), "functions must be a dictionary"

        required_functions = [
            "check_terminal",
            "check_truncated",
            "update_state",
            "update_atoms",
            "get_final_reward",
            "get_distance_reward",
        ]

        for func in required_functions:
            if func not in functions:
                raise ValueError(
                    f"Function '{func}' is required but not provided in 'functions'."
                )

        self._update_state = functions.get("update_state")
        self._check_terminal = functions.get("check_terminal")
        self._check_truncated = functions.get("check_truncated")
        self._update_atoms = functions.get("update_atoms")
        self._get_distance_reward = functions.get("get_distance_reward")
        self._get_final_reward = functions.get("get_final_reward")

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

        self._terminal_state = list(
            self.gridspec.xy_to_ij(*self.final_config.positions[self.moving_atom, 0:2])
        )
        self._e_initial = self.calculator.get_potential_energy(
            self.initial_config_discrete
        )

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
        reward = self.get_reward()

        terminal = self.check_terminal()
        truncated = self.check_truncated()

        config = self.get_current_confg()
        info = {
            "config": config,
            "history": self.history,
        }

        return next_state, reward, terminal, truncated, info

    def get_reward(self) -> float:
        """
        Calculates the reward based on the current state and history.
        """

        if self.check_terminal():
            reward = self._get_final_reward(delta_energies=self.history, 
                                           delta_max=self.barrier_max, 
                                           A=self.reward_scale)

        elif self.check_truncated():
            reward = 0
        else:
            reward = 0
        # Calculate the distance reward
        current_distance = self.get_terminal_distance(self.state)
        reward += self._get_distance_reward(current_distance, self.previous_distance, self.distance_parameter)
        self.previous_distance = self.get_terminal_distance(self.state)

        return reward
    
    def get_terminal_distance(self, state: list[int, int]) -> float:
        distance = np.abs((state[0] - self._terminal_state[0])) + np.abs((state[1] - self._terminal_state[1]))
        return distance            

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
        state = self._update_state(state, action, self.gridspec.grid_size)

        return state.copy()

    def check_terminal(self) -> bool:
        """
        Checks if the current state is the terminal state.
        """
        terminal = self._check_terminal(self.state, self._terminal_state, self.history, self.barrier_max)

        return terminal

    def check_truncated(self) -> bool:
        """
        Checks if the episode is truncated.
        """
        truncated = self._check_truncated(len(self.history), self.max_steps)

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
        config = self._update_atoms(
            atoms=config,
            state=self.state,
            move_index=self.moving_atom,
            grid_spec=self.gridspec,
        )

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

        self.previous_distance = self.get_terminal_distance(self.state)

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

        xf, yf, _ = self.gridspec.ij_to_xyz(
            self._terminal_state[0], self._terminal_state[1]
        )
        self._plot_position(ax, xf, yf, c="green")

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
        from rlmep.utils.colored_line import colored_line

        if state_history is None:
            pass
        else:
            X = []; Y = []; colors = []
            for i, state in enumerate(state_history):
                x, y, _ = self.gridspec.ij_to_xyz(*state)
                color = i / len(state_history)
                X.append(x)
                Y.append(y)
                colors.append(color)

            colored_line(
                X, Y, colors, ax,
                cmap=plt.get_cmap("Blues"),
                linewidth=2,
                alpha=0.9,
            )


def _get_cheat_functions():
    """
    Provide solution functions for the discrete MEP environment.
    """

    def check_terminal(state: list[int, int], terminal_state: list[int, int], history: list[float], delta_max: float) -> bool:
        """
        Check if the state is terminal.
        """
        terminal = False

        if state == terminal_state:
            terminal = True
        elif len(history) > 0 and np.max(history) > delta_max:
            terminal = True
        
        return terminal


    def check_truncated(step: int, max_steps: int) -> bool:
        """
        Check if the episode should be truncated.
        """
        # Your code here
        truncated = step >= max_steps if max_steps is not None else False
        return truncated

    def update_state(
        state: list[int, int], action: int, grid_size: tuple[int, int]
    ) -> list[int, int]:
        actions = {
            0: (1, 0),  # Move right
            1: (-1, 0),  # Move left
            2: (0, 1),  # Move up
            3: (0, -1),  # Move down
        }

        dx, dy = actions[action]
        state[0] = min(max(state[0] + dx, 0), grid_size[0] - 1)
        state[1] = min(max(state[1] + dy, 0), grid_size[1] - 1)

        return state

    def update_atoms(
        atoms: Atoms, state: list[int, int], grid_spec: GridSpec, move_index: int
    ) -> Atoms:
        """
        Update the `ase.Atoms` object based on the state and move index.

        Use the `grid_spec`-object to convert the state to coordinates.
        """
        config = atoms
        config.positions[move_index, :] = grid_spec.ij_to_xyz(*state)

        return atoms
    
    def get_distance_reward(current_distance: float, previous_distance: float, distance_parameter: float) -> float:
        if  current_distance < previous_distance:
            reward = distance_parameter
        elif current_distance > previous_distance:
            reward = -distance_parameter
        else:
            reward = 0.0
        
        return reward

    def get_final_reward(
        delta_energies: list[float], A: float, delta_max: float
    ) -> float:
        barrier = np.min([np.max(delta_energies), delta_max])
        reward = A * (1 - barrier / delta_max)
        return reward

    return {
        "check_terminal": check_terminal,
        "check_truncated": check_truncated,
        "update_state": update_state,
        "update_atoms": update_atoms,
        "get_distance_reward": get_distance_reward,
        "get_final_reward": get_final_reward,
    }