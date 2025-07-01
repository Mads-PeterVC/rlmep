from ase.io import read

from importlib.resources import files
from pathlib import Path

from ase import Atoms
import numpy as np


def relax_config(config: Atoms):
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.constraints import FixAtoms

    config = config.copy()

    constraints = FixAtoms(indices=np.arange(len(config) - 1))
    config.constraints = constraints
    config.calc = EMT()

    optimizer = BFGS(config)
    optimizer.run(fmax=0.01)

    config.constraints = None
    config.calc = None

    return config


def get_initial_config(final_config, level=0) -> Atoms:
    """
    Get the initial configuration from the final configuration.
    The initial configuration is the same as the final configuration,
    but with the last atom removed.
    """
    config = final_config.copy()

    indices = [[5, 2], [2, 4], [4, 3]]

    i, j = indices[level]
    position = (config.positions[i] + config.positions[j]) / 2
    z = np.array([0, 0, 1])
    direction = np.cross(config.positions[i] - config.positions[j], z)
    direction /= np.linalg.norm(direction)
    config.positions[-1] = position + 1.85 * direction

    return config

def get_final_cluster() -> Atoms:
    path = Path(files("rlmep.data")) / "neb.traj"
    final_config = read(path, index=-1)
    final_config.center()

    # Swap the position of the 5th atom and the last atom
    tmp_positions = final_config.positions.copy()
    final_config.positions[5] = tmp_positions[-1]
    final_config.positions[-1] = tmp_positions[5]
    final_config.constraints = None

    return final_config

def get_cluster_data(level: int = 0) -> Atoms:
    """
    Get the cluster data from the trajectory file.
    """
    path = Path(files("rlmep.data"))

    initial_config = read(path / f"initial_{level}.traj")
    final_config = read(path / "final.traj")

    return initial_config, final_config


if __name__ == "__main__":
    from ase.io import write

    final = get_final_cluster()

    write("final.traj", final)

    for level in [0, 1, 2]:
        initial = get_initial_config(final, level)
        initial = relax_config(initial)
        write(f"initial_{level}.traj", initial)

