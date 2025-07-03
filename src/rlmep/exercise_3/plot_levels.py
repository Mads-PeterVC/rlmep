import matplotlib.pyplot as plt

from rlmep.data import get_cluster_data
from rlmep.utils.plot_atoms import plot_atoms


def plot_levels():
    fig, axs = plt.subplots(3, 2, figsize=(4, 6))

    for level in [0, 1, 2]:
        initial_config, final_config = get_cluster_data(level=level)

        for ax, config in zip(axs[level, :], [initial_config, final_config]):
            plot_atoms(ax, config, radius_factor=0.9)

            ax.set_xlim(7, 18)
            ax.set_ylim(7, 18)

        axs[level, 0].scatter([initial_config[-1].position[0]],
                              [initial_config[-1].position[1]],
                              c='r')
        axs[level, 1].scatter([final_config[-1].position[0]],
                              [final_config[-1].position[1]],
                              c='g')

        axs[level, 0].set_title(f'Level {level}', loc='left')

    fig.tight_layout()
