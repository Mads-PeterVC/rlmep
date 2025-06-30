import matplotlib.pyplot as plt
from matplotlib import colors

class QPlotter:

    """
    A class to plot the Q-values for a discrete MEP environment.
    """

    def plot(self, ax, Q, env):
        X, Y = env.gridspec.get_grid()
        gs = env.gridspec.grid_spacing

        cmap = plt.colormaps.get_cmap('Blues')
        norm = colors.Normalize(Q.min(), Q.max())
        colorizer = plt.Colorizer(cmap, norm)

        for i, x in enumerate(X):
            for j, y in enumerate(Y):
                
                if (i, j) == env._terminal_state:
                    continue

                for action_index, method in enumerate(self.get_methods()):
                    xp, yp, to = method(x, y, gs)
                    ax.fill(xp, yp, facecolor=colorizer.to_rgba(Q[i, j, action_index]), alpha=1, edgecolor='black')

                    ax.text(x+to[0], y+to[1], f"{Q[i, j, action_index]:0.2f}", fontsize=6, va='center', ha='center')

    def get_down(self, x, y, gs):
        xp = [x-0.5*gs, x, x+0.5*gs, x, x-0.5*gs]
        yp = [y-0.5*gs, y-0.5*gs, y-0.5*gs, y, y-0.5*gs]
        text_offset = (0, -0.3*gs)
        return xp, yp, text_offset
    
    def get_up(self, x, y, gs):
        xp = [x-0.5*gs, x, x+0.5*gs, x, x-0.5*gs]
        yp = [y+0.5*gs, y+0.5*gs, y+0.5*gs, y, y+0.5*gs]
        text_offset = (0, 0.3*gs)
        return xp, yp, text_offset
    
    def get_left(self, x, y, gs):
        xp = [x-0.5*gs, x-0.5*gs, x-0.5*gs, x, x-0.5*gs]
        yp = [y-0.5*gs, y, y+0.5*gs, y, y-0.5*gs]
        text_offset = (-0.3*gs, 0)
        return xp, yp, text_offset
    
    def get_right(self, x, y, gs):
        xp = [x+0.5*gs, x+0.5*gs, x+0.5*gs, x, x+0.5*gs]
        yp = [y-0.5*gs, y, y+0.5*gs, y, y-0.5*gs]
        text_offset = (0.3*gs, 0)
        return xp, yp, text_offset
    
    def get_methods(self):
        return [self.get_right, self.get_left, self.get_up, self.get_down]


def plot_qvalues(ax, Q, env):
    """
    Plots the Q-values for a discrete MEP environment.

    Args:
        ax (matplotlib.axes.Axes, optional): The axes to plot on. If None, a new figure and axes are created.
        Q (np.ndarray): The Q-values to plot.
        env (DiscreteMEP): The environment containing the grid specifications.

    Returns:
        matplotlib.axes.Axes: The axes with the plotted Q-values.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    plotter = QPlotter()
    plotter.plot(ax, Q, env)

    return ax