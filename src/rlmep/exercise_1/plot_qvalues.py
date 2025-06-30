import matplotlib.pyplot as plt
import numpy as np

def plot_qvalues(Q_table, env, fs=12, figsize=(6, 6), ax=None):

    if env is not None:
        desc = env.unwrapped.desc if hasattr(env.unwrapped, 'desc') else None
    else:
        desc = None
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    cmap = plt.get_cmap('Blues')

    size = Q_table.shape[0]     
    N = int(np.sqrt(size))

    for i in np.linspace(-0.5, N-0.5, N+1):
        for j in np.linspace(-0.5, N-0.5, N+1):
            ax.plot([i, i + 1], [j, j], 'k')
            ax.plot([i, i], [j, j + 1], 'k')

    for s in range(size):
        x = s % N
        y = s // N

        if desc is not None:
            try:
                char = desc[y, x].decode('utf-8')
            except: 
                char = desc[y, x]

            if char == 'H':
                ax.fill([x-0.5, x+0.5, x+0.5, x-0.5, x-0.5], [y-0.5, y-0.5, y+0.5, y+0.5, y-0.5], 'r', alpha=0.5)
                continue
            elif char == 'G':
                ax.fill([x-0.5, x+0.5, x+0.5, x-0.5, x-0.5], [y-0.5, y-0.5, y+0.5, y+0.5, y-0.5], 'g', alpha=0.5)
                continue

        if (Q_table[s] == 0).all():
            continue

        ax.plot([x-0.5, x+0.5], [y-0.5, y+0.5], 'k', alpha=0.5)
        ax.plot([x-0.5, x+0.5], [y+0.5, y-0.5], 'k', alpha=0.5)

        dc = 0.30
        ax.text(x+dc, y, f'{Q_table[s, 2]:.2f}', fontsize=fs, ha='center', va='center') # Right
        ax.text(x, y+dc, f'{Q_table[s, 1]:.2f}', fontsize=fs, ha='center', va='center') # Down 
        ax.text(x-dc, y, f'{Q_table[s, 0]:.2f}', fontsize=fs, ha='center', va='center') # Left
        ax.text(x, y-dc, f'{Q_table[s, 3]:.2f}', fontsize=fs, ha='center', va='center') # Up


        xp = [x-0.5, x, x+0.5, x, x-0.5]
        yp = [y-0.5, y-0.5, y-0.5, y, y-0.5]
        ax.fill(xp, yp, color=cmap(Q_table[s, 3]), alpha=1)

        xp = [x-0.5, x, x+0.5, x, x-0.5]
        yp = [y+0.5, y+0.5, y+0.5, y, y+0.5]
        ax.fill(xp, yp, color=cmap(Q_table[s, 1]), alpha=1)

        xp = [x-0.5, x-0.5, x-0.5, x, x-0.5]
        yp = [y-0.5, y, y+0.5, y, y-0.5]
        ax.fill(xp, yp, color=cmap(Q_table[s, 0]), alpha=1)

        xp = [x+0.5, x+0.5, x+0.5, x, x+0.5]
        yp = [y-0.5, y, y+0.5, y, y-0.5]
        ax.fill(xp, yp, color=cmap(Q_table[s, 2]), alpha=1)

    ax.set_xlim([-0.5, N-0.5])
    ax.set_ylim([N-0.5, -0.5])

    ax.set_xticks([])
    ax.set_yticks([])