import matplotlib.pyplot as plt
from rlmep.exercise_2 import DiscreteMEP, get_diffusion_env
import gymnasium as gym

def example_figure():
    fig, axes = plt.subplots(1, 2, figsize=(2*5, 5))

    env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
    env.reset()
    axes[0].set_title("FrozenLake-v1")
    axes[0].imshow(env.render())
    axes[0].axis("off")

    axes[1].set_title("Discrete Minimum Energy Path (MEP) Environment")
    env = get_diffusion_env(grid_spacing=0.50, grid_size=(12, 7), grid_shift=(-3, -3))
    env.visualize(axes[1], dx=1, dy=1, plot_moving=False)
