import numpy as np
import matplotlib.pyplot as plt

def rolling_average(x, N):
    return np.convolve(np.array(x).flatten(), np.ones((N,)) / N, mode="same")

def plot_returns(ax, returns, window=50):
    ax.plot(returns, label="Returns")
    ax.plot(rolling_average(returns, window), label=f"Returns (rolling average - {window})")
    ax.set_title("Training Returns")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.legend()
