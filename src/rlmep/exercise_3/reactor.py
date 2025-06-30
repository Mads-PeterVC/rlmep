import gymnasium as gym
import numpy as np
from scipy.integrate import solve_ivp

def reaction_odes(t, y, feed_A, feed_B, k1, k2):
    A, B, C, D = y
    rate1 = k1 * A * B
    rate2 = k2 * C
    dA_dt = feed_A - rate1
    dB_dt = feed_B - rate1
    dC_dt = rate1 - rate2
    dD_dt = rate2
    return [dA_dt, dB_dt, dC_dt, dD_dt]

class ReactorEnv(gym.Env):
    def __init__(self, k1=1.0, k2=0.1, dt=0.1, t_final=10.0):
        """
        Initialize the reactor environment.

        Parameters
        ----------
        k1 : float
            Rate constant for the first reaction (A + B -> C).
        k2 : float
            Rate constant for the second reaction (C -> D).
        dt : float
            Time step for the simulation.
        t_final : float
            Final time for the simulation.
        """
        super().__init__()
        self.k1 = k1
        self.k2 = k2
        self.dt = dt
        self.t_final = t_final

    def step(self, action: tuple[float, float]):
        """
        Perform a step in the environment.

        Parameters
        ----------
        action : tuple[float, float]
            Feed rates for A and B in mol/s.
        """
        feed_A, feed_B = action
        feed_A = min(max(feed_A, 0.0), 1.0)  # Ensure non-negative feed rates
        feed_B = min(max(feed_B, 0.0), 1.0)


        t_span = (0, self.dt)
        current_state = self.state.copy()

        # Integrate ODEs over the time step
        sol = solve_ivp(
            reaction_odes, t_span, current_state,
            args=(feed_A, feed_B, self.k1, self.k2), method='RK45', t_eval=[self.dt]
        )

        # Update the state with the new concentrations
        self.state = sol.y[:, -1]

        # Check if the episode is done
        terminal = False
        self.time += self.dt

        if np.allclose(self.time, self.t_final):
            terminal = True
            reward = self.state[2] - self.state[3]  # Reward based on concentration of C and D
        else:
            reward = 0.0

        info = self._get_info()

        return self.state.copy(), reward, terminal, False, info

    def _get_info(self) -> dict:
        """
        Returns additional information about the current state.
        """
        return {'t': self.time}

    def get_current_state(self) -> np.ndarray:
        """
        Returns the current state of the environment.
        """
        return self.state

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """
        Resets the environment to the initial state.
        """
        super().reset()

        # Initial concentrations [A, B, C, D]
        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        self.time = 0

        return self.state.copy(), {'t':self.time}

if __name__ == "__main__":

    # Example usage
    env = ReactorEnv(k1=1.0, k2=0.1, dt=0.1, t_final=10.0)
    state, _ = env.reset()
    states = []
    times = []

    for step in range(100):
        action = np.array([1.0, 1.0])
        state, reward, terminal, truncated, info = env.step(action)
        states.append(state.copy())
        times.append(info['t'])
        if terminal:
            print(reward, times[-1])
            break


    import matplotlib.pyplot as plt

    states = np.array(states)
    times = np.array(times)

    plt.figure(figsize=(10, 6))
    labels = ['[A]', '[B]', '[C]', '[D]']
    for i in range(4):
        plt.plot(times, states[:, i], label=labels[i])
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (mol/L)')
    plt.title('Reaction Dynamics: A + B → C → D')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
