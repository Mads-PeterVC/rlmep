import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the reaction system: A + B -> C -> D
def reaction_odes(t, y, feed_A, feed_B, k1, k2):
    A, B, C, D = y
    rate1 = k1 * A * B
    rate2 = k2 * C
    dA_dt = feed_A - rate1
    dB_dt = feed_B - rate1
    dC_dt = rate1 - rate2
    dD_dt = rate2
    return [dA_dt, dB_dt, dC_dt, dD_dt]

# Simulate the reaction over time
def simulate_reaction(feed_A_schedule, feed_B_schedule, t_span, y0, k1=1.0, k2=0.1, dt=0.1):
    times = np.arange(t_span[0], t_span[1] + dt, dt)
    concentrations = [y0]

    current_state = np.array(y0)
    for i in range(len(times) - 1):
        t_start = times[i]
        t_end = times[i + 1]

        # Use current feed rates
        feed_A = feed_A_schedule[i]
        feed_B = feed_B_schedule[i]

        # Integrate ODEs over this small time window
        sol = solve_ivp(
            reaction_odes, [t_start, t_end], current_state,
            args=(feed_A, feed_B, k1, k2), method='RK45', t_eval=[t_end]
        )

        current_state = sol.y[:, -1]
        concentrations.append(current_state)

    return times, np.array(concentrations)

# Example feed schedules (constant feed for demo)
t_total = 10.0
dt = 0.1
n_steps = int(t_total / dt)
feed_A_schedule = np.full(n_steps, 0.1)  # mol/s
feed_B_schedule = np.full(n_steps, 0.25)  # mol/s

# Initial concentrations [A, B, C, D]
y0 = [0.0, 0.0, 0.0, 0.0]

# Run the simulation
times, concentrations = simulate_reaction(feed_A_schedule, feed_B_schedule, (0, t_total), y0)

# Plot the results
plt.figure(figsize=(10, 6))
labels = ['[A]', '[B]', '[C]', '[D]']
for i in range(4):
    plt.plot(times, concentrations[:, i], label=labels[i])
plt.xlabel('Time (s)')
plt.ylabel('Concentration (mol/L)')
plt.title('Reaction Dynamics: A + B → C → D')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
