import numpy as np
import matplotlib.pyplot as plt


def minmod(p, q):
    if p * q > 0:
        return np.sign(p) * min(abs(p), abs(q))
    return 0


def ENO(u, Δt, Δx):
    v = np.copy(u)
    v[2:-1] = u[2:-1] - (
                (Δt * ((u[2:-1] + u[1:-2]) / 2)) * (((u[2:-1] - u[1:-2]) / Δx) + (Δx / 2) * np.vectorize(minmod)(
            (u[3:] - 2 * u[2:-1] + u[1:-2]) / (Δx ** 2), (u[2:-1] - 2 * u[1:-2] + u[:-3]) / (Δx ** 2))))
    v[0] = 1.0
    v[-1] = 0.0
    return v


# Define the grid parameters
L = 5.0  # Length of the domain in the x direction
Δx = 0.025  # Grid spacing in the x direction
Nx = int(L / Δx)  # Number of grid points in the x direction

sim_time = 2.0  # Total simulation time
Δt = Δx  # time step size
num_time_step = round(sim_time / Δt)  # Number of time steps

# Define the initial condition
x_values = np.linspace(0, L, Nx)
u = np.zeros(Nx)
u[x_values < 0.25] = 1
u[(0.25 <= x_values) & (x_values <= 1.25)] = 1.25 - x_values[(0.25 <= x_values) & (x_values <= 1.25)]

# Plot the initial condition
plt.plot(x_values, u, label="Initial Condition")

# Run the simulation
for j in range(num_time_step):
    u = ENO(u, Δt, Δx)

# Numerical
plt.plot(x_values, u, label=f"After {sim_time} seconds (numerically)")

# Run the simulation
for j in range(num_time_step):
    u = ENO(u, Δt, Δx)

# Numerical
plt.plot(x_values, u, label=f"After {2 * sim_time} seconds (numerically)")

# Run the simulation
for j in range(num_time_step):
    u = ENO(u, Δt, Δx)

# Numerical
plt.plot(x_values, u, label=f"After {3 * sim_time} seconds (numerically)")

plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("ENO")
plt.legend()
plt.grid()
plt.show()
