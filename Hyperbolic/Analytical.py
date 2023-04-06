import numpy as np
import matplotlib.pyplot as plt


def Analytical(u, t, α, Δx):  # Analytic Solution
    v = u.copy()
    n = len(u)
    for i in range(n):
        v[i] = u[int(i - (α * t / Δx) % Nx)]
    return v


# Define the grid parameters
L = 10.0  # Length of the domain in the x direction
Δx = 0.01  # Grid spacing in the x direction
Nx = int(L / Δx)  # Number of grid points in the x direction
c = 0.5  # Courant Numbers

# Define the physical parameters
α = 2  # Speed of Propagataion

# Define the simulation parameters
sim_time = 4  # Total simulation time
Δt = round(c * Δx / α, 4)  # time step size
num_time_step = int(sim_time / Δt)  # Number of time steps

# Define the initial condition
x_values = np.linspace(0, L, Nx)
u = np.exp(-4 * (x_values - 5) ** 2)

# Plot the initial condition
plt.plot(x_values, u, label="Initial Condition")

# Analytical
plt.plot(x_values, Analytical(u, sim_time, α, Δx), label=f"After {sim_time} seconds (analytically)")

# Define the simulation parameters
sim_time_2 = 6  # Total simulation time
num_time_step_2 = int(sim_time_2 / Δt)  # Number of time steps
# Analytical
plt.plot(x_values, Analytical(u, sim_time + sim_time_2, α, Δx),
         label=f"After {sim_time + sim_time_2} seconds (analytically)")

plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("Analytical")
plt.legend()
plt.grid()
plt.show()
