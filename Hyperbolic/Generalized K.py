import numpy as np
import matplotlib.pyplot as plt


def GEN_K(u, c):
    n = len(u)
    v = u.copy()
    ϵ = 0
    k = -1
    for i in range(n):
        in1 = (i - 1) % n  # index of i-1 with periodic boundary
        in2 = (i - 2) % n  # index of i-2 with periodic boundary
        ip1 = (i + 1) % n  # index of i+1 with periodic boundary
        v[i] = u[i] - c * (u[i] - u[in1]) - (ϵ * c / 4) * (
                    (1 - k) * (u[i] - 2 * u[in1] + u[in2]) + (1 + k) * (u[ip1] - 2 * u[i] + u[in1]))
    return v


# Define the grid parameters
L = 10.0  # Length of the domain in the x direction
Δx = 0.01  # Grid spacing in the x direction
Nx = int(L / Δx)  # Number of grid points in the x direction
c = 0.5  # Courant Numbers

# Define the physical parameters
α = 2  # Speed of Propagation

# Define the simulation parameters
sim_time = 4  # Total simulation time
Δt = round(c * Δx / α, 4)  # time step size
num_time_step = int(sim_time / Δt)  # Number of time steps

# Define the initial condition
x_values = np.linspace(0, L, Nx)
u = np.exp(-4 * (x_values - 5) ** 2)

# Plot the initial condition
plt.plot(x_values, u, label="Initial Condition")
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("Generalized K")
plt.legend(loc="upper left")
plt.grid(True)

# Run the simulation
for j in range(num_time_step):
    u = GEN_K(u, c)

# Numerical
plt.plot(x_values, u, label=f"After {sim_time} seconds (numerically)")

# Define the simulation parameters
sim_time_2 = 6  # Total simulation time
num_time_step_2 = int(sim_time_2 / Δt)  # Number of time steps

# Run the simulation
for j in range(num_time_step_2):
    u = GEN_K(u, c)

# Numerical
plt.plot(x_values, u, label=f"After {sim_time + sim_time_2} seconds (numerically)")

plt.legend()
plt.show()
