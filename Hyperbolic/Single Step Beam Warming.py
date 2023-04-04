import numpy as np
import matplotlib.pyplot as plt


def BW(u, c):
    n = len(u)
    v = u.copy()
    for i in range(0, n):
        v[i] = u[i] - (c / 2) * (3 * u[i] - 4 * u[i-1] + u[i-2]) + (c ** 2 / 2) * (u[i] - 2 * u[i-1] + u[i-2])
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


def T(x_values, t, α):  # Analytic Solution
    # Solution = np.exp(-4 * (x_values - ( 5 + α * t)) ** 2)    With Periodic BCs
    peak_point = 5 + α * t
    while peak_point>10:
        peak_point = peak_point - 10
    return np.exp(-4 * (x_values - peak_point) ** 2)


# Run the simulation
for j in range(num_time_step):
    u = BW(u, c)

# Numerical
plt.plot(x_values, u, label=f"After {sim_time} seconds (numerically)")
# Analytical
plt.plot(x_values, T(x_values, sim_time, α), label=f"After {sim_time} seconds (analytically)")

# Define the simulation parameters
sim_time_2 = 6  # Total simulation time
num_time_step_2 = round(sim_time_2 / Δt)  # Number of time steps

# Run the simulation
for j in range(num_time_step_2):
    u = BW(u, c)

# Numerical
plt.plot(x_values, u, label=f"After {sim_time + sim_time_2} seconds (numerically)")
# Analytical
plt.plot(x_values, T(x_values, sim_time + sim_time_2, α), label=f"After {sim_time + sim_time_2} seconds (analytically)")

# Add plot details and show the plot
plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("Single Step Beam Warming")
plt.legend()
plt.grid()
plt.show()
