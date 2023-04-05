import numpy as np
import matplotlib.pyplot as plt


def Analytical(x_values, t, α):  # Analytic Solution
    # Solution = np.exp(-4 * (x_values - ( 5 + α * t)) ** 2)    With Periodic BCs
    peak_point = 5 + α * t
    while peak_point >= 10:
        peak_point = peak_point - 10
    peak_point_2 = 2 + α * t
    while peak_point_2 >= 10:
        peak_point_2 = peak_point_2 - 10
    return np.exp(-(x_values - peak_point) ** 2) + np.exp(-(x_values - (peak_point + 10)) ** 2) + np.exp(
        -30 * (x_values - peak_point_2) ** 2) + np.exp(-30 * (x_values - (peak_point_2 + 10)) ** 2)
    # return np.exp(-4 * (x_values - peak_point) ** 2)+np.exp(-4 * (x_values - (peak_point+10)) ** 2)


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
# u = np.exp(-4 * (x_values - 5) ** 2)
u = np.exp(-30 * (x_values - 2) ** 2) + np.exp(-1 * (x_values - 5) ** 2)
# Plot the initial condition
plt.plot(x_values, u, label="Initial Condition")

# Analytical
plt.plot(x_values, Analytical(x_values, sim_time, α), label=f"After {sim_time} seconds (analytically)")

# Define the simulation parameters
sim_time_2 = 6  # Total simulation time
num_time_step_2 = int(sim_time_2 / Δt)  # Number of time steps
# Analytical
plt.plot(x_values, Analytical(x_values, sim_time + sim_time_2, α),
         label=f"After {sim_time + sim_time_2} seconds (analytically)")

plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("Analytical")
plt.legend()
plt.grid()
plt.show()