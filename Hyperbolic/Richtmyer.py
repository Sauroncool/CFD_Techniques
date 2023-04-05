import numpy as np
import matplotlib.pyplot as plt


def predictor(u, c):  # Lax Friedrich Method
    v = np.copy(u)
    v[0] = (u[1] + u[-1]) / 2 - (c / 4) * (u[1] - u[-1])
    v[1:-1] = (u[2:] + u[:-2]) / 2 - (c / 4) * (u[2:] - u[:-2])
    v[-1] = (u[0] + u[-2]) / 2 - (c / 4) * (u[0] - u[-2])
    return v


def corrector(u, u_prev, c):  # Leap Frog Method
    v = np.copy(u)
    v[0] = u_prev[0] - (c / 2) * (u[1] - u[-1])
    v[1:-1] = u_prev[1:-1] - (c / 2) * (u[2:] - u[:-2])
    v[-1] = u_prev[-1] - (c / 2) * (u[0] - u[-2])
    return v


def Richtmyer(u, c):
    u_prev = np.copy(u)
    u_predicted = predictor(u, c)
    u_corrected = corrector(u_predicted, u_prev, c)
    return u_corrected


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

# Run the simulation
for j in range(num_time_step):
    u = Richtmyer(u, c)

# Numerical
plt.plot(x_values, u, label="After {} seconds (numerically)".format(sim_time))

# Define the simulation parameters
sim_time_2 = 6  # Total simulation time
num_time_step_2 = int(sim_time_2 / Δt)  # Number of time steps

# Run the simulation
for j in range(num_time_step_2):
    u = Richtmyer(u, c)

# Numerical
plt.plot(x_values, u, label="After {} seconds (numerically)".format(sim_time + sim_time_2))

plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("Richtmyer")
plt.legend()
plt.grid()
plt.show()
