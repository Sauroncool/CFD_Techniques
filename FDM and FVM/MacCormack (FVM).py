import numpy as np
import matplotlib.pyplot as plt


def f(u):
    return u ** 2 / 2


def predictor(u, Δt, Δx):  # FTFS Method
    v = np.copy(u)
    v[:-1] = u[:-1] - (Δt / Δx) * (f(u[1:]) - f(u[:-1]))
    return v


def corrector(u, u_prev, Δt, Δx):  # BTBS Method
    v = np.copy(u)
    v[1:] = ((u_prev[1:] + u[1:]) / 2) - (Δt / (2 * Δx)) * (f(u[1:]) - f(u[:-1]))
    return v


def MacCormack(u, Δt, Δx):
    u_prev = np.copy(u)
    u_predicted = predictor(u, Δt, Δx)
    u_corrected = corrector(u_predicted, u_prev, Δt, Δx)
    u_corrected[0] = 1.0
    u_corrected[-1] = 0.0
    return u_corrected


# Define the grid parameters
L = 5.0  # Length of the domain in the x direction
Δx = 0.025  # Grid spacing in the x direction
Nx = int(L / Δx)  # Number of grid points in the x direction

sim_time = 2.0  # Total simulation time
Δt = Δx  # time step size
num_time_step = round(sim_time / Δt)    # Number of time steps

# Define the initial condition
x_values = np.linspace(0, L, Nx)
u = np.zeros(Nx)
u[x_values < 0.25] = 1
u[(0.25 <= x_values) & (x_values <= 1.25)] = 1.25 - x_values[(0.25 <= x_values) & (x_values <= 1.25)]

# Plot the initial condition
plt.plot(x_values, u, label="Initial Condition")

# Run the simulation
for j in range(num_time_step):
    u = MacCormack(u, Δt, Δx)

# Numerical
plt.plot(x_values, u, label=f"After {sim_time} seconds (numerically)")

# Run the simulation
for j in range(num_time_step):
    u = MacCormack(u, Δt, Δx)

# Numerical
plt.plot(x_values, u, label=f"After {2 * sim_time} seconds (numerically)")

# Run the simulation
for j in range(num_time_step):
    u = MacCormack(u, Δt, Δx)

# Numerical
plt.plot(x_values, u, label=f"After {3 * sim_time} seconds (numerically)")

plt.xlabel("x")
plt.ylabel("Amplitude")
plt.title("MacCormack")
plt.legend()
plt.grid()
plt.show()
