import numpy as np
import matplotlib.pyplot as plt


def FTCS(u, dt, dx, alpha):
    v = np.copy(u)
    v[1:-1] = u[1:-1] + alpha * dt * ((u[2:] - 2 * u[1:-1] + u[:-2]) / dx ** 2)
    return v


# Define the grid parameters
L = 1.0
dx = 0.001
d = 0.4
Nx = int(L / dx)

# Define the physical parameters
alpha = 6e-05

# Define the simulation parameters
sim_time = 5 * 60
dt = round(d * dx ** 2 / alpha, 4)
num_time_step = int(sim_time / dt)

# Define the initial condition
x_values = np.linspace(0, L, Nx)
u = 60 * x_values + 100 * np.sin(np.pi * x_values)

# Plot the initial condition
plt.plot(x_values, u, label="Initial Condition")
plt.xlabel("x (m)")
plt.ylabel("Temperature (°C)")


def T(x, t, alpha):  # Analytic Solution
    return 60 * x + 100 * (np.exp(-alpha * np.pi ** 2 * t)) * np.sin(np.pi * x)


# Run the simulation
for i in range(1, 7):
    for j in range(num_time_step):
        u = FTCS(u, dt, dx, alpha)
    plt.plot(x_values, u, label="After {} minutes (numerically)".format(i * sim_time / 60))
    plt.plot(x_values, T(x_values, 60 * 5 * i, alpha), label="After {} minutes (analytically)".format(5 * i))

plt.grid()
plt.legend()
plt.show()
