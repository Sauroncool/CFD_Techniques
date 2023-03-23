import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def create_banded_matrix(Nx, dx, alpha, dt):
    """
    Creates a tridiagonal banded matrix A used in the BTCS method
    """
    r = alpha * dt / dx ** 2
    diagonals = [-r, 1 + 2 * r, -r]
    offsets = [-1, 0, 1]
    A = diags(diagonals, offsets, shape=(Nx, Nx)).tocsr()

    # Apply boundary conditions
    A[0, 0] = 1
    A[0, 1] = 0
    A[-1, -1] = 1
    A[-1, -2] = 0
    return A


# Define the grid parameters
L = 1.0  # Length of the domain in the x direction (m)
dx = 0.001  # Grid spacing in the x direction (m)
Nx = int(L / dx)  # Number of grid points in the x direction

# Define the physical parameters
alpha = 6e-05  # Thermal diffusivity (m^2/s)

# Define the simulation parameters
total_time = 300  # Total simulation time (s)
time_step = 0.04  # Time step size (s)
num_time_steps = round(total_time / time_step)  # Number of time steps

# Define the initial condition
u_initial = np.zeros(Nx)
x_values = np.linspace(0, L, Nx)
u_initial = 60 * x_values + 100 * np.sin(np.pi * x_values)

# Plot the initial condition
plt.plot(x_values, u_initial, label="Initial Condition")


def T(x, t, alpha):  # Analytic Solution
    return 60 * x + 100 * (np.exp(-alpha * np.pi ** 2 * t)) * np.sin(np.pi * x)


# Run the simulation using the BTCS method with a banded matrix
A = create_banded_matrix(Nx, dx, alpha, time_step)
for i in range(1, 7):
    for j in range(1, num_time_steps):
        u_initial = spsolve(A, u_initial)
    plt.plot(x_values, u_initial, label="After {} minutes Numerically".format((i * total_time) / 60))
    plt.plot(x_values, T(x_values, 60 * 5 * i, alpha), label="After {} minutes (analytically)".format(5 * i))

plt.grid()
plt.legend()
plt.show()
