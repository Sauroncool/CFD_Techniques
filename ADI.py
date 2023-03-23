import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Define the grid and physical parameters
Lx, Ly = 2.0, 1.0  # Length of the domain in the x and y direction (m)
dx, dy = 0.1, 0.1  # Grid spacing in the x and y direction (m)
Nx, Ny = int(Lx / dx), int(Ly / dy)  # Number of grid points in the x and y direction
alpha = 2e-04  # Thermal diffusivity (m^2/s)

# Define the simulation parameters
total_time = 150  # Total simulation time (s)
d = 0.5
dt = round(d * dx ** 2 / alpha, 4)
num_time_step = round(total_time / dt)  # Number of time steps

# Define the initial condition
u = 10 * np.ones((Nx, Ny))
u[:, 0] = 80  # Top boundary condition


def create_matrix(n, d):
    # Creates a tridiagonal banded matrix used
    diagonals = [-d, 1 + 2 * d, -d]
    offsets = [-1, 0, 1]
    matrix = diags(diagonals, offsets, shape=(n, n)).tocsr()

    # Apply boundary conditions
    matrix[0, 0] = 1
    matrix[0, 1] = 0
    matrix[-1, -1] = 1
    matrix[-1, -2] = 0
    return matrix


r1 = alpha * dt / (2 * dx ** 2)
r2 = alpha * dt / (2 * dy ** 2)

A = (create_matrix(Nx, r1))
B = create_matrix(Nx, -r1)
C = create_matrix(Ny, -r2)
D = (create_matrix(Ny, r2))

A = A.tocsc()
D = D.tocsc()


# Define the functions for the FTCS and BTCS schemes
def first_step(u):
    Nx, Ny = u.shape
    v = u.copy()
    for j in range(1, Ny - 1):  # We are preserving top and bottom layer as BCs
        b = B.dot(u[:, j])
        c = spsolve(A, u[:, j])
        v[:, j] = b + c - u[:, j]
    return v


def second_step(u):
    Nx, Ny = u.shape
    v = u.copy()
    for i in range(1, Nx - 1):  # We are preserving left and right layer as BCs
        b = C.dot(u[i, :])
        c = spsolve(D, b)
        v[i, :] = b + c - u[i, :]
    return v


# Define the function for ADI scheme
def adi_scheme(u):
    v = first_step(u)
    w = second_step(v)
    return w


# Perform the time integration using the ADI scheme
for n in range(num_time_step):
    u = adi_scheme(u)

# Visualize the final solution
plt.imshow(u.T, origin='upper', cmap='viridis')
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.colorbar()
plt.show()

# # Create a 3D plot of the final solution
fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.arange(0, Lx, dx)
y = np.arange(0, Ly, dy)
X, Y = np.meshgrid(x, y)
surf = ax.plot_surface(X, Y, u.T, cmap='viridis')
ax.set_title('3D plot')
ax.set_xlabel("x (cm)")
ax.set_ylabel("y (cm)")
ax.set_zlabel("Temperature (Celsius)")
fig.colorbar(surf)
plt.show()
