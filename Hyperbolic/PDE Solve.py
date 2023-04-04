from sympy import symbols, Function, pdsolve, sin, cos, exp ,pprint

# Define symbolic variables
x, t, c = symbols("x t c")

f = Function('f')
u = f(x, t)
ux = u.diff(x)
ut = u.diff(t)
pde = ut + c * ux

# Solve the PDE
sol = pdsolve(pde)
pprint(sol)
