from sympy import symbols, Function, pdsolve, pprint

# Define symbolic variables
x, t, a = symbols("x t a")

f = Function('f')
u = f(x, t)
ux = u.diff(x)
ut = u.diff(t)
pde = ut + a * ux

# Solve the PDE
sol = pdsolve(pde)
pprint(sol)
