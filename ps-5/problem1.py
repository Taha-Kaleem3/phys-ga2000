import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
dx = 0.1
def diff_central(func=None, x=None, dx=None):
    return((func(x + 0.5 * dx) - func(x - 0.5 * dx)) / dx)

def f(x):
    return 1 + (1/2) * np.tanh(2*x)
def f_jax(x):
    return 1 + (1/2) * jnp.tanh(2*x)
def dfdx(x):
    return (1/(np.cosh(2*x))) ** 2

x = np.float64(np.arange(-2, 2, dx))


dfnumerical = diff_central(func=f, x = x, dx = dx)
dfcalculated = dfdx(x)

dv_jax = jax.grad(f_jax)
dv = jax.vmap(dv_jax)(x)

fig, ax = plt.subplots()
ax.scatter(x, dfnumerical, label = "central difference", color = "b", s = 20)
ax.plot(x, dfcalculated, label = "analytic", color = "r")
ax.scatter(x, dv, label = "autodiff", color = "y", s = 10)
ax.legend()

ax.set_title("Derivative of a simple function")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")

plt.savefig("ps-5/plots/tanh")