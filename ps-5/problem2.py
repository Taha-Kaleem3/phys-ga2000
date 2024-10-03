import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad

def integrand(x, a):
    return x ** (a-1) * np.exp(-x)

x = np.arange(0, 5, 0.1)
ais2 = integrand(x, 2)
ais3 = integrand(x, 3)
ais4 = integrand(x, 4)

fig, ax = plt.subplots()
ax.plot(x, ais2, label = "a = 2")
ax.plot(x, ais3, label = "a = 3")
ax.plot(x, ais4, label = "a = 4")

ax.set_title("value of $x^{(a-1)} * e^{-x}$")
ax.set_xlabel("x")
ax.set_ylabel("$x^{(a-1)} * e^{-x}$")
ax.legend()

plt.savefig("ps-5/plots/integrand")
N = 1000
xp, wp = np.polynomial.legendre.leggauss(N)
def Integral(a = None, z = None):
    c = a-1
    x = z*c/(1-z)
    integrand = (c/((1-z) ** 2)) * np.exp(c * np.log(x) - x)
    return integrand
def func_rescale(a = None,  xp=None, range=None):
    weight = (range[1] - range[0]) * 0.5
    z = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return(weight * Integral(a = a, z = z))

def integration(a, wp, xp):
    # gauss_integral = (Integral(a = a, x = xp) * wp).sum()
    range = [0, 1]
    gauss_integral = (func_rescale(a = a, xp = xp, range = range) * wp).sum()
    return gauss_integral

print(f"Gamma(3/2) = {integration(a = 1.5, wp = wp, xp = xp)}")
# print(fixed_quad(Integral, 0, 1))
print(f"Gamma(3) = {integration(a = 3, wp = wp, xp = xp)}")
print(f"Gamma(6) = {integration(a = 6, wp = wp, xp = xp)}")
print(f"Gamma(10) = {integration(a = 10, wp = wp, xp = xp)}")