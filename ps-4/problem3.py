import numpy as np
import matplotlib.pyplot as plt
import math as math
from scipy.integrate import fixed_quad
"""
Function created with the assistance of chatGPT.
Prompt 1: How do I write a nonrecursive hermite polynomial
Prompt 2: in python with  H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
"""
def H(n, x):
    if n == 0:
        return 1
    if n == 1:
        return 2 * x
    
    H_n_minus_2 = 1  # H(0, x)
    H_n_minus_1 = 2 * x  # H(1, x)
    
    for i in range(2, n + 1):
        H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
        H_n_minus_2 = H_n_minus_1
        H_n_minus_1 = H_n
    
    return H_n

def QuantumHarmonicOscillatorWF(x, n):
    norm = 1/np.sqrt(np.exp2(n) * math.factorial(n) * np.sqrt(np.pi))
    functional = np.exp(-(x ** 2)/2) * H(n, x)
    return norm * functional


x = np.arange(-4, 4, 0.01)
n = [0, 1, 2, 3]

fig, ax = plt.subplots()

for n in n:
    ax.plot(x, QuantumHarmonicOscillatorWF(x, n), label = f"n = {n}")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("Wavefunction")
ax.set_title("Quantum harmonic oscillator for n = 0, 1, 2, 3")

plt.savefig("ps-4/plots/QHO")

x = np.arange(-10, 10, 0.1)
n = 30
y = QuantumHarmonicOscillatorWF(x, n)

fig, ax = plt.subplots()

ax.plot(x, QuantumHarmonicOscillatorWF(x, n), label = f"n = {n}")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("Wavefunction")
ax.set_title("Quantum harmonic oscillator for n = 30")

plt.savefig("ps-4/plots/QHOLong")

N = 100
xp, wp = np.polynomial.hermite.hermgauss(N)

def Integral(n = None, z = None):
    integrand = np.exp(z ** 2) * (z ** 2 * QuantumHarmonicOscillatorWF(z, n) ** 2)
    return integrand
def func_rescale(n = None,  xp=None, range=None):
    weight = (range[1] - range[0]) * 0.5
    z = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return(weight * Integral(n = n, z = z))

def uncertainty(n, wp, xp):
    gauss_integral = (Integral(n = n, z = xp) * wp).sum()
    # range = [-np.pi/2, np.pi/2]
    # gauss_integral = (func_rescale(n = n, xp = xp, range = range) * wp).sum()
    return gauss_integral

uncertainty = np.sqrt(uncertainty(n = 5, wp = wp, xp = xp))
error = uncertainty - np.sqrt(11/2)

print(f"We get an uncertainty of: {uncertainty} for the 5th wavefunction")
print(f"This is about {error} off from the correct result")

