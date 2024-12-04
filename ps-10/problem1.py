import numpy as np

L = 10 ** -8
M = 9.109 * 10 ** -31
N = 1000.
a = L/N
x0 = L/2
h = 10 ** -18
sigma = 10 ** -10
K = 5e10
hbar = 1.054571817 * 10 ** -34

b1 = 1 - h * (1j * hbar)/(2 * M * a ** 2)
b2 = h * (1j * hbar)/(4 * M * a ** 2)

x = np.arange(0, N+1)

psi_0 = np.exp(-(x - x0) ** 2 / (2* sigma ** 2)) * np.exp(1j * K * x)

