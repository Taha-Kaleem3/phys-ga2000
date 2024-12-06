import numpy as np
from banded import banded
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# from vpython import sphere,rate
from math import cos,sin,pi
from numpy import arange
# from vpython import sphere, vector, rate, canvas
# from vpython import sphere, curve, vector, rate
import numpy as np
from scipy.fftpack import dst, idst
# Constants used
L = 10 ** -8
M = 9.109 * 10 ** -31
N = 1000
a = L/N
x0 = L/2
h = 10 ** -18
sigma = 10 ** -10
K = 5e10
hbar = 1.054571817 * 10 ** -34


#a and b parameters
b1 = 1 - h * (1j * hbar)/(2 * M * a ** 2)
b2 = h * (1j * hbar)/(4 * M * a ** 2)

a1 = 1 + h * (1j * hbar)/(2 * M * a ** 2)
a2 = -h * (1j * hbar)/(4 * M * a ** 2)

#up and down parameters
up = 1
down = 1
# grids
x = np.arange(0, L, a)

psi_0 = np.exp(-((x - x0) ** 2 )/ (2* sigma ** 2)) * np.exp(1j * K * x)

psi_0_components = [np.real(psi_0), np.imag(psi_0)]

a_k = dst(psi_0_components[0]) * a
n_k = dst(psi_0_components[1]) * a

b_k = a_k + 1j * n_k

def psi_real_t(t):
    """
    Compute the real part of the wavefunction at time t.
    """
    k = np.arange(0, N)  # Mode indices
    E_k = -(np.pi**2 * hbar**2 * k**2) / (2 * M * L**2)  # Energy levels
    coeff = (a_k * np.cos(E_k * t / hbar) - n_k * np.sin(E_k * t / hbar)) * np.sin(np.pi * k * a /N)

    psi = idst(coeff, type=1) / a
    A = np.sqrt(np.sum(psi ** 2))
    psi_norm = psi/A

    return psi_norm

# Calculate the wavefunction at t = 10^-16 s
t = 1e-18
psi_t_real = psi_real_t(t)

fig, ax = plt.subplots()

ax.plot(x, psi_t_real)
ax.set_xlabel("x (m)")
ax.set_ylabel("Real(ψ)")
ax.legend()
ax.grid()
plt.savefig("ps-10/plots/spectral_initial")

# Matplotlib setup
fig = plt.figure(figsize=(10, 6))

# Subplots for real part, imaginary part, and probability density
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(3, 1, 2)
# ax3 = fig.add_subplot(3, 1, 3)

# Initial plots
line_real, = ax1.plot(x, np.real(psi_0), color='blue')
ax1.set_title("Real part of the wavefunction")
ax1.set_xlabel("x")
ax1.set_ylabel("Real(ψ)")
ax1.set_ylim(-0.5, 0.5)

# line_imag, = ax2.plot(x, np.imag(psi_0), color='red')
# ax2.set_title("Imaginary part of the wavefunction")
# ax2.set_xlabel("x")
# ax2.set_ylabel("Imaginary(ψ)")
# ax2.set_ylim(-1, 1)

# line_prob, = ax3.plot(x, np.abs(psi_0) ** 2, color='green')
# ax3.set_title("Probability density |ψ(x)|^2")
# ax3.set_xlabel("x")
# ax3.set_ylabel("|ψ(x)|^2")
# ax3.set_ylim(0, 1)


def update(frame):
    global psi
    psi = psi_real_t(frame*t)  # Update the wavefunction
    
    # Update real part
    line_real.set_ydata(np.real(psi))
    
    # # Update imaginary part
    # line_imag.set_ydata(np.imag(psi))
    
    # # Update probability density
    # line_prob.set_ydata(np.abs(psi) ** 2)
    
    return line_real,# line_imag, line_prob

# Create animation
ani = FuncAnimation(fig, update, frames=2000, blit=True, interval=1)

# Show the animation
plt.tight_layout()
plt.show()
plt.savefig("ps-10/plots/crank_nicholson")