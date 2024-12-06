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

#Initializing A and initial wavefunction
A = np.zeros((N, N), dtype=complex)
for i in range(N-1):
    A[0, i + 1] = a2  # Upper diagonal
    A[1, i] = a1      # Main diagonal
    A[2, i] = a2      # Lower diagonal
A[1, N - 1] = a1      # Last diagonal element
psi_0 = np.exp(-((x - x0) ** 2 )/ (2* sigma ** 2)) * np.exp(1j * K * x)
# norm = np.sqrt(np.sum(psi_0 ** 2))
# psi_0 = psi_0/norm


def crank_nicholson_step(psi_0, A, up, down):
    v = np.zeros_like(psi_0, dtype = complex)

    for i in range(1, N-1):
        v[i] = b1 * psi_0[i] + b2 * (psi_0[i+1] + psi_0[i-1])

    psi = banded(A, v, up, down)
    return psi

psi = crank_nicholson_step(psi_0, A, up, down)
print(f"crank nicholson step for Psi_0: {psi}")

# Plot the real part, imaginary part, and modulus (probability density)

# plt.figure(figsize=(10, 6))

# # Plot the real part of psi_final
# plt.subplot(3, 1, 1)
# plt.plot(x, np.real(psi), label='Real part', color='blue')
# plt.title("Real part of the wavefunction")
# plt.xlabel("x")
# plt.ylabel("Real(ψ)")

# # Plot the imaginary part of psi_final
# plt.subplot(3, 1, 2)
# plt.plot(x, np.imag(psi), label='Imaginary part', color='red')
# plt.title("Imaginary part of the wavefunction")
# plt.xlabel("x")
# plt.ylabel("Imaginary(ψ)")

# # Plot the modulus (probability density)
# plt.subplot(3, 1, 3)
# plt.plot(x, np.abs(psi) ** 2, label='Probability density', color='green')
# plt.title("Probability density |ψ(x)|^2")
# plt.xlabel("x")
# plt.ylabel("|ψ(x)|^2")

# # Show the plots
# plt.tight_layout()
# plt.show()

# # Matplotlib setup
# fig, ax = plt.subplots(figsize=(8, 4))
# ax.set_xlim(0, L)
# ax.set_ylim(-1, 1)
# ax.set_xlabel("x")
# ax.set_ylabel("Re(ψ)")
# ax.set_title("Wavefunction Evolution (Real Part)")

# line, = ax.plot([], [], lw=2)

# # Initialization function
# def init():
#     line.set_data(x, np.real(psi_0))
#     return line,

# # Animation update function
# psi = psi_0.copy()  # Mutable wavefunction for updates
# def update(frame):
#     global psi
#     psi = crank_nicholson_step(psi, A, up, down)  # Update the wavefunction
#     line.set_data(x, np.real(psi))  # Update the line with the real part
#     return line,

# # Create animation
# ani = FuncAnimation(fig, update, frames=2000, init_func=init, blit=True, interval=5)

# # Display the animation
# plt.show()

# Matplotlib setup
fig = plt.figure(figsize=(10, 6))

# Subplots for real part, imaginary part, and probability density
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

# Initial plots
line_real, = ax1.plot(x, np.real(psi_0), color='blue')
ax1.set_title("Real part of the wavefunction")
ax1.set_xlabel("x")
ax1.set_ylabel("Real(ψ)")
ax1.set_ylim(-1, 1)

line_imag, = ax2.plot(x, np.imag(psi_0), color='red')
ax2.set_title("Imaginary part of the wavefunction")
ax2.set_xlabel("x")
ax2.set_ylabel("Imaginary(ψ)")
ax2.set_ylim(-1, 1)

line_prob, = ax3.plot(x, np.abs(psi_0) ** 2, color='green')
ax3.set_title("Probability density |ψ(x)|^2")
ax3.set_xlabel("x")
ax3.set_ylabel("|ψ(x)|^2")
ax3.set_ylim(0, 1)

# Animation update function
psi = psi_0.copy()  # Mutable wavefunction for updates
def update(frame):
    global psi
    psi = crank_nicholson_step(psi, A, up, down)  # Update the wavefunction
    
    # Update real part
    line_real.set_ydata(np.real(psi))
    
    # Update imaginary part
    line_imag.set_ydata(np.imag(psi))
    
    # Update probability density
    line_prob.set_ydata(np.abs(psi) ** 2)
    
    return line_real, line_imag, line_prob

# Create animation
ani = FuncAnimation(fig, update, frames=200, blit=True, interval=1)

# Show the animation
plt.tight_layout()
plt.show()
plt.savefig("ps-10/plots/crank_nicholson")