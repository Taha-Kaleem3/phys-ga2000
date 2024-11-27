

import numpy as np
import matplotlib.pyplot as plt

# Constants
R = .08  # radius in meters
m1 = 1.0  # mass 1
m2 = 500.0  # mass 2
v0 = 100.0  # initial velocity
theta = 30 * np.pi / 180  # launch angle in radians
vx0 = v0 * np.cos(theta)
vy0 = v0 * np.sin(theta)
p = 1.22  # air density
C = 0.47  # drag coefficient
g = 9.8  # gravity

# Scaled parameters
T1 = np.sqrt(m1 / (R**2 * p * C * g))
T2 = np.sqrt(m2 / (R**2 * p * C * g))
K1 = (R**2 * p * C * g * T1**2) / m1
K2 = (R**2 * p * C * g * T2**2) / m2
L1 = v0 * T1
L2 = v0 * T2
# Define function for equations of motion
def f(r, t, K):
    x, vx, y, vy = r
    speed = np.sqrt(vx**2 + vy**2)
    fx = vx
    fy = vy
    fvx = -K * speed * vx
    fvy = -g - K * speed * vy
    return np.array([fx, fvx, fy, fvy], float)

# RK4 integration
def cannonODE(K, tpoints):
    r = np.array([0.0, vx0, 0.0, vy0], float)
    xpoints, ypoints = [], []
    for t in tpoints:
        xpoints.append(r[0])
        ypoints.append(r[2])
        k1 = h * f(r, t, K)
        k2 = h * f(r + 0.5 * k1, t + 0.5 * h, K)
        k3 = h * f(r + 0.5 * k2, t + 0.5 * h, K)
        k4 = h * f(r + k3, t + h, K)
        r += (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return np.array(xpoints), np.array(ypoints)

# Time parameters
a, b = 0.0, 10.0
N = 1000
h = (b - a) / N
tpoints = np.arange(a, b, h)

# Integrate for two masses
x1, y1 = cannonODE(K1, tpoints)
x2, y2 = cannonODE(K2, tpoints)

# Plotting
fig, ax = plt.subplots()
ax.set_title("Cannonball trajectory")
ax.set_xlabel("x position (scaled)")
ax.set_ylabel("y position (scaled)")
ax.set_ylim(0, 150000)
ax.plot(x1*L1, y1*L1, label="Mass 1 (m=1)")
ax.plot(x2*L2, y2*L2, label="Mass 2 (m=2)")
ax.legend()
plt.savefig("ps-9/plots/cannon_trajectory.png")
