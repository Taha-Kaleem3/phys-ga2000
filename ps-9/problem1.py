import numpy as np
import scipy
import matplotlib.pyplot as plt
"""
dv/dt = -w^2 x
dx/dt = v

"""
w = 1
def f(r,t):
    x = r[0]
    v = r[1]
    fx = v
    fv = - w ** 2 * x 
    return np.array([fx,fv] ,float) 
a = 0.0
b = 50.0
N = 1000
h = (b-a)/N

x0 = 1
v0 = 0

tpoints = np.arange(a,b,h)
vpoints = []
xpoints = []

r = np.array([x0, v0], float)
for t in tpoints:
    xpoints.append(r[0])
    vpoints.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6


fig, ax = plt.subplots()

ax.set_title("Harmonic Oscillator $x_0$ = 1")
ax.set_xlabel("time")

ax.plot(tpoints, xpoints, label = "Position")
ax.plot(tpoints, vpoints, label = "Velocity")
ax.legend()

plt.savefig("ps-9/plots/harmonic")


x0 = 2
v0 = 0

tpoints = np.arange(a,b,h)
v2points = []
x2points = []

r = np.array([x0, v0], float)
for t in tpoints:
    x2points.append(r[0])
    v2points.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6


fig, ax = plt.subplots()

ax.set_title("Harmonic Oscillator $x_0$ = 1 and $x_0$ = 2")
ax.set_xlabel("time")

ax.plot(tpoints, xpoints, label = "$x_0$ = 1")
ax.plot(tpoints, x2points, label = "$x_0$ = 2")
ax.legend()

plt.savefig("ps-9/plots/harmonic2initial")
"""
dv/dt = -w^2 x^3
dx/dt = v

"""
def f(r,t):
    x = r[0]
    v = r[1]
    fx = v
    fv = - w ** 2 * x ** 3 
    return np.array([fx,fv] ,float) 

a = 0.0
b = 50.0
N = 1000
h = (b-a)/N

x0 = 1
v0 = 0

tpoints = np.arange(a,b,h)
v3points = []
x3points = []

r = np.array([x0, v0], float)
for t in tpoints:
    x3points.append(r[0])
    v3points.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6


fig, ax = plt.subplots()

ax.set_title("Anharmonic Oscillator $x_0$ = 1")
ax.set_xlabel("time")

ax.plot(tpoints, x3points, label = "Position")
ax.plot(tpoints, v3points, label = "Velocity")
ax.legend()

plt.savefig("ps-9/plots/anharmonic")

x0 = 2
v0 = 0

tpoints = np.arange(a,b,h)
v4points = []
x4points = []

r = np.array([x0, v0], float)
for t in tpoints:
    x4points.append(r[0])
    v4points.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6


fig, ax = plt.subplots()

ax.set_title("Anharmonic Oscillator $x_0$ = 1 and $x_0$ = 2")
ax.set_xlabel("time")

ax.plot(tpoints, x3points, label = "$x_0$ = 1")
ax.plot(tpoints, x4points, label = "$x_0$ = 2")
ax.legend()

plt.savefig("ps-9/plots/anharmonic2initial")

fig, ax = plt.subplots()

ax.set_title("phase space plots of different oscillators")
ax.set_xlabel("x")
ax.set_ylabel("dx/dt")

ax.plot(xpoints, vpoints, label = "harmonic oscillator $x_0$ = 1")
ax.plot(x2points, v2points, label = "harmonic oscillator $x_0$ = 2")
ax.plot(x3points, v3points, label = "anharmonic oscillator $x_0$ = 1")
ax.plot(x4points, v4points, label = "anharmonic oscillator $x_0$ = 2")
ax.legend()

plt.savefig("ps-9/plots/phaseSpace")

w = 1
u = 1
def f(r,t):
    x = r[0]
    v = r[1]
    fx = v
    fv = u * v * (1 - x ** 2) - w ** 2 * x  
    return np.array([fx,fv] ,float)

 
x0 = 1
v0 = 0

a = 0.0
b = 20.0
N = 1000
h = (b-a)/N

tpoints = np.arange(a,b,h)
v5points = []
x5points = []

r = np.array([x0, v0], float)
for t in tpoints:
    x5points.append(r[0])
    v5points.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6

w = 1
u = 2
def f(r,t):
    x = r[0]
    v = r[1]
    fx = v
    fv = u * v * (1 - x ** 2) - w ** 2 * x  
    return np.array([fx,fv] ,float)

tpoints = np.arange(a,b,h)
v6points = []
x6points = []

r = np.array([x0, v0], float)
for t in tpoints:
    x6points.append(r[0])
    v6points.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6

w = 1
u = 4
def f(r,t):
    x = r[0]
    v = r[1]
    fx = v
    fv = u * v * (1 - x ** 2) - w ** 2 * x  
    return np.array([fx,fv] ,float)

tpoints = np.arange(a,b,h)
v7points = []
x7points = []

r = np.array([x0, v0], float)
for t in tpoints:
    x7points.append(r[0])
    v7points.append(r[1])

    k1 = h*f(r,t)
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6


fig, ax = plt.subplots()

ax.set_title("Van Der Pol Phase Space Plot")
ax.set_xlabel("position")
ax.set_ylabel("velocity")

ax.plot(x5points, v5points, label = "$\mu = 1$")
ax.plot(x6points, v6points, label = "$\mu = 2$")
ax.plot(x7points, v7points, label = "$\mu = 4$")

ax.legend()

plt.savefig("ps-9/plots/vanderPol")