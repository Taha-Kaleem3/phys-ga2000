import numpy as np
import matplotlib.pyplot as plt

rho = 6.022 * 10 ** 28 # m^-3
theta = 428 #K
V = 1000e-6 #m
kb = 1.38 * 10 ** -23 #m^2 kg s*-2 K^-1
N = 50
xp, wp = np.polynomial.legendre.leggauss(N)
def Integral(x=None):
    return(np.exp(x) * (x ** 4))/((np.exp(x) - 1) ** 2)

def func_rescale(xp=None, range=None):
    weight = (range[1] - range[0]) * 0.5
    x = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return(weight * Integral(x=x))

def const_Cv(V, rho, theta, kb, T):
    return 9*V*rho*kb*((T/theta) ** 3)

def cv(T, wp, xp):
    constant = const_Cv(V, rho, theta, kb, T)
    range = [0, theta/T]
    gauss_integral = constant * (func_rescale(xp, range = range) * wp).sum()
    return gauss_integral

print(f"single heat capacity for T = 50 K: {cv(50, wp, xp)}")

def plot_cv(wp, xp):
    T = np.arange(5, 500)
    cV = np.zeros(len(T))

    i = 0
    for t in T:
        cV[i] = cv(T[i], wp, xp)
        i+=1
    return T, cV

T, cV = plot_cv(wp, xp)

fig, ax = plt.subplots()

ax.plot(T, cV)
plt.savefig("ps-4/plots/heatCapacity")

N_list = [10, 20, 30, 40, 50, 60, 70]

xpwp = []
Ts = []
cVs = []
for n in N_list:
    xpwp.append(np.polynomial.legendre.leggauss(n))
    T, cV = plot_cv(np.polynomial.legendre.leggauss(n), xp)
    Ts.append(T)
    cVs.append(cV)

fig, ax = plt.subplots()
for i in range(len(N_list)):
    ax.plot(Ts[i], cV[i], label = N_list[i])

ax.legend()

plt.savefig("ps-4/plots/heatCapacityConvergence")