import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

m = 1
N = 20
xp, wp = np.polynomial.legendre.leggauss(N)
def Integral(a = None, x=None):
    diff = a **4 - x **4
    good = np.where(diff >0, diff, np.nan)
    return 1/np.sqrt(good)

def func_rescale(a = None, xp=None, range=None):
    weight = (range[1] - range[0]) * 0.5
    x = range[0] + 0.5 * (range[1] - range[0]) * (xp + 1.)
    return(weight * Integral(a = a, x=x))

def const_T(m):
    return np.sqrt(m*8)

def T(a, wp, xp):
    constant = const_T(m)
    range = [0, a]
    gauss_integral = constant * (func_rescale(a = a, xp = xp, range = range) * wp).sum()
    return gauss_integral

def plot_T_a():
    Ts = []
    ass = []
    for a in np.arange(0, 2, 0.01):
        ass.append(a)
        Ts.append( T(a, wp, xp))

    return ass, Ts

ass, Ts = plot_T_a()

fig, ax = plt.subplots()
ax.plot(ass, Ts)
ax.set_title("Anharmonic Oscillator")
ax.set_xlabel("amplitude")
ax.set_ylabel("Period")
plt.savefig("ps-4/plots/period")


