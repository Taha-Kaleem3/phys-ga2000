import numpy as np
import matplotlib.pyplot as plt


T1= 3.053 * 60
N = 1000
mu = np.log(2)/(T1)
z = np.random.rand(N)
x = -(1/mu) * (np.log(1 - z))
y = np.arange(999, -1, -1)
x = np.sort(x)

fig, ax = plt.subplots()

ax.plot(x, y)
ax.set_xlabel("time (s)")
ax.set_ylabel("Number of  $Tl^{209}$")
ax.set_title("Decay of Thallium")

plt.savefig("ps-3/plots/EfficientDecay")