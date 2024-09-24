import numpy as np
import matplotlib.pyplot as plt
random = np.random.default_rng()

def normal(x):
    return 1/np.sqrt(2*np.pi)*np.exp((-x ** 2)/2)

n = 1000
m = 1000
x = random.exponential(size = (n, m))
y = np.sum(x, axis = 0) / n
z = np.sqrt(n)*(y-1)

N_values = np.arange(10, n, 10)

mean_list = []
variance_list = []
skewness_list = []
kurtosis_list = []

x_gauss = np.linspace(-5, 5, 100)
y_gauss = normal(x_gauss)

fig, axes = plt.subplots()

axes.plot(x_gauss, y_gauss)
axes.hist(z, bins=30, density=True, alpha=0.6, color='r', label="Transformed Exponential Data")


plt.savefig("ps-3/plots/gauss")

skewness_done, kurtosis_done = 0, 0
def kurtosis(z):
    n = len(z)
    mean_z = np.mean(z)
    sigma_z = np.std(z)
    fourth_moment = np.sum((z - mean_z) ** 4) / n
    return fourth_moment / (sigma_z ** 4) - 3

def skew(z):
    n = len(z)
    mean_z = np.mean(z)
    sigma_z = np.std(z)
    third_moment = np.sum((z - mean_z) ** 3) / n
    return third_moment / (sigma_z ** 3)

for n in N_values:
    random = np.random.default_rng()
    x = random.exponential(size=(n, m))
    
    y = np.sum(x, axis=0) / n
    
    z = np.sqrt(n) * (y - 1)
    
    uy = np.mean(y)
    sigmaSquaredy = np.var(y)/n
    sigmay = np.sqrt(sigmaSquaredy)

    mean_list.append(uy)
    variance_list.append(sigmaSquaredy)

    skewness_list.append(skew(z))
    kurtosis_list.append(kurtosis(z))  # Fisher=True gives excess kurtosis

# Plot the results
fig, ax = plt.subplots(2, 2, figsize=(12, 8))

# Plot mean
ax[0, 0].plot(N_values, mean_list, label="Mean")
ax[0, 0].set_title("Mean vs N")
ax[0, 0].set_xlabel("N")
ax[0, 0].set_ylabel("Mean")
ax[0, 0].legend()

# Plot variance
ax[0, 1].plot(N_values, variance_list, label="Variance", color="orange")
ax[0, 1].set_title("Variance vs N")
ax[0, 1].set_xlabel("N")
ax[0, 1].set_ylabel("Variance")
ax[0, 1].legend()

# Plot skewness
ax[1, 0].plot(N_values, skewness_list, label="Skewness", color="green")
ax[1, 0].set_title("Skewness vs N")
ax[1, 0].set_xlabel("N")
ax[1, 0].set_ylabel("Skewness")
ax[1, 0].legend()

# Plot kurtosis
ax[1, 1].plot(N_values, kurtosis_list, label="Kurtosis", color="red")
ax[1, 1].set_title("Kurtosis vs N")
ax[1, 1].set_xlabel("N")
ax[1, 1].set_ylabel("Kurtosis (Excess)")
ax[1, 1].legend()

plt.tight_layout()
plt.savefig("ps-3/plots/Statistical_Quantities")

print(kurtosis_list[0], kurtosis_list[-1:], len(kurtosis_list))
print(skewness_list[0], skewness_list[-1:], len(skewness_list))