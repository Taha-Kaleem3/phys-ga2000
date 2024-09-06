import numpy as np
import matplotlib.pyplot as plt

def gaussian(xrange, mean = 0, stddev = 1):
    """
    Returns a normalized gaussian centered at a default mean of 0 and with a default standard deviation
    of 1. xrange is a numpy array that specifies the x values we want to plot.
    """
    return np.exp(-np.power(np.subtract(xrange, mean),2)/(np.multiply(2,stddev**2)))

# plotting 50 x and y values of the gaussian
xrange = np.linspace(-10,10, 50, endpoint = True)
yrange = gaussian(xrange, mean = 0, stddev= 3)

#Plotting and saving using matplotlib
fig, ax = plt.subplots()

ax.plot(xrange, yrange)
ax.set_title("Gaussian with mean 0 and standard deviation of 3")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.savefig(fname = "C:\\Users\\kalee\\Documents\\GitHub\\phys-ga2000\\plot_gaussian")