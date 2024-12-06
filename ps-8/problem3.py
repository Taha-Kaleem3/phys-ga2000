import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft

# Step (a): Load the data and plot the original data
data = np.loadtxt('ps-8/dow.txt')

fig, ax = plt.subplots()

ax.plot(data, label='Original Data', color='blue')
ax.set_xlabel("Days")
ax.set_ylabel("Dow Jones Closing Value")
ax.set_title("Dow Jones Industrial Average (2006-2010)")


# Step (b): Calculate the discrete Fourier transform using rfft
fft_result = rfft(data)

# Step (c): Set all but the first 10% of the Fourier coefficients to zero
fft_10_percent = np.copy(fft_result)
cutoff_10_percent = int(0.1 * len(fft_10_percent))
fft_10_percent[cutoff_10_percent:] = 0

# Step (d): Calculate the inverse Fourier transform with the truncated array and plot it
filtered_10_percent = irfft(fft_10_percent) 

# Plot the 10% filtered data
ax.plot(filtered_10_percent, label='Filtered Data (10% Coefficients)', color='orange')

# Step (e): Set all but the first 2% of the Fourier coefficients to zero
fft_2_percent = np.copy(fft_result)
cutoff_2_percent = int(0.02 * len(fft_2_percent))
fft_2_percent[cutoff_2_percent:] = 0

# Calculate the inverse Fourier transform with the 2% truncated array
filtered_2_percent = irfft(fft_2_percent) 

# Plot the 2% filtered data
plt.plot(filtered_2_percent, label='Filtered Data (2% Coefficients)', color='green')

# Show the plot with legends
ax.legend()
ax.grid(True)
plt.savefig("ps-8/plots/DOW")
