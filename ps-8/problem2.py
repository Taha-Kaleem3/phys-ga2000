import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

data = np.loadtxt("ps-8/piano.txt")

fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel("samples")
ax.set_ylabel("Magnitude")
ax.set_title("Piano magnitude")
ax.grid(True)
plt.savefig("ps-8/plots/piano")

fft_result = fft(data)


n = len(data)  # Number of samples
sampling_rate = 100000  # If you have a specific sampling rate, use it here
frequencies = fftfreq(n, d=1/sampling_rate)


fig, ax = plt.subplots()
ax.plot(frequencies, np.abs(fft_result))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_xlim(xmin = 0, xmax= 30000)
ax.set_title("FFT of Piano")
ax.grid(True)
plt.savefig("ps-8/plots/pianofft")

data = np.loadtxt("ps-8/trumpet.txt")

fig, ax = plt.subplots()
ax.plot(data)
ax.set_xlabel("samples")
ax.set_ylabel("Magnitude")
ax.set_title("trumpet")
ax.grid(True)
plt.savefig("ps-8/plots/trumpet")

fft_result = fft(data)


n = len(data)  # Number of samples
sampling_rate = 100000  # If you have a specific sampling rate, use it here
frequencies = fftfreq(n, d=1/sampling_rate)


fig, ax = plt.subplots()
ax.plot(frequencies, np.abs(fft_result))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_xlim(xmin = 0, xmax= 30000)
ax.set_title("FFT of Trumpet")
ax.grid(True)
plt.savefig("ps-8/plots/trumpetfft")